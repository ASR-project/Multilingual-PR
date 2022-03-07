from torchmetrics import Metric
import torch
from torch import Tensor, tensor
from typing import Any, Dict, List, Optional, Union, Tuple

class PhonemeErrorRate(Metric):
    '''
    https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/text/wer.py#L23-L93
    '''
    def __init__(self, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds, targets):
        '''
        preds : list of sentence phoneme
        targets : list of sentence phoneme
        '''
        errors, total = _per_update(preds, targets)

        self.errors += errors
        self.total += total

    def compute(self):
        return _per_compute(self.errors, self.total)


def _per_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:
    """Update the wer score with the current set of references and predictions.
    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred.split()
        tgt_tokens = tgt.split()
        errors += _edit_distance(pred_tokens, tgt_tokens)
        total += len(tgt_tokens)
    return errors, total


def _per_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the word error rate.
    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references
    Returns:
        Word error rate score
    """
    return errors / total


def _edit_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    """Standard dynamic programming algorithm to compute the edit distance.
    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]