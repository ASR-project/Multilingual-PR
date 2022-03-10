import torch.nn as nn
from transformers import (HubertForCTC,
                          Wav2Vec2ForCTC,
                          WavLMForCTC)


class BaseModel(nn.Module):
    """
        BaseFeaturesExtractor class that will extract features according to the type of model
        https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        outputs = self.model(x)
        return outputs

class Wav2Vec2(BaseModel):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """

    def __init__(self, params):
        super().__init__(params)

        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(in_features=in_features, out_features=self.params.vocab_size)

class WavLM(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/wavlm#transformers.WavLMForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(in_features=in_features, out_features=self.params.vocab_size)

class Hubert(BaseModel):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#transformers.HubertForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(in_features=in_features, out_features=self.params.vocab_size)
