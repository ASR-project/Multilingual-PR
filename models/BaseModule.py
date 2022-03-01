import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import Wav2Vec2PhonemeCTCTokenizer
from utils.agent_utils import get_features_extractors
from utils.logger import init_logger

from models.CTC_model import CTC_model


class BaseModule(LightningModule):
    def __init__(self, network_param, feat_param, optim_param):
        """
            method used to define our model parameters
        """
        super(BaseModule, self).__init__()

        logger = init_logger("BaseModule", "INFO")

        # Loss function
        self.loss = nn.CTCLoss(blank=0, reduction="mean") # FIXME blank is variable depends on the dataset

        # Optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        logger.info(
            f"Optimizer : {optim_param.optimizer}, lr : {optim_param.lr}")

        # Tokenizer
        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py
        self.phonemes_tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_file=feat_param.vocab_file,
                                                     eos_token=feat_param.eos_token,
                                                     bos_token=feat_param.bos_token,
                                                     unk_token=feat_param.unk_token,
                                                     pad_token=feat_param.pad_token,
                                                     word_delimiter_token=feat_param.word_delimiter_token,
                                                     do_phonemize=True,
                                                     phonemizer_lang=feat_param.phonemizer_lang,
                                                     phonemizer_backend=feat_param.phonemizer_backend
                                                     )

        # Network
        features_extractor = get_features_extractors(
            feat_param.network_name, feat_param)
        logger.info(f"Features extractor : {feat_param.network_name}")

        CTC = CTC_model(network_param)

        if feat_param.weight_checkpoint != "":
            features_extractor.load_state_dict(torch.load(
                feat_param.weight_checkpoint)["state_dict"])

        if network_param.weight_checkpoint != "":
            features_extractor.load_state_dict(torch.load(
                network_param.weight_checkpoint)["state_dict"])

        self.model = nn.Sequential(
            features_extractor,
            CTC
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss = self._get_loss(batch)

        # Log loss
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss
        self.log("val/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch)

        # Log loss
        self.log("val/loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):

        x = batch
        output = self(x)
        output = torch.sigmoid(output)

        return output

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr,
                              weight_decay=self.optim_param.weight_decay)

        if self.optim_param.scheduler:
            # scheduler = LinearWarmupCosineAnnealingLR(
            #     optimizer, warmup_epochs=self.optim_param.warmup_epochs, max_epochs=self.optim_param.max_epochs
            # )
            scheduler = {"scheduler": ReduceLROnPlateau(
                optimizer, mode="min", patience=5, min_lr=5e-6
            ),
                "monitor": "val/loss"
            }

            return [[optimizer], [scheduler]]

        return optimizer

    def _get_loss(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x = batch

        output = self(x['array'])

        # FIXME 
        # process outputs
        log_probs = F.log_softmax(output, dim=2)
        input_lengths = torch.LongTensor([len(targ) for targ in targets])

        # process targets
        targets = self.phonemes_tokenizer(x['sentence']).input_ids # FIXME not phonemized
        # self.phonemes_tokenizer._tokenize(x['sentence'][0])
        target_lengths = torch.LongTensor([len(targ) for targ in targets])

        loss = self.loss(log_probs, targets, input_lengths, target_lengths)

        return loss
