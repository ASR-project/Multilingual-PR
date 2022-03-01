import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.loss = nn.CTCLoss()

        # Optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        logger.info(f"Optimizer : {optim_param.optimizer}, lr : {optim_param.lr}")

        # Network
        features_extractor = get_features_extractors(feat_param.network_name, feat_param)
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
        loss = self._get_loss(batch, split="train")

        # Log loss
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch, split="validation")

        # Log loss
        self.log("val/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss = self._get_loss(batch, split="test")

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

    def _get_loss(self, batch, split):
        """convenience function since train/valid/test steps are similar"""
        x = batch
        
        # TODO implement correctly
        output = self(x['array'])
        
        output_logits = output.logits

        loss = self.loss(output_logits, targets, input_lengths, target_lengths)

        return loss
