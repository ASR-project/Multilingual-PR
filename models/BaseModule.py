import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from utils.agent_utils import get_model
from utils.logger import init_logger

from itertools import chain


from torch.profiler import profile, record_function, ProfilerActivity


class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param):
        """
            method used to define our model parameters
        """
        super(BaseModule, self).__init__()

        logger = init_logger("BaseModule", "INFO")

        # Optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        logger.info(
            f"Optimizer : {optim_param.optimizer}, lr : {optim_param.lr}")

        # Tokenizer
        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py
        self.phonemes_tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_file=network_param.vocab_file,
                                                              eos_token=network_param.eos_token,
                                                              bos_token=network_param.bos_token,
                                                              unk_token=network_param.unk_token,
                                                              pad_token=network_param.pad_token,
                                                              word_delimiter_token=network_param.word_delimiter_token,
                                                              do_phonemize=False,
                                                              return_attention_mask=False,
                                                              )

        network_param.vocab_size = self.phonemes_tokenizer.vocab_size

        # Loss function
        self.loss = nn.CTCLoss(blank= self.phonemes_tokenizer.encoder[network_param.word_delimiter_token])

        # Feature_extractor
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

        logger.info(f"Features extractor : {network_param.network_name}")
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=self.phonemes_tokenizer)

        # Model
        self.model = get_model(network_param.network_name, network_param)
        logger.info(f"Model: {network_param.network_name}")

        if network_param.freeze:
            self.model.model.freeze_feature_extractor()
        
        logger.info(f"Feature extactor:{' not'*(not network_param.freeze)} freezed")

        if network_param.freeze_transformer:
            self.model.model.requires_grad_(False)
            self.model.model.lm_head.requires_grad_(True)

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)
        if loss != loss:
            print("loss is nan, model collapse, exiting")
            exit(1)
        # Log loss
        self.log("train/loss", loss, batch_size=len(preds))
        
        return {"loss": loss, "logits": logits.detach(), "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch,batch_idx)

        # Log loss
        self.log("val/loss", loss)

        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch,batch_idx)

        # Log loss
        self.log("test/loss", loss)

        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr,
                              weight_decay=self.optim_param.weight_decay)

        if self.optim_param.scheduler!=None:
            if self.optim_param.scheduler=="Cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=self.optim_param.warmup_epochs, 
                    max_epochs=self.optim_param.max_epochs,
                    warmup_start_lr=self.optim_param.warmup_start_lr,
                    eta_min=self.optim_param.eta_min
                )
            elif self.optim_param.scheduler=="StepLR":
                scheduler = StepLR(
                    optimizer,
                    step_size=self.optim_param.step_size,
                    gamma=self.optim_param.gamma
                )
            elif self.optim_param.scheduler=="MultiStepLR":
                scheduler = MultiStepLR(
                    optimizer,
                    milestones=self.optim_param.milestones,
                    gamma=self.optim_param.gamma
                )
            else:
                scheduler = {"scheduler": ReduceLROnPlateau(
                    optimizer, 
                    mode="min", 
                    patience=self.optim_param.patience, 
                    min_lr=self.optim_param.min_lr
                ),
                    "monitor": "val/loss"
                }

            return [[optimizer], [scheduler]]

        return optimizer

    def _get_outputs(self, batch, batch_idx):
        """convenience function since train/valid/test steps are similar"""
        x = batch

        # x['array'] gives the actual raw audio
        output = self(x['array']).logits  

        # process outputs
        log_probs = F.log_softmax(output, dim=-1)
        input_lengths = torch.LongTensor([len(b) for b in log_probs])
        log_probs = log_probs.permute(1, 0, 2)

        # process targets
        # extract the indices from the dictionary 
        with self.processor.as_target_processor():
            # tokenizattion but no phonemization 
            x["labels"] = self.processor(x["phonemes"]).input_ids

        target_lengths = torch.LongTensor([len(targ) for targ in  x["labels"]])
        targets = torch.Tensor(list(chain.from_iterable(x["labels"]))).int()
        
        loss = self.loss(log_probs, targets, input_lengths, target_lengths)
        
        # to compute metric and log samples
        phone_preds = self.processor.batch_decode(torch.argmax(output, dim=-1))   
        phone_targets = self.processor.batch_decode(x["labels"]) 
        
        return loss, output, phone_preds, phone_targets
