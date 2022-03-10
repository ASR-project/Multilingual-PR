import pytorch_lightning as pl
import torch
import wandb
from models.BaseModule import BaseModule
from pytorch_lightning.callbacks import (LearningRateMonitor, RichProgressBar)
from utils.agent_utils import get_artifact, get_datamodule
from utils.callbacks import AutoSaveModelCheckpoint, LogMetricsCallback, LogAudioPrediction
from utils.logger import init_logger
import re

from utils.dataset_utils import create_vocabulary, create_vocabulary2
import os 

class BaseTrainer:
    def __init__(self, config, run=None) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param
        self.feat_param = config.feat_param

        self.logger = init_logger("BaseTrainer", "INFO")

        self.logger.info('Loading artifact...')
        self.load_artifact(config.network_param, config.data_param)

        # self.logger.info(
        #     f'Create vocabulary ISO6393 : {config.data_param.ISO6393} ...')

        # config.feat_param.vocab_file, config.network_param.len_vocab = create_vocabulary(
        #     config.data_param.ISO6393,
        #     config.data_param.phoible_csv_path,
        #     eos_token=config.feat_param.eos_token,
        #     bos_token=config.feat_param.bos_token,
        #     unk_token=config.feat_param.unk_token,
        #     pad_token=config.feat_param.pad_token,
        #     word_delimiter_token=config.feat_param.word_delimiter_token,
        # )

        self.logger.info(
            f'Create vocabulary language : {config.data_param.language} ...')

        config.feat_param.vocab_file, config.network_param.len_vocab = create_vocabulary2(
            config.data_param.language,
            config.data_param.root_path_annotation,
            eos_token=config.feat_param.eos_token,
            bos_token=config.feat_param.bos_token,
            unk_token=config.feat_param.unk_token,
            pad_token=config.feat_param.pad_token,
            word_delimiter_token=config.feat_param.word_delimiter_token,
        )

        self.logger.info(f'Vocabulary file : {config.feat_param.vocab_file}')

        self.logger.info('Loading Data module...')
        self.datamodule = get_datamodule(
            config.data_param
        )

        self.logger.info('Loading Model module...')
        self.pl_model = BaseModule(
            config.network_param, config.feat_param, config.optim_param)

        self.wb_run.watch(self.pl_model.model)

    def run(self):
        if self.config.tune_lr:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
                default_root_dir=self.wb_run.save_dir,
            )
            trainer.logger = self.wb_run
            trainer.tune(self.pl_model, datamodule=self.datamodule)

        if not self.config.debug:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)
            torch.backends.cudnn.benchmark = True

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            log_every_n_steps=1,
            fast_dev_run=self.config.dev_run,
            amp_backend="apex",
            val_check_interval=self.config.val_check_interval,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            accumulate_grad_batches=self.config.accumulate_grad_batches
        )

        trainer.logger = self.wb_run

        chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\。]'

        def prepare_batch(batch):
            audio = batch["audio"]
            # tokenize the raw audio
            batch["audio"] = self.pl_model.processor([ad["array"] for ad in audio], sampling_rate=16000).input_values
            return batch
        
        self.datamodule.prepare_data()
        
        if not os.path.exists(self.datamodule.train_save_data_path):

            self.logger.info('Processing train dataset ...')

            self.datamodule.train_dataset = self.datamodule.train_dataset.map(lambda x: {"sentence": re.sub(chars_to_remove_regex, '', x["sentence"]).lower()}, num_proc=self.datamodule.config.num_proc)
            self.datamodule.train_dataset = self.datamodule.train_dataset.map(prepare_batch, batched=True, batch_size=512, num_proc=self.datamodule.config.num_proc)
            
            self.logger.info('Saving train dataset ...')
            self.datamodule._save_dataset("train")
        
        if not os.path.exists(self.datamodule.val_save_data_path):
            
            self.logger.info('Processing validation dataset ...')

            self.datamodule.val_dataset = self.datamodule.val_dataset.map(lambda x: {"sentence": re.sub(chars_to_remove_regex, '', x["sentence"]).lower()}, num_proc=self.datamodule.config.num_proc)
            self.datamodule.val_dataset = self.datamodule.val_dataset.map(prepare_batch, batched=True, batch_size=512, num_proc=self.datamodule.config.num_proc)
            
            self.logger.info('Saving validation dataset ...')
            self.datamodule._save_dataset("validation")

        trainer.fit(self.pl_model, datamodule=self.datamodule)

    def predict(self):
        if not self.config.debug:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)
            torch.backends.cudnn.benchmark = True

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            gpus=self.config.gpu,  # use all available GPU's
            max_epochs=self.config.max_epochs,  # number of epochs
            log_every_n_steps=1,
            fast_dev_run=self.config.dev_run,
            amp_backend="apex",
            val_check_interval=self.config.val_check_interval,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            accumulate_grad_batches=self.config.accumulate_grad_batches
        )

        trainer.logger = self.wb_run

        chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\。]'

        def prepare_batch(batch):
            audio = batch["audio"]
            # tokenize the raw audio
            batch["audio"] = self.pl_model.processor([ad["array"] for ad in audio], sampling_rate=16000).input_values
            return batch
        
        self.datamodule.prepare_data_test()
        
        if not os.path.exists(self.datamodule.test_save_data_path):

            self.logger.info('Processing test dataset ...')

            self.datamodule.test_dataset = self.datamodule.test_dataset.map(lambda x: {"sentence": re.sub(chars_to_remove_regex, '', x["sentence"]).lower()}, num_proc=self.datamodule.config.num_proc)
            self.datamodule.test_dataset = self.datamodule.test_dataset.map(prepare_batch, batched=True, batch_size=512, num_proc=self.datamodule.config.num_proc)
            
            self.logger.info('Saving test dataset ...')
            self.datamodule._save_dataset("test")

        
        path_model = f"{self.config.wandb_entity}/{self.config.wandb_project}/{self.config.best_model_run}:top-1"
        best_model_path = get_artifact(path_model, type="model")

        trainer.test(
            self.pl_model, self.datamodule, ckpt_path=best_model_path
        )

        return

    def load_artifact(self, network_param, data_param):
        return
        # data_param.phoneme_labels_file = get_artifact(
        #     data_param.phoneme_artifact, type="dataset")
        # network_param.weight_checkpoint = get_artifact(
        #     network_param.artifact, type="model")
        # data_param.abstract_embeddings_file = get_artifact(
        #     data_param.abstract_embeddings_artifact, type="dataset")
        # data_param.keywords_embeddings_file = get_artifact(
        #     data_param.keywords_embeddings_artifact, type="dataset")
        # data_param.keywords_file = get_artifact(
        #     data_param.keywords_artifact, type="dataset")

    def get_callbacks(self):
        callbacks = [RichProgressBar(), LearningRateMonitor(), LogMetricsCallback(), LogAudioPrediction(self.config.log_freq_audio, self.config.log_nb_audio)]
        monitor = "val/loss"
        mode = "min"
        wandb.define_metric(monitor, summary=mode)
        save_top_k = 1
        every_n_epochs = 1
        callbacks += [
            AutoSaveModelCheckpoint  # ModelCheckpoint
            (
                config=(self.network_param).__dict__,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                monitor=monitor,
                mode=mode,
                filename="epoch-{epoch:02d}-val_loss={val/loss:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                auto_insert_metric_name=False
            )
        ]  # our model checkpoint callback

        return callbacks
