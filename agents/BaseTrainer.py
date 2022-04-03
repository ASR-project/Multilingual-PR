import pytorch_lightning as pl
import torch
import wandb
from models.BaseModule import BaseModule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
)
from utils.agent_utils import get_artifact, get_datamodule
from utils.callbacks import (
    AutoSaveModelCheckpoint,
    LogMetricsCallback,
    LogAudioPrediction,
)
from utils.logger import init_logger

from utils.dataset_utils import create_vocabulary, create_vocabulary2


class BaseTrainer:
    def __init__(self, config, run=None) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param

        self.logger = init_logger("BaseTrainer", "INFO")

        self.logger.info(
            f"Create vocabulary language : {config.data_param.language} ..."
        )

        if config.data_param.subset == "en":
            (
                config.network_param.vocab_file,
                config.network_param.len_vocab,
            ) = create_vocabulary(
                config.data_param.language,
                config.data_param.phoible_csv_path,
                eos_token=config.network_param.eos_token,
                bos_token=config.network_param.bos_token,
                unk_token=config.network_param.unk_token,
                pad_token=config.network_param.pad_token,
                word_delimiter_token=config.network_param.word_delimiter_token,
            )
        else:
            (
                config.network_param.vocab_file,
                config.network_param.len_vocab,
            ) = create_vocabulary2(
                config.data_param.language,
                config.data_param.root_path_annotation,
                eos_token=config.network_param.eos_token,
                bos_token=config.network_param.bos_token,
                unk_token=config.network_param.unk_token,
                pad_token=config.network_param.pad_token,
                word_delimiter_token=config.network_param.word_delimiter_token,
            )

        self.logger.info(f"Vocabulary file : {config.network_param.vocab_file}")

        self.logger.info("Loading Data module...")
        self.datamodule = get_datamodule(config.data_param)

        self.logger.info("Loading Model module...")
        self.pl_model = BaseModule(config.network_param, config.optim_param)

        self.wb_run.watch(self.pl_model.model)

    def run(self):
        if self.config.tune_lr:
            tune_lr_trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
                default_root_dir=self.wb_run.save_dir,
            )
            tune_lr_trainer.logger = self.wb_run

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
            enable_progress_bar=self.config.enable_progress_bar,
            val_check_interval=self.config.val_check_interval,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
        )

        trainer.logger = self.wb_run

        self.datamodule.load_data("train")
        self.datamodule.process_dataset("train", self.pl_model.processor)

        self.datamodule.load_data("val")
        self.datamodule.process_dataset("val", self.pl_model.processor)

        if self.config.tune_lr:
            tune_lr_trainer.tune(self.pl_model, datamodule=self.datamodule)

        trainer.fit(self.pl_model, datamodule=self.datamodule)

    @torch.no_grad()
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
            log_every_n_steps=1,
            fast_dev_run=self.config.dev_run,
            amp_backend="apex",
            enable_progress_bar=self.config.enable_progress_bar,
        )

        trainer.logger = self.wb_run

        self.datamodule.load_data("test")
        self.datamodule.process_dataset("test", self.pl_model.processor)

        path_model = f"{self.config.wandb_entity}/{self.config.wandb_project}/{self.config.best_model_run}:top-1"
        best_model_path = get_artifact(path_model, type="model")

        trainer.test(self.pl_model, self.datamodule, ckpt_path=best_model_path)

        return

    def get_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            LogMetricsCallback(),
            LogAudioPrediction(self.config.log_freq_audio, self.config.log_nb_audio),
        ]

        if self.config.enable_progress_bar:
            callbacks += [RichProgressBar()]

        if self.config.early_stopping:
            callbacks += [EarlyStopping(**self.config.early_stopping_params)]

        monitor = "val/per"
        mode = "min"
        wandb.define_metric(monitor, summary=mode)
        save_top_k = 1
        every_n_epochs = 1
        callbacks += [
            AutoSaveModelCheckpoint(  # ModelCheckpoint
                config=(self.network_param).__dict__,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                monitor=monitor,
                mode=mode,
                filename="epoch-{epoch:02d}-val_per={val/per:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{str(wandb.run.name)}",
                save_top_k=save_top_k,
                every_n_epochs=every_n_epochs,
                auto_insert_metric_name=False,
            )
        ]  # our model checkpoint callback

        return callbacks
