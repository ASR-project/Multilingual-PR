import pytorch_lightning as pl
import torch
import wandb
from models.BaseModule import BaseModule
from pytorch_lightning.callbacks import (LearningRateMonitor, RichProgressBar)
from utils.agent_utils import get_artifact, get_datamodule
from utils.callbacks import AutoSaveModelCheckpoint, LogMetricsCallback, LogAudioPrediction
from utils.logger import init_logger

from utils.dataset_utils import create_vocabulary
import os 

class BaseTrainer:
    def __init__(self, config, run=None) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param

        logger = init_logger("BaseTrainer", "INFO")

        logger.info('Loading artifact...')
        self.load_artifact(config.network_param, config.data_param)

        logger.info(
            f'Create vocabulary ISO6393 : {config.data_param.ISO6393} ...')

        config.feat_param.vocab_file, config.network_param.len_vocab = create_vocabulary(
            config.data_param.ISO6393,
            config.data_param.phoible_csv_path,
            eos_token=config.feat_param.eos_token,
            bos_token=config.feat_param.bos_token,
            unk_token=config.feat_param.unk_token,
            pad_token=config.feat_param.pad_token,
            word_delimiter_token=config.feat_param.word_delimiter_token,
        )

        logger.info(f'Vocabulary file : {config.feat_param.vocab_file}')

        logger.info('Loading Data module...')
        self.datamodule = get_datamodule(
            config.data_param
        )

        logger.info('Loading Model module...')
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
            # val_check_interval=self.config.val_check_interval,
            # limit_val_batches = 2
        )
        trainer.logger = self.wb_run
        
        def prepare_batch(batch):
            # @TODO the dataset is a bit dirty for now
            audio = batch["audio"]

            # tokenize the raw audio
            batch["audio"] = self.pl_model.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
            batch["input_length"] = len(batch["audio"])
            
            with self.pl_model.processor.as_target_processor():
                batch["labels"] = self.pl_model.processor(batch["sentence"]).input_ids
            return batch
        
        self.datamodule.prepare_data()
        
        if not os.path.exists(self.datamodule.train_save_data_path):
            self.datamodule.train_dataset = self.datamodule.train_dataset.map(prepare_batch,num_proc = 4)
            self.datamodule.val_dataset = self.datamodule.val_dataset.map(prepare_batch,num_proc = 4)
        
        trainer.fit(self.pl_model, datamodule=self.datamodule)

    def predict(self):
        return
        # trainer = pl.Trainer(gpus=self.config.gpu)
        # best_path = f"altegrad-gnn-link-prediction/{self.config.wandb_project}/{self.config.best_model}:top-1"
        # best_model = get_artifact(best_path, type="model")

        # raw_predictions = trainer.predict(
        #     self.pl_model, self.datamodule, ckpt_path=best_model)
        # raw_predictions = torch.cat(raw_predictions, axis=0)

        # y_pred = raw_predictions.detach().cpu().numpy()
        # predictions = zip(range(len(y_pred)), y_pred)

        # with open(f"submissions/{self.config.best_model}{'-debug'*self.config.debug}.csv", "w") as pred:
        #     csv_out = csv.writer(pred)
        #     csv_out.writerow(['id', 'predicted'])
        #     for row in predictions:
        #         csv_out.writerow(row)

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
