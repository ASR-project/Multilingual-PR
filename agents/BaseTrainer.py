import pytorch_lightning as pl
import torch
import wandb
from datasets import load_dataset
from models.BaseModule import BaseModule
from pytorch_lightning.callbacks import (LearningRateMonitor, RichProgressBar,
                                         StochasticWeightAveraging)
from utils.agent_utils import (get_artifact, get_datamodule,
                               get_features_extractor)
from utils.callbacks import AutoSaveModelCheckpoint
from utils.logger import init_logger


class BaseTrainer:
    def __init__(self, config, run=None) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param

        logger = init_logger("BaseTrainer", "INFO")

        if self.config.compute_features:
            features_extractor = get_features_extractor(config.feat_param.network_name, config)
            dataset = load_dataset(config.data_param.dataset_name, 
                                    config.data_param.subset,
                                    split=config.feat_param.split,
                                    use_auth_token=config.data_param.use_auth_token, 
                                    download_mode=config.data_param.download_mode, 
                                    cache_dir=config.data_param.cache_dir)
            path_features = features_extractor.extract_features(dataset)
            features_extractor.push_artifact(path_features)
        else:
            logger.info('Loading artifact...')
            self.load_artifact(config.network_param, config.data_param)

            logger.info('Loading Data module...')
            self.datamodule = get_datamodule(
                config.data_param
            )
            # config.network_param.input_size = self.datamodule.dataset.get_input_dim()
            logger.info('Loading Model module...')
            self.pl_model = BaseModule(
                config.network_param, config.optim_param)

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
            amp_backend="apex"
        )
        trainer.logger = self.wb_run
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
        # network_param.weight_checkpoint = get_artifact(
        #     network_param.artifact, type="model")
        # data_param.abstract_embeddings_file = get_artifact(
        #     data_param.abstract_embeddings_artifact, type="dataset")
        # data_param.keywords_embeddings_file = get_artifact(
        #     data_param.keywords_embeddings_artifact, type="dataset")
        # data_param.keywords_file = get_artifact(
        #     data_param.keywords_artifact, type="dataset")

    def get_callbacks(self):
        callbacks = [RichProgressBar(), LearningRateMonitor(),
                     StochasticWeightAveraging()]
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
