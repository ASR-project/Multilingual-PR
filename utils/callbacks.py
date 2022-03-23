from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import wandb
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import _METRIC, _PATH, STEP_OUTPUT

from utils.metrics import MetricsModule


class AutoSaveModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, project, entity, dirpath: Optional[_PATH] = None, filename: Optional[str] = None, monitor: Optional[str] = None, verbose: bool = False, save_last: Optional[bool] = None, save_top_k: int = 1, save_weights_only: bool = False, mode: str = "min", auto_insert_metric_name: bool = True, every_n_train_steps: Optional[int] = None, train_time_interval: Optional[timedelta] = None, every_n_epochs: Optional[int] = None, save_on_train_epoch_end: Optional[bool] = None, every_n_val_epochs: Optional[int] = None):
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode, auto_insert_metric_name,
                         every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end, every_n_val_epochs)
        self.config = config
        self.project = project
        self.entity = entity

    def _update_best_and_save(
        self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, _METRIC]
    ) -> None:
        k = len(self.best_k_models) + \
            1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(
                self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(
            self.best_k_models, key=self.best_k_models.get)
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates.get("epoch")
            step = monitor_candidates.get("step")
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:0.5f}"
                f' (best {self.best_model_score:0.5f}), saving model to "{filepath}" as top {k}'
            )
        trainer.save_checkpoint(filepath, self.save_weights_only)

        if del_filepath is not None and filepath != del_filepath:
            trainer.training_type_plugin.remove_checkpoint(del_filepath)

        reverse = False if self.mode == "min" else True
        score = sorted(self.best_k_models.values(), reverse=reverse)
        # indices = [(i+1) for i, x in enumerate(score) if x == current]
        self.alias = f"latest"                      #
        self.name = f"{wandb.run.name}"
        
        self.filepath = filepath

    
    def log_artifact(self):
        rank_zero_info(f"Logging artifact")

        api = wandb.Api(overrides={"project": self.project, "entity": self.entity})
        model_artifact = wandb.Artifact(
            type="model",
            name=self.name,
            metadata=self.config
        )
        
        
        
        
        
        model_artifact.add_file(self.filepath)
        wandb.log_artifact(model_artifact, aliases=[self.alias])
        model_artifact.wait()
        rank_zero_info(f"Done. Saved '{self.name}' weights to wandb")
        rank_zero_info(f"Cleaning up artifacts")
        artifacts = []
        for art in list(api.artifact_versions("model", self.name)):
            try:
                per = art.logged_by().summary.get("val/per", 0)
                artifacts.append((art,per))
            except: 
                pass
            
        artifacts = sorted(artifacts,key=lambda art: art[-1]['min'], reverse=False)
        
        for i,artifact in enumerate(artifacts):
            artifact[0].aliases = [f'top-{i+1}']
            try:
                artifact[0].save()
            except: 
                pass
        
        rank_zero_info(f"Done")
        
        
        
    def del_artifacts(self):
        api = wandb.Api(
            overrides={"project": self.project, "entity": self.entity})
        artifact_type, artifact_name = "model", f"{wandb.run.name}"
        try:
            for version in api.artifact_versions(artifact_type, artifact_name):
                # Clean previous versions with the same alias, to keep only the latest top k.
                if len(version.aliases) == 0:  # this means that it does not have the latest alias
                    # either this works, or I will have to remove the model with the alias first then log the next
                    version.delete()
        except:
            print("error in del artifact to ignore")
            return

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        self.log_artifact()
        return super().on_exception(trainer, pl_module, exception)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_artifact()


class LogMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device

        self.metrics_module_train = MetricsModule(
            "train", device
        )

        self.metrics_module_validation = MetricsModule(
            "val", device
        )
    
    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device

        self.metrics_module_test = MetricsModule(
            "test", device
        )


    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        self.metrics_module_train.update_metrics(outputs['preds'], outputs['targets'])

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train/", pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        self.metrics_module_validation.update_metrics(outputs['preds'], outputs['targets'])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val/", pl_module)
    
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        self.metrics_module_test.update_metrics(outputs['preds'], outputs['targets'])

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_test.log_metrics("test/", pl_module)

class LogAudioPrediction(Callback):
    def __init__(self, log_freq_audio, log_nb_audio) -> None:
        super().__init__()
        self.log_freq_audio = log_freq_audio
        self.log_nb_audio = log_nb_audio

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_audio == 0:
            self.log_audio(pl_module, "val", batch, self.log_nb_audio, outputs, trainer.datamodule.sampling_rate)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_audio == 0:
            self.log_audio(pl_module, "train", batch, self.log_nb_audio, outputs, trainer.datamodule.sampling_rate)
    
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the test batch ends."""

        if batch_idx == 0 and pl_module.current_epoch % self.log_freq_audio == 0:
            self.log_audio(pl_module, "test", batch, self.log_nb_audio, outputs, trainer.datamodule.sampling_rate)

    def log_audio(self, pl_module, name, batch, n, outputs, sampling_rate):
        x = batch
        audios = x['array'][:n].detach().cpu()

        samples = []
        for i in range(len(audios)):
            samples.append([
                wandb.Audio(
                    audios[i], sample_rate=sampling_rate
                ),
                x['sentence'][i], outputs['targets'][i], outputs['preds'][i]]
            )
        
        columns= ["Audio sample", "sentence", "target", "prediction"]
        table = wandb.Table(data=samples,columns=columns)

        wandb.run.log({f"{name}/predictions":table})