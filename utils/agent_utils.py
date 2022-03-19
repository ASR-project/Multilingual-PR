import importlib
import os
import errno

import wandb
from config.hparams import Parameters
from Datasets.datamodule import BaseDataModule
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)


def get_net(network_name, network_param):
    """
    Get Network Architecture based on arguments provided
    """

    mod = importlib.import_module(f"models.{network_name}")
    net = getattr(mod,network_name)
    return net(network_param)

def get_artifact(name: str, type: str) -> str:
    """Artifact utilities
    Extracts the artifact from the name by downloading it locally>
    Return : str = path to the artifact        
    """
    if name != "" and name is not None:
        artifact = wandb.run.use_artifact(name, type=type)
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        return file_path
    else: 
        return None

def get_datamodule(data_param):
    """
    Fetch Datamodule Function Pointer
    """
    return BaseDataModule(data_param)

def get_features_extractors(features_extractors_name, params):
    """
    get features extractors
    """
    try:
        mod = importlib.import_module(f"models.FeaturesExtractors")
        net = getattr(mod, features_extractors_name)
        return net(params)
    except NotImplementedError:
        raise NotImplementedError(f'Not implemented only Wav2vec, WavLM and Hubert')

def get_model(model_name, params):
    """
    get features extractors
    """
    try:
        mod = importlib.import_module(f"models.models")
        net = getattr(mod, model_name)
        return net(params)
    except NotImplementedError:
        raise NotImplementedError(f'Not implemented only Wav2vec, WavLM and Hubert')

def parse_params(parameters: Parameters) -> dict:
    wdb_config = {}
    for k,v in vars(parameters).items():
        for key,value in vars(v).items():
            wdb_config[f"{k}-{key}"]=value
    return wdb_config

def get_progress_bar():
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "\n"
    )

def create_directory(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
