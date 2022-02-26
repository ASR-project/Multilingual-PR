import importlib
import os

import wandb
from config.hparams import Parameters
from Datasets.datamodule import BaseDataModule

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


def parse_params(parameters: Parameters) -> dict:
    wdb_config = {}
    for k,v in vars(parameters).items():
        for key,value in vars(v).items():
            wdb_config[f"{k}-{key}"]=value
    return wdb_config
