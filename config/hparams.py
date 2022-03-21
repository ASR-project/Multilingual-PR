import os
from pickle import FALSE
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional, dict_field

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim

################################## Global parameters ##################################

@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb
    wandb_entity    : str          = "asr-project"         # name of the project
    debug           : bool         = False            # test code before running, if testing, no checkpoints are written
    test: bool = True
    wandb_project: str = f"{'test-'*test}asr"
    root_dir        : str          = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the whole run
    gpu         : int = 1  # number or gpu
    max_epochs  : int = -1  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")

    # modes
    tune_lr: bool  = False  # tune the model on first run
    dev_run: bool  = False
    train   : bool = True

    best_model: str = ""
    
    log_freq_audio : int = 1
    log_nb_audio   : int = 2

    # trainer params
    val_check_interval: float = 1.0 # 1.0 (at the end of the epoch)
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    enable_progress_bar: bool = True

    # testing params
    best_model_run: str = "Wav2Vec2_it"

    # Early Stopping
    early_stopping: bool = True
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(monitor="val/per", patience=60, mode="min", verbose=True)
    )

@dataclass
class NetworkParams:
    network_name                  : str           = "Wav2Vec2"     # Hubert, Wav2Vec2, WavLM
    pretrained_name                    : Optional[str] = ""

    freeze                        : bool          = True

    # Phoneme Tokenizer
    eos_token                     : str           = "</s>"
    bos_token                     : str           = "<s>"
    unk_token                     : str           = "<unk>"
    pad_token                     : str           = "<pad>"
    word_delimiter_token          : str           = "|"

@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """
    # Hugging Face datasets parameters
    dataset_name            : str                     = "common_voice"    # https://huggingface.co/mozilla-foundation or https://huggingface.co/datasets/common_voice # dataset, use <Dataset>Eval for FT
    use_auth_token          : bool                    = False             # True if use mozilla-foundation datasets
    subset                  : str                     = "zh-TW"              # chosen language (see https://huggingface.co/datasets/common_voice)
    download_mode           : str                     = "reuse_dataset_if_exists"
    cache_dir               : str                     = osp.join(os.getcwd(), "assets")

    # to create vocabulary of phonemes
    language                 : str                     = "zh" 
    root_path_annotation     : str                     = osp.join(os.getcwd(), "assets", "common_voices_splits")
    phoible_csv_path        : str                     = osp.join(os.getcwd(), "assets")

    # Dataloader parameters
    num_workers             : int                     = 20         # number of workers for dataloaders
    batch_size              : int                     = 2 
    
    # Dataset processing parameters
    max_input_length_in_sec : float                   = 5
    num_proc                : int                     = 4

    create_dataset        : bool                    = False 

@dataclass
class OptimizerParams: 
    """Optimization parameters"""

    optimizer     : str   = "AdamW"  # Optimizer default vit: AdamW, default resnet50: Adam
    # lr            : float = 3e-5     # learning rate,               default = 5e-4
    lr            : float = 3e-5
    min_lr        : float = 5e-9     # min lr reached at the end of the cosine schedule
    weight_decay  : float = 1e-8

    accumulate_grad_batches: int = 16 # 1 for no accumulation

    # Scheduler parameters
    scheduler     : bool  = True

@dataclass
class Parameters:
    """base options."""
    hparams       : Hparams           = Hparams()
    data_param    : DatasetParams     = DatasetParams()
    network_param : NetworkParams     = NetworkParams()
    optim_param   : OptimizerParams   = OptimizerParams()
    
    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)

        self.hparams.wandb_project = f"{'test-'*self.hparams.test}asr"

        self.network_param.phonemizer_lang = self.data_param.language
        print(f'Phonemizer language : {self.network_param.phonemizer_lang }')

        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

        if self.network_param.pretrained_name == "":
            if self.network_param.network_name == "Wav2Vec2":
                # self.network_param.pretrained_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
                self.network_param.pretrained_name = "facebook/wav2vec2-base-960h"
            elif self.network_param.network_name == "WavLM":
                self.network_param.pretrained_name = "microsoft/wavlm-base"
            elif self.network_param.network_name == "Hubert":
                self.network_param.pretrained_name = "facebook/hubert-large-ls960-ft"
            else:
                raise NotImplementedError("Only Wav2Vec2, WavLM and Hubert are available !")
        print(f"Pretrained model: {self.network_param.pretrained_name}")

        self.data_param.wandb_project = self.hparams.wandb_project
        self.hparams.accumulate_grad_batches = self.optim_param.accumulate_grad_batches

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance