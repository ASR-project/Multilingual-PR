import os
from pickle import FALSE
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional

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
    max_epochs  : int = 30  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")

    # modes
    tune_lr: bool  = False  # tune the model on first run
    dev_run: bool  = False
    train   : bool = True

    best_model: str = ""
    
    log_freq_audio : int = 1
    log_nb_audio   : int = 2

    # trainer params
    val_check_interval: float = 0.5 # 1.0 (at the end of the epoch)
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    accumulate_grad_batches: int = 32 # 1 for no accumulation

    # testing params
    best_model_run: str = "Wav2Vec2_it"

    # TODO add backbone name -> use the one pre-trained on English only

@dataclass
class NetworkParams:
    network_name                  : str           = "Hubert"     # HuBERT, Wav2vec, WavLM
    pretrained                    : str           = ""

    freeze                        : bool          = False

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
    subset                  : str                     = "sv-SE"              # chosen language (see https://huggingface.co/datasets/common_voice)
    download_mode           : str                     = "reuse_dataset_if_exists"
    cache_dir               : str                     = osp.join(os.getcwd(), "assets")

    # to create vocabulary of phonemes
    # ISO6393                 : str                     = "jpn"    # look at the phoible.csv file https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv
    # phoible_csv_path        : str                     = osp.join(os.getcwd(), "assets")
    language                 : str                     = "sv" 
    root_path_annotation     : str                     = osp.join(os.getcwd(), "assets", "common_voices_splits")

    # Dataloader parameters
    num_workers             : int                     = 20         # number of workers for dataloaders
    batch_size              : int                     = 1 
    
    # Dataset processing parameters
    max_input_length_in_sec : float                   = 5
    num_proc                : int                     = 4

    recreate_dataset        : bool                    = False

    # dataset artifact TODO

@dataclass
class OptimizerParams: 
    """Optimization parameters"""

    optimizer     : str   = "AdamW"  # Optimizer default vit: AdamW, default resnet50: Adam
    # lr            : float = 3e-5     # learning rate,               default = 5e-4
    lr            : float = 3e-3
    min_lr        : float = 5e-9     # min lr reached at the end of the cosine schedule
    weight_decay  : float = 1e-8

    # Scheduler parameters
    scheduler     : bool  = True
    warmup_epochs : int   = 5
    max_epochs    : int   = 20

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

        if self.network_param.pretrained == "":
            if self.network_param.network_name == "Wav2Vec2":
                self.network_param.pretrained = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            elif self.network_param.network_name == "WavLM":
                self.network_param.pretrained = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
            elif self.network_param.network_name == "Hubert":
                self.network_param.pretrained = "facebook/hubert-large-ls960-ft"
            else:
                raise NotImplementedError("Only Wav2vec2, WavLM and Hubert are available !")
        print(f"Pretrained model: {self.network_param.pretrained}")

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance