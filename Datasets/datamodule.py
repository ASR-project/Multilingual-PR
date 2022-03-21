import os.path as osp
import pickle
import re
import shutil

import numpy as np
import utils.agent_utils as ag_u
import wandb
from datasets import Audio, load_dataset
from librosa.effects import trim
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.constant import CHARS_TO_REMOVE_REGEX
from utils.dataset_utils import coll_fn
from utils.logger import init_logger


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()

        self.config = dataset_param
        self.logger = init_logger("BaseDataModule", "INFO")
        self.logger.info(
            f"Loading Dataset : {self.config.dataset_name}, language : {self.config.subset}")


    def prepare_data(self) -> None:
        return super().prepare_data()


    def load_data(self, split) -> None:
        '''
            Function to load dataset
        '''

        self.logger.info(f"Loading the dataset in  load_data: {split}")

        setattr(self, f"{split}_save_data_path", osp.join("assets", "datasets", f"{split}_{self.config.dataset_name}-{self.config.subset}"))

        save_path = getattr(self, f"{split}_save_data_path")
        name_file = getattr(self, f'{split}_save_data_path').split('/')[-1]
        name_file_path = osp.join(save_path, name_file)
        name_dataset = f"{split}_dataset"

        ag_u.create_directory(save_path)

        if not osp.exists(name_file_path) or self.config.create_dataset:
            # if not self.config.create_dataset:
            # # try
            #     path = f"asr-project/{self.config.wandb_project}/{name_file}:latest"
            #     self.logger.info(f"Try loading {path} in artifacts ...")

            #     file = ag_u.get_artifact(path, type="dataset")
                
            #     shutil.copy2(file, save_path)

            #     self.logger.info(f"Load {path} in artifacts OK")
                
            #     file = open(name_file_path, "rb")
            #     setattr(self, name_dataset, pickle.load(file))
            #     self.logger.info(
            #         f"Loaded {split} dataset : {name_file_path}")
            # else:
            # except
            setattr(self, name_dataset, load_dataset(self.config.dataset_name,
                                                    self.config.subset,
                                                    split=split if split!="val" else "validation",
                                                    use_auth_token=self.config.use_auth_token,
                                                    download_mode=self.config.download_mode,
                                                    cache_dir=self.config.cache_dir
                                                    )
                    )

            setattr(self, name_dataset, getattr(self, name_dataset).remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]))

            setattr(self, name_dataset, getattr(self, name_dataset).cast_column(
                "audio", Audio(sampling_rate=16000)))

            metadata_artifact = {
                                "dataset_name": self.config.dataset_name,
                                "subset": self.config.subset,
                                "split": split,
                                "sampling_rate": 16000,
                                }

            self._save_dataset(split, name_file_path, metadata_artifact, f"{split} dataset")
        else:
            file = open(name_file_path, "rb")
            setattr(self, name_dataset, pickle.load(file))

        self.sampling_rate = 16000

        self.logger.info(f"Done prepare_data {split}")


    def process_dataset(self, split, processor, batch_size=512):
        '''
            Function to process data of a dataset (remove indesirable characters and process audio with processor)
        '''

        save_path = getattr(self, f"{split}_save_data_path")
        name_file = save_path.split('/')[-1] + "_process"
        name_file_path = osp.join(save_path, name_file)
        name_dataset = f"{split}_dataset"

        if not osp.exists(name_file_path) or self.config.create_dataset:
            # if not self.config.create_dataset:
            # # try
            #     path = f"asr-project/{self.config.wandb_project}/{name_file}:latest"
            #     self.logger.info(f"Try loading {path} in artifacts ...")

            #     file = ag_u.get_artifact(path, type="dataset")
                
            #     shutil.copy2(file, save_path)

            #     self.logger.info(f"Load {path} in artifacts OK")
                
            #     file = open(name_file_path, "rb")
            #     setattr(self, name_dataset, pickle.load(file))
            #     self.logger.info(
            #         f"Loaded processed {split} dataset : {name_file_path}")
            # else:
            # except
            self.logger.info(f'Processing {split} dataset ...')

            setattr(self, name_dataset, getattr(self, name_dataset).map(lambda x: {"sentence": re.sub(CHARS_TO_REMOVE_REGEX, '', x["sentence"]).lower()}, 
                                                    num_proc=self.config.num_proc, 
                                                    load_from_cache_file=False)
                                                    )
            setattr(self, name_dataset, getattr(self, name_dataset).map(lambda batch: {"audio": processor([ad["array"] for ad in batch["audio"]], sampling_rate=16000).input_values}, 
                                                    batched=True, 
                                                    batch_size=batch_size, 
                                                    num_proc=self.config.num_proc,
                                                    load_from_cache_file=False)
                                                    )

            self.logger.info(f"Saving {split} dataset ...")

            metadata_artifact = {
                                "dataset_name": self.config.dataset_name,
                                "subset": self.config.subset,
                                "split": split,
                                "sampling_rate": self.sampling_rate
                                }

            self._save_dataset(split, name_file_path, metadata_artifact, f"{split} dataset processed")
        else:
            self.logger.info(f"{split} dataset already exists no processing necessary ...")
            
            file = open(name_file_path, "rb")
            setattr(self, name_dataset, pickle.load(file))
            self.logger.info(
                f"Loaded processed {split} dataset : {name_file_path}")



    def filtered_data(self, split, top_db=15) -> None:
        '''
            Function to filter dataset (remove silence and remove long audio )
        '''

        self.logger.info(f"Filtering {split} dataset ...")

        save_path = getattr(self, f"{split}_save_data_path")
        name_file = f"{save_path.split('/')[-1]}_filter_{top_db}_{self.config.max_input_length_in_sec}"
        name_file_path = osp.join(save_path, name_file)
        name_dataset = f'{split}_dataset'
        
        if not osp.exists(name_file_path) or self.config.create_dataset:
            # if not self.config.create_dataset:
            # # try
            #     path = f"asr-project/{self.config.wandb_project}/{name_file}:latest"
            #     self.logger.info(f"Try loading {path} in artifacts ...")

            #     file = ag_u.get_artifact(path, type="dataset")

            #     shutil.copy2(file, getattr(self, f'{split}_save_data_path'))

            #     file = open(name_file_path, "rb")
            #     setattr(self, name_dataset, pickle.load(file))
            #     self.logger.info(
            #         f"Loaded filtered {split} dataset : {name_file_path}")
            # else:
            # # except
            self.logger.info(
                f"Length {split} dataset before filter {len(getattr(self, name_dataset))}")
            
            setattr(self, name_dataset, getattr(self, name_dataset).map(lambda x: {'audio': trim(np.array(x["audio"]), top_db=top_db)[0]}, 
                                                    num_proc=self.config.num_proc,
                                                    load_from_cache_file=False)
                                                    )
            setattr(self, name_dataset, getattr(self, name_dataset).filter(lambda x: len(x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, 
                                                    num_proc=self.config.num_proc,
                                                    load_from_cache_file=False)
                                                    )

            self.logger.info(
                f"Length {split} dataset after filter {len(getattr(self, name_dataset))}")

            metadata_artifact = {
                                "dataset_name": self.config.dataset_name,
                                "subset": self.config.subset,
                                "split": split,
                                "sampling_rate": self.sampling_rate,
                                "top_db": top_db,
                                "max_input_length_in_sec": self.config.max_input_length_in_sec
                                }

            self._save_dataset(split, name_file_path, metadata_artifact, f"{split} dataset processed and filtered")

        else:
            file = open(name_file_path, "rb")
            setattr(self, name_dataset, pickle.load(file))
            self.logger.info(
                f"Loaded filtered {split} dataset : {name_file_path}")

        self.logger.info(f"Length {split} dataset : {len(getattr(self, name_dataset))}")


    def create_phonemes(self, split) -> None:
        '''
            Function to phonemize all sentence of the dataset
        '''

        self.logger.info(f"Creating {split} phonemes ...")
        language = self.config.language[:2] if self.config.language[:2] != "zh" else "cmn"
        backend = EspeakBackend(language)
        separator = Separator(phone=" ", word="| ", syllable="")
        
        name_dataset = f"{split}_dataset"

        setattr(self, name_dataset, getattr(self, name_dataset).add_column('phonemes', backend.phonemize(
            getattr(self, name_dataset)['sentence'], njobs=self.config.num_proc, separator=separator)))


    def _save_dataset(self, split, name_file_path, metadata_artifact, description_artifact):

        file = open(name_file_path, "wb")
        pickle.dump(getattr(self, f"{split}_dataset"), file)

        self.logger.info(f"Saved to {name_file_path}")

        self.push_artefact(
            name_file_path, 
            metadata_artifact,
            description_artifact
            )


    def push_artefact(self, path_artifact, metadata, description):
        artifact = wandb.Artifact(
            name=osp.basename(path_artifact),
            type="dataset",
            metadata=metadata,
            description=description
        )
        artifact.add_file(path_artifact)
        wandb.log_artifact(artifact, aliases=["latest"])


    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            
            self.filtered_data("train")
            self.filtered_data("val")

            self.create_phonemes("train")
            self.create_phonemes("val")

        if stage == "test":
            self.create_phonemes("test")

        if stage == "predict":
            self.dataset = load_dataset(self.config.dataset_name,
                                        self.config.subset, split='other',
                                        use_auth_token=self.config.use_auth_token,
                                        download_mode=self.config.download_mode,
                                        cache_dir=self.config.cache_dir
                                        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn
        )
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=coll_fn
        )
        return val_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=coll_fn
        )
        return predict_loader
