from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.dataset_utils import coll_fn
from utils.logger import init_logger
from datasets import Audio
from librosa.effects import trim
import pickle
import os
import numpy as np
import wandb
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()

        self.config = dataset_param
        self.logger = init_logger("BaseDataModule", "INFO")
        self.logger.info(
            f"Loading Dataset : {self.config.dataset_name}, language : {self.config.subset}")

    def prepare_data(self) -> None:
        self.logger.info(
            "Preparing the dataset in prepare_data")

        self.train_save_data_path = f"assets/datasets/train_{self.config.dataset_name}-{self.config.subset}"
        if os.path.exists(self.train_save_data_path):
            file = open(self.train_save_data_path, "rb")
            self.train_dataset = pickle.load(file)
        else:
            self.train_dataset = load_dataset(self.config.dataset_name,
                                              self.config.subset,
                                              split='train',
                                              use_auth_token=self.config.use_auth_token,
                                              download_mode=self.config.download_mode,
                                              cache_dir=self.config.cache_dir
                                              )

            self.train_dataset = self.train_dataset.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            self.train_dataset = self.train_dataset.cast_column(
                "audio", Audio(sampling_rate=16_000))

        # self.train_dataset.features['audio'].sampling_rate FIXME
        self.sampling_rate = 16000

        self.val_save_data_path = f"assets/datasets/val_{self.config.dataset_name}-{self.config.subset}"
        if os.path.exists(self.val_save_data_path):
            file = open(self.val_save_data_path, "rb")
            self.val_dataset = pickle.load(file)
        else:
            self.val_dataset = load_dataset(self.config.dataset_name,
                                            self.config.subset,
                                            split='validation',
                                            use_auth_token=self.config.use_auth_token,
                                            download_mode=self.config.download_mode,
                                            cache_dir=self.config.cache_dir
                                            )
            self.val_dataset = self.val_dataset.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            self.val_dataset = self.val_dataset.cast_column(
                "audio", Audio(sampling_rate=16_000))

        self.logger.info("Done prepare_data")

        return super().prepare_data()

    def prepare_data_test(self) -> None:
        self.logger.info(
            "Preparing the dataset in prepare_data")

        self.test_save_data_path = f"assets/datasets/test_{self.config.dataset_name}-{self.config.subset}"
        if os.path.exists(self.test_save_data_path):
            file = open(self.test_save_data_path, "rb")
            self.test_dataset = pickle.load(file)
        else:
            self.test_dataset = load_dataset(self.config.dataset_name,
                                              self.config.subset,
                                              split='test',
                                              use_auth_token=self.config.use_auth_token,
                                              download_mode=self.config.download_mode,
                                              cache_dir=self.config.cache_dir
                                              )

            self.test_dataset = self.test_dataset.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            self.test_dataset = self.test_dataset.cast_column(
                "audio", Audio(sampling_rate=16_000))

        # self.train_dataset.features['audio'].sampling_rate FIXME
        self.sampling_rate = 16000

        self.logger.info("Done prepare_data")

        return super().prepare_data()


    def _save_dataset(self, split):

        if split == "train":
            if not os.path.exists(self.train_save_data_path):
                file = open(self.train_save_data_path, "wb")
                pickle.dump(self.train_dataset, file)

                self.logger.info(f"Saved to {self.train_save_data_path}")

                self.push_artefact(self.train_save_data_path, {
                                   "dataset_name": self.config.dataset_name, 
                                   "subset": self.config.subset, 
                                   "split": "train", 
                                   "sampling_rate": self.sampling_rate}, 
                                   "train dataset processed")

        elif "val" in split:
            if not os.path.exists(self.val_save_data_path):
                file = open(self.val_save_data_path, "wb")
                pickle.dump(self.val_dataset, file)

                self.logger.info(f"Saved to {self.val_save_data_path}")
                
                self.push_artefact(self.val_save_data_path, {
                                   "dataset_name": self.config.dataset_name, 
                                   "subset": self.config.subset, 
                                   "split": "validation", 
                                   "sampling_rate": self.sampling_rate}, 
                                   "validation dataset processed")

        else:
            if not os.path.exists(self.test_save_data_path):
                file = open(self.test_save_data_path, "wb")
                pickle.dump(self.test_dataset, file)

                self.logger.info(f"Saved to {self.test_save_data_path}")
                
                self.push_artefact(self.test_save_data_path, {
                                   "dataset_name": self.config.dataset_name, 
                                   "subset": self.config.subset, 
                                   "split": "test", 
                                   "sampling_rate": self.sampling_rate}, 
                                   "test dataset processed")

    def push_artefact(self, path_artifact, metadata, description):
        artifact = wandb.Artifact(
            name=os.path.basename(path_artifact),
            type="dataset",
            metadata=metadata,
            description=description
        )
        artifact.add_file(path_artifact)
        wandb.log_artifact(artifact, aliases=["latest"])


    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):

            top_db = 15
            name_train_filter_path = f"{self.train_save_data_path}_filter_{top_db}_{self.config.max_input_length_in_sec}"
            name_val_filter_path = f"{self.val_save_data_path}_filter_{top_db}_{self.config.max_input_length_in_sec}"

            if not os.path.exists(name_train_filter_path):
                self.logger.info(
                    f"Length train dataset before filter {len(self.train_dataset)}")
                self.train_dataset = self.train_dataset.map(
                    lambda x: {'audio': trim(np.array(x["audio"]), top_db=top_db)[0]}, num_proc=self.config.num_proc)
                self.train_dataset = self.train_dataset.filter(lambda x: len(
                    x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, num_proc=self.config.num_proc)
                self.logger.info(
                    f"Length train dataset after filter {len(self.train_dataset)}")

                file = open(name_train_filter_path, "wb")
                pickle.dump(self.train_dataset, file)

                self.logger.info(f"Saved to {name_train_filter_path}")
                self.push_artefact(name_train_filter_path, {
                                   "dataset_name": self.config.dataset_name, 
                                   "subset": self.config.subset, 
                                   "split": "train", 
                                   "sampling_rate": self.sampling_rate,
                                   "top_db": top_db,
                                   "max_input_length_in_sec": self.config.max_input_length_in_sec}, 
                                   "train dataset processed and filtered")
            else:
                file = open(name_train_filter_path, "rb")
                self.train_dataset = pickle.load(file)
                self.logger.info(f"Loaded filtered train dataset : {name_train_filter_path}")

            if not os.path.exists(name_val_filter_path):
                self.logger.info(
                    f"Length val dataset before filter {len(self.val_dataset)}")
                self.val_dataset = self.val_dataset.map(
                    lambda x: {'audio': trim(np.array(x["audio"]), top_db=top_db)[0]}, num_proc=self.config.num_proc)
                self.val_dataset = self.val_dataset.filter(lambda x: len(
                    x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, num_proc=self.config.num_proc)
                self.logger.info(
                    f"Length val dataset after filter {len(self.val_dataset)}")

                file = open(name_val_filter_path, "wb")
                pickle.dump(self.val_dataset, file)

                self.logger.info(f"Saved to {name_val_filter_path}")
                self.push_artefact(name_val_filter_path, {
                                   "dataset_name": self.config.dataset_name, 
                                   "subset": self.config.subset, 
                                   "split": "validation", 
                                   "sampling_rate": self.sampling_rate,
                                   "top_db": top_db,
                                   "max_input_length_in_sec": self.config.max_input_length_in_sec}, 
                                   "validation dataset processed and filtered")
            else:
                file = open(name_val_filter_path, "rb")
                self.val_dataset = pickle.load(file)
                self.logger.info(f"Loaded filtered val dataset : {name_val_filter_path}")

            self.logger.info(f"Creating phonemes")
            backend = EspeakBackend(self.config.language)
            separator = Separator(phone=" ", word="| ", syllable="")

            self.train_dataset = self.train_dataset.add_column('phonemes', backend.phonemize(self.train_dataset['sentence'], njobs=self.config.num_proc, separator=separator))
            self.val_dataset = self.val_dataset.add_column('phonemes', backend.phonemize(self.val_dataset['sentence'], njobs=self.config.num_proc, separator=separator))

        if stage == "test":
            self.logger.info(f"Creating phonemes")
            backend = EspeakBackend(self.config.language)
            separator = Separator(phone=" ", word="| ", syllable="")
            self.test_dataset = self.test_dataset.add_column('phonemes', backend.phonemize(self.test_dataset['sentence'], njobs=self.config.num_proc, separator=separator))
            
            # self.test_dataset = load_dataset(self.config.dataset_name,
            #                                  self.config.subset,
            #                                  split='test',
            #                                  use_auth_token=self.config.use_auth_token,
            #                                  download_mode=self.config.download_mode,
            #                                  cache_dir=self.config.cache_dir
            #                                  )

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
