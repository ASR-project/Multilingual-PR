from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.dataset_utils import coll_fn
from utils.logger import init_logger
from datasets import Audio
from librosa.effects import trim
import pickle
import os
import os.path as osp
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
        return super().prepare_data()


    def load_data(self, split) -> None:
        self.logger.info(f"Preparing the dataset in prepare_data: {split}")

        setattr(self, f"{split}_save_data_path", f"assets/datasets/{split}_{self.config.dataset_name}-{self.config.subset}")

        save_path = getattr(self, f"{split}_save_data_path")
        name_dataset = f"{split}_dataset"

        if osp.exists(save_path) and not self.config.recreate_dataset:
            file = open(save_path, "rb")
            setattr(self, name_dataset, pickle.load(file))
        else:
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
                "audio", Audio(sampling_rate=16_000)))

        self.sampling_rate = 16000

        self.logger.info(f"Done prepare_data {split}")


    def _save_dataset(self, split):
        save_path = getattr(self, f"{split}_save_data_path")

        file = open(save_path, "wb")
        pickle.dump(getattr(self, f"{split}_dataset"), file)

        self.logger.info(f"Saved to {save_path}")

        self.push_artefact(save_path, {
            "dataset_name": self.config.dataset_name,
            "subset": self.config.subset,
            "split": split,
            "sampling_rate": self.sampling_rate},
            f"{split} dataset processed")


    def push_artefact(self, path_artifact, metadata, description):
        artifact = wandb.Artifact(
            name=osp.basename(path_artifact),
            type="dataset",
            metadata=metadata,
            description=description
        )
        artifact.add_file(path_artifact)
        wandb.log_artifact(artifact, aliases=["latest"])

    
    def filtered_data(self, split, top_db=15) -> None:

        self.logger.info(f"Filtering {split} dataset ...")

        name_filter_path = f"{getattr(self, f'{split}_save_data_path')}_filter_{top_db}_{self.config.max_input_length_in_sec}"

        name_dataset = f'{split}_dataset'
        dataset = getattr(self, name_dataset)
        
        if not osp.exists(name_filter_path) or self.config.recreate_dataset:
            self.logger.info(
                f"Length {split} dataset before filter {len(dataset)}")
            
            setattr(self, name_dataset, dataset.map(
                lambda x: {'audio': trim(np.array(x["audio"]), top_db=top_db)[0]}, num_proc=self.config.num_proc))
            setattr(self, name_dataset, dataset.filter(lambda x: len(
                x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, num_proc=self.config.num_proc))

            self.logger.info(
                f"Length {split} dataset after filter {len(dataset)}")

            file = open(name_filter_path, "wb")
            pickle.dump(dataset, file)

            self.logger.info(f"Saved to {name_filter_path}")
            
            self.push_artefact(name_filter_path, {
                                "dataset_name": self.config.dataset_name,
                                "subset": self.config.subset,
                                "split": split,
                                "sampling_rate": self.sampling_rate,
                                "top_db": top_db,
                                "max_input_length_in_sec": self.config.max_input_length_in_sec},
                                "{split} dataset processed and filtered")
        else:
            file = open(name_filter_path, "rb")
            dataset = pickle.load(file)
            self.logger.info(
                f"Loaded filtered {split} dataset : {name_filter_path}")

        self.logger.info(f"Length {split} dataset : {len(dataset)}")

    def create_phonemes(self, split) -> None:
        self.logger.info(f"Creating {split} phonemes ...")
        backend = EspeakBackend(self.config.language)
        separator = Separator(phone=" ", word="| ", syllable="")
        
        name_dataset = f"{split}_dataset"

        setattr(self, name_dataset, getattr(self, name_dataset).add_column('phonemes', backend.phonemize(
            getattr(self, name_dataset)['sentence'], njobs=self.config.num_proc, separator=separator)))


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
