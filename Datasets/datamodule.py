from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import json

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
# from utils.dataset_utils import coll_fn

class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()

        self.config = dataset_param

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            self.train_dataset = load_dataset(self.config.dataset_name, 
                                            self.config.subset, 
                                            split='train',
                                            use_auth_token=self.config.use_auth_token, 
                                            download_mode=self.config.download_mode, 
                                            cache_dir=self.config.cache_dir
                                            )

            self.val_dataset = load_dataset(self.config.dataset_name, 
                                            self.config.subset, 
                                            split='validation',
                                            use_auth_token=self.config.use_auth_token, 
                                            download_mode=self.config.download_mode, 
                                            cache_dir=self.config.cache_dir
                                            )

            phoneme_labels = json.load(open(self.config.phoneme_labels_file, 'r'))
            self.train_labels = phoneme_labels['train']
            self.val_labels = phoneme_labels['validation']


        if stage == "test":
            self.test_dataset = load_dataset(self.config.dataset_name, 
                                            self.config.subset, 
                                            split='test',
                                            use_auth_token=self.config.use_auth_token, 
                                            download_mode=self.config.download_mode, 
                                            cache_dir=self.config.cache_dir
                                            )

        if stage == "predict":
            self.dataset = load_dataset(self.config.dataset_name, 
                                        self.config.subset, split='other',
                                        use_auth_token=self.config.use_auth_token, 
                                        download_mode=self.config.download_mode, 
                                        cache_dir=self.config.cache_dir
                                        )

    def collate_fn(self, batch, nsplit):
        list_tensors = [torch.from_numpy(b['audio']['array']) for b in batch]
        list_labels = []
        batch_audio = pad_sequence(list_tensors, padding_value=-100, batch_first=True)
        #batch_audio = F.pad(([torch.from_numpy(b['audio']['array']) for b in batch]), pad=-100)
        # TODO padding for labels
        return batch_audio

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.coll_fn
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.coll_fn
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
