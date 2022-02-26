from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets import load_dataset


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()

        self.config = dataset_param

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            self.train_dataset = load_dataset(self.config.dataset_name, self.config.subset, split='train')
            self.val_dataset = load_dataset(self.config.dataset_name, self.config.subset, split='validation')
        
        if stage == "test":
            self.test_dataset = load_dataset(self.config.dataset_name, self.config.subset, split='test')
        
        if stage == "predict":
            self.dataset = load_dataset(self.config.dataset_name, self.config.subset, split='other')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        return val_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,

        )
        return predict_loader
