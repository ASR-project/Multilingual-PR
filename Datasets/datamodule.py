from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.dataset_utils import coll_fn
from utils.logger import init_logger
from datasets import Audio
from librosa.effects import trim 
import pickle
import os
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
        
        self.train_save_data_path = f"assets/datasets/{self.config.dataset_name}"
        if os.path.exists(self.train_save_data_path) : 
            file = open(self.train_save_data_path,"rb")
            self.train_dataset = pickle.load(file)
        else:
            self.train_dataset = load_dataset(self.config.dataset_name,
                                                self.config.subset,
                                                split='train',
                                                use_auth_token=self.config.use_auth_token,
                                                download_mode=self.config.download_mode,
                                                cache_dir=self.config.cache_dir
                                                )

            self.train_dataset = self.train_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            self.train_dataset = self.train_dataset.cast_column("audio",Audio(sampling_rate=16_000))
            
            
        self.sampling_rate = 16000 #self.train_dataset.features['audio'].sampling_rate
        
        
        self.val_save_data_path = f"assets/datasets/{self.config.dataset_name}"
        if os.path.exists(self.val_save_data_path) : 
            file = open(self.val_save_data_path,"rb")
            self.val_dataset = pickle.load(file)
        else:
            self.val_dataset = load_dataset(self.config.dataset_name,
                                                self.config.subset,
                                                split='validation',
                                                use_auth_token=self.config.use_auth_token,
                                                download_mode=self.config.download_mode,
                                                cache_dir=self.config.cache_dir
                                                )
            self.val_dataset = self.val_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
            self.val_dataset = self.val_dataset.cast_column("audio",Audio(sampling_rate=16_000))
            
            
            
        return super().prepare_data()

    def setup(self, stage=None):
        # Build dataset
        if stage in (None, "fit"):
            # self.train_dataset = self.train_dataset.map(lambda x: {'audio':trim(x["audio"], top_db = 15)[0]})
            if not os.path.exists(self.train_save_data_path) : 
                self.logger.info(f"Length train dataset before filter {len(self.train_dataset)}")
                self.train_dataset = self.train_dataset.filter(lambda x: len(x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, num_proc=4)
                self.logger.info(f"Length train dataset after filter {len(self.train_dataset)}")


                file = open(self.train_save_data_path,"wb")
                pickle.dump(self.train_dataset,file)
            
            # self.val_dataset = self.val_dataset.map(lambda x: {'audio': trim(x["audio"], top_db = 15)[0]})
            if not os.path.exists(self.val_save_data_path) : 
                self.logger.info(f"Length val dataset before filter {len(self.val_dataset)}")
                self.val_dataset = self.val_dataset.filter(lambda x: len(x["audio"]) < self.config.max_input_length_in_sec * self.sampling_rate, num_proc=4)
                self.logger.info(f"Length val dataset after filter {len(self.val_dataset)}")
                file = open(self.val_save_data_path,"wb")
                pickle.dump(self.val_dataset,file)

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
