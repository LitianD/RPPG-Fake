from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pickle as pkl
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
import os
import re
import fnmatch

class PolitifactDataset(Dataset):
    def __init__(self, data_path,num_classes,image_path,propagation_path):
        self.data_path = data_path
        self.num_classes = num_classes
        self.image_path = image_path
        self.propagation_path = propagation_path
        
        self.samples = self.load_samples()


    def load_samples(self):
        # load_samples here, this is a function to read data
        # with open(self.data_path,"rb") as f:
        #     data_df = pkl.load(f)
        #print(data_df.columns)
        data_df = pd.read_csv(self.data_path)
        data_df = data_df.sample(frac=1, random_state=666).reset_index(drop=True)[:4175]
        image_path_list = []
        users_path_list = []
        text_list = list(data_df["text"])
        id_list = list(data_df["id"])
        label_list = list(data_df["label"])
        # for i,text in enumerate(text_list):
        #     if len(text) > 500:
        #         text_list[i] = text[:500]
        for id in id_list:
            image_path = self.find_image_path(id,origin_path=self.image_path)
            image_path_list.append(image_path)
        for id in id_list:
            user_path = self.find_image_path(id,origin_path=self.propagation_path)
            users_path_list.append(user_path)
        combined_list = list(zip(text_list,image_path_list,label_list,users_path_list))
        # combined_list:[("text",0),("text1",1)]
        return combined_list
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, id, label,user = self.samples[idx]
        # data preprocessing
        target = torch.tensor(int(label))
        final_target = F.one_hot(target,num_classes=self.num_classes).float()
        # if self.tokenizer is not None:
        #     text = self.tokenizer(text,return_tensors='pt',truncation = True,padding='max_length',max_length=512)
        #print(text.shape)
        return text,id,final_target,user
    
    def find_image_path(self,id,origin_path):
        """
            find image path
        """
        files_list = os.listdir(origin_path)
        pattern = f"{id}.*"
        matching_files = fnmatch.filter(files_list,pattern)
        
        if matching_files:
            image_path = os.path.join(origin_path,matching_files[0])
            return image_path
        else:
            return '-1'
        
class PolitifactDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir : str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes : int = 2,
        image_path='',
        propagation_path='',
    ):
        super().__init__()
        #self.data_dir = data_dir
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        # self.image_path = image_path
        # self.propagation_path = propagation_path
    
    @property
    def num_classes(self):
        return 2
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        dataset = PolitifactDataset(data_path = self.hparams.data_dir,num_classes=self.hparams.num_classes, image_path=  self.hparams.image_path, propagation_path= self.hparams.propagation_path)
        if stage=="fit" or stage is None:
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            indices = self.hparams.train_val_test_split
            self.data_train = Subset(dataset, range(indices[0]))
            # self.data_val = Subset(dataset, range(indices[0],indices[0]+indices[1]))
            # self.data_test = Subset(dataset, range(indices[0]+indices[1],indices[0]+indices[1]+indices[2]))
            self.data_test = Subset(dataset, range(indices[0],indices[0]+indices[1]))
            self.data_val = Subset(dataset, range(indices[0]+indices[1],indices[0]+indices[1]+indices[2]))
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
if __name__ == "__main__":
    test = SnopesDataModule()