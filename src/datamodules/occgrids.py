import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NumpyDataset(Dataset): # inherits from torchvision VisionDataset class

    def __init__(self, np_data, transform=None):
        super().__init__()
        self.data = np_data
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.data[index]

        # convert to Img so PIL-style transforms work
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        out_transform = transforms.Compose([transforms.ToTensor()]) 
        img = out_transform(img).float() # always cast to float32 tensor at end

        return img
    
    def __len__(self):
        return len(self.data)

import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision import transforms

# adapted from PL authors:
class OccgridDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='./', transform=None, batch_size=32, num_workers=1,**kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size 
        self.num_workers = num_workers 

        # self.dims is returned when you call dm.size()
        self.dims = (1, 512, 512) # default, but might change

    def prepare_data(self):
	# sometimes, you download the dataset here if it doesn't exist on disk
        pass

    def setup(self, stage=None):
        data_filepath = os.path.join(self.data_dir, 'HallNav/occgrids.npy')
        data_filepath_test = os.path.join(self.data_dir, 'HallNav/occgrids_test.npy')
        occgrids = np.load(data_filepath) # deserialize the .npy file
        occgrids_test = np.load(data_filepath_test) 

        # assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            occs_full = NumpyDataset(occgrids, transform=self.transform)
            self.occs_train, self.occs_val = random_split(occs_full, [950, 50])

            # we want our dm.size() to be consistent with the train transforms
	    # (if any were applied) so update dims:
            self.dims = tuple(self.occs_train[0].shape)

        # assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.occs_test = NumpyDataset(occgrids_test, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.occs_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                         )

    def val_dataloader(self):
        return DataLoader(self.occs_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                         )

    def test_dataloader(self):
        return DataLoader(self.occs_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                         )
