import sys
sys.path.insert(0, "/home/herb/WRK/ken/l5kit_repo/l5kit")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset.dataloader_builder import build_dataloader
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_coords_as_csv, compute_mse_error_csv
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
import pytorch_lightning as pl
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import os

class LyftL5PredictionDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.cfg = load_config_data("./agent_motion_config.yaml")
        self.dm = LocalDataManager(None)
        self.rasterizer = build_rasterizer(self.cfg, self.dm)
    
    def train_dataloader(self):
        return build_dataloader(self.cfg, "train", self.dm, EgoDataset, self.rasterizer)

    def val_dataloader(self):
        return build_dataloader(self.cfg, "val", self.dm, EgoDataset, self.rasterizer)

    def test_dataloader(self):
        pass
    
    def visualize_image(self, image):
        image_rgb = self.rasterizer.to_rgb(image.transpose(1, 2, 0))
        plt.imshow(image_rgb[::-1])

# model: a simple resnet50 pretrained on imagenet    
def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True)

    # change input size
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size
    # X, Y  * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model

def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    targets = data["target_positions"].to(device).reshape(len(data["target_positions"]), -1)
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss.mean()
    return loss, outputs

if __name__ == '__main__':
    # create an argument parser and a namespace from all parsed arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.gpus = 0                   # set args from main for debugging purposes
    args.fast_dev_run = True

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/home/herb/WRK/ken/data"

    # ==== INIT DATAMODULE
    import ipdb; ipdb.set_trace()
    dm = LyftL5PredictionDataModule()
    '''
    dm = LyftL5PredictionDataModule(data_dir=args.data_path, 
                                    num_workers=args.num_workers)
    '''
    dm.prepare_data()
    dm.setup(stage = "fit")

    # print out a single sample from data module
    train_batch_1 = next(iter(dm.train_dataloader()))
    print(train_batch_1)
    # import ipdb; ipdb.set_trace()
    print(train_batch_1['image'])
    print(train_batch_1['image'].shape)
    print(train_batch_1.keys())

    # ==== INIT MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(dm.cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    # Initlialize Trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # trainer.fit(model, datamodule=dm)

    # trainer.test(model, datamodule=dm)