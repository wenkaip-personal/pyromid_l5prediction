import sys
sys.path.insert(0, "/home/herb/WRK/ken/l5kit_repo/l5kit")
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from typing import Dict
from tempfile import gettempdir
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from l5kit.evaluation import write_coords_as_csv, compute_mse_error_csv
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from prettytable import PrettyTable
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from src.models.ae import AE
from torchvision.models.resnet import resnet50
from src.datamodules.mnist import FashionMNISTDataModule
from src.datamodules.LyftL5Prediction import LyftL5PredictionDataModule
from src.utils import get_project_root

class AE_exp(AE):

    def __init__(self,
                 input_dim,
                 hidden_layer_dim_list=[2,4,8],
                 latent_dim=16,
                 **kwargs
                 ):
        self.save_hyperparameters()
        super().__init__(**self.hparams)
    
    def _step(self, batch):
        x = batch['image']
        x = x.view(x.shape[0],-1) # flatten image to raster scan
        z = self(x)
        x_hat = self.decoder(z)         
        loss = self.loss(x_hat, x) 

        # Needed for viz callbacks
        #self.img_shape = batch['image'].shape[0]
        #self.last_batch = batch
        return {
            'loss': loss,
            'x': x,
            'z': z,
            'x_hat': x_hat,
            'viz': {            # contents of dict will be img to autoviz
                    'x': x,
                    'x_hat': x_hat,
                   },
        }

def build_model_resnet50(cfg: Dict) -> torch.nn.Module:
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

def forward_resnet50(data, model, device, criterion):
    inputs = data["image"].to(device)
    targets = data["target_positions"].to(device).reshape(len(data["target_positions"]), -1)
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss.mean()
    return loss, outputs

if __name__ == '__main__':
    ROOT_DIR = get_project_root()

    # create an argument parser and a namespace from all parsed arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AE.add_model_specific_args(parser)
    args = parser.parse_args()
    args = AE.args_strings_to_lists(args)
    args.data_dir = ROOT_DIR.joinpath('data/')
    args.weights_summary = 'full'
    # args.gpus = 0                   # set args from main for debugging purposes
    # args.fast_dev_run = True

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = "/home/herb/WRK/ken/data"
    cfg = load_config_data("./agent_motion_config.yaml")
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)
    # ==== INIT DATAMODULE
    dm = LyftL5PredictionDataModule()
    # dm = FashionMNISTDataModule(data_dir=args.data_dir,
    #                         num_workers=args.num_workers,
    #                         )
    dm.prepare_data()
    dm.setup(stage = "fit")

    # print out a single sample from data module
    train_batch_1 = next(iter(dm.train_dataloader()))
    print(train_batch_1)
    print(train_batch_1['image'])
    print(train_batch_1['image'].shape)
    print(train_batch_1.keys())

    train_agent_dataset = dm.train_dataloader().dataset.datasets[0].dataset
    example_image = train_agent_dataset[0]['image']


    input_dim = np.prod(example_image.shape) # serialized to single dimension
    # input_dim is technically not a hyperparam here, so it is excluded from args

    # Initialize Model
    model = AE_exp(input_dim, **vars(args))
    # ==== INIT MODEL
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = build_model_resnet50(dm.cfg).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss(reduction="none")

    # Initlialize Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
