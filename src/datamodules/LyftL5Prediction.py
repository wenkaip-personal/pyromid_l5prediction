import sys
sys.path.insert(0, "/home/herb/WRK/ken/l5kit_repo/l5kit")

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.dataset.dataloader_builder import build_dataloader
from l5kit.rasterization import build_rasterizer

import pytorch_lightning as pl
import matplotlib.pyplot as plt

class LyftL5PredictionDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.cfg = load_config_data("/home/herb/WRK/ken/l5kit_repo/agent_motion_config.yaml")
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