from argparse import ArgumentParser 
import torch
from torch import nn
from torch import distributions
from torch.nn import functional as F

#==Only necessary for running modules in place (python ae.py  @ ~/../src/models)
# TODO: replace this with more elegant solution
import sys
from pathlib import Path
sys.path.insert(0,str(Path().resolve().parents[1]))
#====================================================

import pytorch_lightning as pl
from src.models.components.blocks import ConvBlock, UpConvBlock

class ConvAE(pl.LightningModule):

    ''' An autoencoder with x -DownConv-> z -UpConv-> x_hat
        IN: 
        OUT:

    '''

    def __init__(self,                              
                 img_shape,
                 hidden_layer_dim_list=[2,4,8],     
                 latent_dim=16,                     
                 **kwargs                           
                 ):
        super().__init__()
        self.save_hyperparameters()

        self._init_encoder(img_shape,
                           self.hparams.hidden_layer_dim_list,
                           self.hparams.latent_dim)

        self._init_decoder(img_shape,
                           list(reversed(self.hparams.hidden_layer_dim_list)),
                           self.hparams.latent_dim)

    def _init_encoder(self, img_shape, hidden_layer_dim_list, latent_dim):
        self.encoder = ConvBlock(img_shape,hidden_layer_dim_list+[latent_dim])

    # requires img_shape to be power of 2 and square
    def _init_decoder(self, img_shape, hidden_layer_dim_list, latent_dim):
        self.decoder = UpConvBlock(img_shape,[latent_dim]+hidden_layer_dim_list)

    def loss(self, x_hat, x):
        return F.mse_loss(x_hat, x)

    def forward(self, x):
        return self.encoder(x)

    def _step(self, batch):
        x, _ = batch
        z = self(x)
        x_hat = self.decoder(z)         
        loss = self.loss(x_hat, x) 

        # Needed for viz callbacks
        self.img_shape = batch[0].shape[1:]
        self.last_batch = batch
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

    def training_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        result = pl.TrainResult(minimize=step_dict['loss'],
                                checkpoint_on=step_dict['loss'],
                               )
        
        # logging
        result.log('train_loss', step_dict['loss'], prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        result = pl.EvalResult(checkpoint_on=step_dict['loss'],
                               early_stop_on=step_dict['loss'],
                              )
        
        # logging
        result.log('avg_val_loss', step_dict['loss'],
                   on_epoch=True, reduce_fx=torch.mean)
        
        return result

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        result = pl.EvalResult(checkpoint_on=step_dict['loss'],
                              )
        
        # logging
        result.log('test_loss', step_dict['loss'],
                   on_epoch=True, reduce_fx=torch.mean)
        
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def args_strings_to_lists(args):
        if isinstance(args.hidden_layer_dim_list, str):
            args.hidden_layer_dim_list = \
            [int(i) for i in args.hidden_layer_dim_list.split(',')]
        return args

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model hyperparameters: 
        parser.add_argument('--hidden_layer_dim_list', type=str, default='2,4,8',
                            help='intermediate layers dimensions for enc/decoder')
        parser.add_argument('--latent_dim', type=int, default=16,
                            help='dimension of embedding z')

        # Training hyperparameters for this specific model:
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=0.001)

        # Dataloaders args 
        parser.add_argument('--num_workers', type=int, default=16) 
        parser.add_argument('--data_dir', type=str, default='data/') 
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    from torchvision import datasets, transforms

    import pytorch_lightning as pl
    from src.datamodules.mnist import MNISTDataModule
    from src.callbacks.viz import VizReconstructions
    from src.utils import get_project_root

    # Ground the runtime to the directory above 'src/models'
    # This is where outputs should go (logs, etc.)
    ROOT_DIR = get_project_root()

    # Setup all arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ConvAE.add_model_specific_args(parser)
    args = parser.parse_args()
    args = ConvAE.args_strings_to_lists(args)
    args.data_dir = ROOT_DIR.joinpath('data/')
    args.weights_summary = 'full'

    # Pad datapoints to achieve 32 x 32 (square, base 2)
    transforms = transforms.Compose([
                                    transforms.Pad(2),
                                    transforms.ToTensor(),
                                    ])
    # Initialize DataModule (contains pytorch dataloaders)
    dm = MNISTDataModule(data_dir=args.data_dir,
                         num_workers=args.num_workers,
                         train_transforms=transforms,
                         val_transforms=transforms,
                         test_transforms=transforms,
                        )
    img_shape = dm.size()
    #img_shape = [1,32,32]

    # Initialize Model
    model = ConvAE(img_shape, **vars(args))

    # Initlialize Trainer and Fit
    callbacks = [VizReconstructions(serialize_input=False)]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, default_root_dir=ROOT_DIR)
    trainer.fit(model, datamodule=dm)
