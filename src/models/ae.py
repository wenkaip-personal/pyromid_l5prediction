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
from src.models.components.blocks import FCBlock 

class AE(pl.LightningModule):

    ''' An autoencoder with x -FC-> z -FC-> x_hat
        IN: 
        OUT:

    '''

    def __init__(self,                              # Necessary model params
                 input_dim,
                 hidden_layer_dim_list=[2,4,8],     # "
                 latent_dim=16,                     # "
                 **kwargs                           # note this is important
                 # For complete encapsulation, can include defaults for
                 # trainer, optimizers, dataloaders, etc. here as well.
                 ):
        super().__init__()
        self.save_hyperparameters() # takes dict of init args and saves as attribute
        #saves all hparams, if you want it leaner, just pass the important ones

        self._init_encoder(input_dim,
                           self.hparams.hidden_layer_dim_list,
                           self.hparams.latent_dim)

        self._init_decoder(input_dim,
                           list(reversed(self.hparams.hidden_layer_dim_list)),
                           self.hparams.latent_dim)

    #TODO: decide if the 'overriding case' actually holds. If not, remove these methods.
    def _init_encoder(self, input_dim, hidden_layer_dim_list, latent_dim):
        #TODO: ideally, this method could be overridden in 'run' and a different encoder could be used 
        #TODO: decide if we should actually have 'Encoder' abstraction or just put layers here
        self.encoder = FCBlock([input_dim]+hidden_layer_dim_list+[latent_dim])

    def _init_decoder(self, input_dim, hidden_layer_dim_list, latent_dim):
        self.decoder = FCBlock([latent_dim]+hidden_layer_dim_list+[input_dim])

    #NOTE: This generic 'loss' method can just be overridden at higher level
    def loss(self, x_hat, x):
        return F.mse_loss(x_hat, x)

    def forward(self, x):
        return self.encoder(x)

    def _step(self, batch):
        x, _ = batch
        x = x.view(x.shape[0],-1) # flatten image to raster scan
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
        result = pl.EvalResult(checkpoint_on=step_dict['loss']
                              )
        
        # logging
        result.log('avg_val_loss', step_dict['loss'],
                   on_epoch=True, reduce_fx=torch.mean)
        
        return result

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        result = pl.EvalResult(
                              )
        
        # logging
        result.log('avg_test_loss', step_dict['loss'],
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
    import numpy as np

    import pytorch_lightning as pl
    #from pl_bolts.datamodules import MNISTDataModule # get from external repo
    from src.datamodules.mnist import MNISTDataModule
    from src.callbacks.viz import VizReconstructions
    from src.utils import get_project_root

    # Ground the runtime to the directory above 'src/models'
    # This is where outputs should go (logs, etc.)
    ROOT_DIR = get_project_root()

    # Setup all arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AE.add_model_specific_args(parser)
    args = parser.parse_args()
    args = AE.args_strings_to_lists(args)
    args.data_dir = ROOT_DIR.joinpath('data/')
    args.weights_summary = 'full'

    # Initialize DataModule (contains pytorch dataloaders)
    dm = MNISTDataModule(data_dir=args.data_dir,
                         num_workers=args.num_workers,
                        )
    input_dim = np.prod(dm.size()) # serialized to one dimension

    #old way with dataloaders:
    #input_dim = int(*train_loader.dataset.__getitem__(0)[0].view(-1).shape)

    # Initialize Model
    model = AE(input_dim, **vars(args))

    # Initlialize Trainer and Fit
    callbacks = [VizReconstructions()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, default_root_dir=ROOT_DIR)
    trainer.fit(model, datamodule=dm)
    #trainer.fit(model,train_dataloader=train_loader, val_dataloaders=val_loader) # for plain pytorch dataloders
