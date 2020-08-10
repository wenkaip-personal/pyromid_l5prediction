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

class VAE(pl.LightningModule):

    ''' A variational autoencoder with x-FC->z-FC->x_hat
        IN: 
        OUT:

    '''
    #TODO: clearer var names
    # x, data, traj (or img or occ)
    # c, cond, img, occ 
    # z, latent, bottleneck, embedding, representation 

    def __init__(self,                              
                 data_dim,
                 hidden_layer_dim_list=[64,64],
                 latent_dim=1,                     
                 **kwargs                           
                 ):
        super().__init__()
        self.save_hyperparameters() 

        self._init_encoder(data_dim,
                           self.hparams.hidden_layer_dim_list,
                           self.hparams.latent_dim)

        self._init_decoder(data_dim,
                           list(reversed(self.hparams.hidden_layer_dim_list)),
                           self.hparams.latent_dim)

    def _init_encoder(self, data_dim, hidden_layer_dim_list, latent_dim):
        layer_dim_list = [data_dim] \
                       + hidden_layer_dim_list \
                       + [latent_dim]

        self.encoder_mu = FCBlock(layer_dim_list)
        self.encoder_sigma = FCBlock(layer_dim_list)

    def _init_decoder(self, data_dim, hidden_layer_dim_list, latent_dim):
        layer_dim_list = [latent_dim] \
                       + hidden_layer_dim_list \
                       + [data_dim]

        self.decoder = FCBlock(layer_dim_list)

    def get_prior(self, z_mu, z_std):
        # Prior ~ Normal(0,1)
        P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
        return P

    def get_approx_posterior(self, z_mu, z_std):
        # Approx Posterior ~ Normal(mu, sigma)
        Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
        return Q

    def loss(self, x, P, Q):
        #ELBO loss = Reconstruction loss + KL loss

        # Reconstruction loss
        z = Q.rsample()
        pxz = self(z)
        recon_loss = F.mse_loss(pxz, x, reduction='none')

        # sum across dimensions because sum of log probabilities of iid univariate gaussians is the same as
        # multivariate gaussian
        recon_loss = recon_loss.sum(dim=-1)

        # KL divergence loss
        log_qz = Q.log_prob(z)
        log_pz = P.log_prob(z)
        kl_div = (log_qz - log_pz).sum(dim=1)

        # ELBO = reconstruction + KL
        loss = recon_loss + kl_div

        # average over batch
        loss = loss.mean()
        recon_loss = recon_loss.mean()
        kl_div = kl_div.mean()

        return loss, recon_loss, kl_div, pxz

    def forward(self, z):
        return self.decoder(z)

    def _step(self, batch):
        x, _ = batch
        x = x.view(x.shape[0], -1) # flatten image to raster scan

        z_mu = self.encoder_mu(x) 
        z_log_var = self.encoder_sigma(x)
        z_std = torch.exp(z_log_var / 2) #TODO: necessary?

        P = self.get_prior(z_mu, z_std)
        Q = self.get_approx_posterior(z_mu, z_std)

        loss, recon_loss, kl_div, pxz = self.loss(x, P, Q) #TODO: refactor loss

        # Needed for viz callbacks
        self.img_shape = batch[0].shape[1:]
        self.last_batch = batch
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_div': kl_div,
            'pxz': pxz,
            'x': x,
            'viz': {            # contents of dict will be img to autoviz
                    'x': x,
                    'pxz': pxz,
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
        result = pl.EvalResult(early_stop_on=step_dict['loss'],
                               checkpoint_on=step_dict['loss'],
                              )
        
        # logging
        result.log('avg_val_loss', step_dict['loss'],
                   on_epoch=True, reduce_fx=torch.mean)
        
        return result

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        result = pl.EvalResult(early_stop_on=step_dict['loss'],
                               checkpoint_on=step_dict['loss'],
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
        parser.add_argument('--hidden_layer_dim_list', type=str, default='2,4,8,16',
                            help='intermediate layers dimensions for enc/decoder')
        parser.add_argument('--latent_dim', type=int, default=32,
                            help='dimension of embedding z')

        # Training hyperparameters for this specific model:
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=0.001)

        # Dataloaders args 
        parser.add_argument('--num_workers', type=int, default=16) 
        parser.add_argument('--data_path', type=str, default='data/') 
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser
    from torchvision import datasets, transforms
    import numpy as np

    import pytorch_lightning as pl
    from src.datamodules.mnist import MNISTDataModule
    from src.callbacks.viz import VizStepDict
    from src.utils import get_project_root

    # Ground the runtime to the directory above 'src/models'
    # This is where outputs should go (logs, etc.)
    ROOT_DIR = get_project_root()

    # Setup all arguments
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()
    args = VAE.args_strings_to_lists(args)
    args.data_dir = ROOT_DIR.joinpath('data/')
    args.weights_summary = 'full'

    # Initialize DataModule (contains pytorch dataloaders)
    dm = MNISTDataModule(data_dir=args.data_dir,
                         num_workers=args.num_workers,
                        )
    img_shape = dm.size()
    #img_shape = [1,32,32] 
    data_dim = np.prod(img_shape)

    # Initialize Model
    model = VAE(data_dim, **vars(args))

    # Initlialize Trainer and Fit
    callbacks = [VizStepDict()]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, default_root_dir=ROOT_DIR)
    trainer.fit(model, datamodule=dm)
