import math
import torch
from torch import nn
from torch.nn import functional as F

# Usage Note: If you don't want to use this block format, you can also
# manually add a sequence of nn.Linear or nn.Conv2D to your top-level module.
# Think of this as a convenience constructor that accepts dimension lists
# and takes care of adding:
# Good defaults: Ensuring correct initializations are used. He for ReLU, Xavier for tanh, etc.
#                Reasonable hyperparams for things like dropout, conv windows, stride, etc.
# Tricks for faster training: BatchNorm.
# Tricks for regularization: Dropout.
# TODO: Enables a quick unit test through main function.
# TODO: Elaborate on design decision for why no nn.Sequential or nn.ModuleList (instead just dict)

class FCBlock(nn.Module):

    """
    In: 
    Out: 
    """

    # TODO: should this be 'hidden_layer_dim_list'?
    # I think yes, but test out
    def __init__(self, layer_dim_list, drop_p=0.2):
        super().__init__()
        # TODO: clean these up, add set defaults?
        self.layer_dim_list = layer_dim_list
        self.drop_p = drop_p
        self.module_dict = nn.ModuleDict() 
        # TODO: rather than single dropout value, consider passing list.
        # Alternatively, autopopulate list with more dropout on wider layers

        if len(layer_dim_list) < 2:
            raise ValueError('len(layer_dim_list) must be >= 2')

        self._init_layers()
             

    # For each i/o dimension pair, build:
    # This ordering is still contested.
    # Rationale: Dropout always after activation.
    # BN should 'reset' input for next layer. It is meant to stabilize the non-linearity of the layer.
    # Find out if there is a 'disharmony' between dropout and batchnorm
    # FC -> ReLU -> Dropout -> BatchNorm
    def _init_layers(self):
        io_tuple_list = zip(self.layer_dim_list[:-1], self.layer_dim_list[1:])
        for i, (in_dim, out_dim) in enumerate(io_tuple_list):
            self.module_dict['fc'+str(i)]       =  nn.Linear(in_dim, out_dim)
            self.module_dict['fc'+str(i)+'_bn'] =  nn.BatchNorm1d(out_dim)

    # TODO: forward always comes last, its easy to find thatway (_init comes first)
    def forward(self, x):
        for i, _ in enumerate(self.layer_dim_list[:-1]):
            x = self.module_dict['fc'+str(i)](x)
            x = F.relu(x)
            x = F.dropout(x, self.drop_p)
            x = self.module_dict['fc'+str(i)+'_bn'](x)

        return x

class ConvBlock(nn.Module):

    """
    In: C x H x W Image
        hparams_dict contains: {filter_sz, filter_stride, pad, pool_sz, pool_stride}
    Out: 1D vector representation of dimension LATENT_DIM
    """
    # TODO: if necessary, [[kernel_sz, stride_sz, pad_sz] for L1, ... [.,.,.] for LN-1]]

    def __init__(self, img_dims, hidden_layer_dim_list, hparams_dict=None):
        super().__init__()
        self.module_dict = nn.ModuleDict() 
        self.hparams = hparams_dict
        if self.hparams is None: self._set_default_hparams()

        if len(hidden_layer_dim_list) < 1:
            #TODO: simplify this to check existance?
            raise ValueError('len(hidden_layer_dim_list) must be >= 1')

        self.layer_dim_list = [img_dims[0]] + hidden_layer_dim_list # first layer is img_ch dim 
        
        self._init_layers()
        self.out_dims = self._compute_output_dims(*img_dims)

    # These defaults take [H, W] to [H/2, W/2] at each layer
    def _set_default_hparams(self):
        self.hparams_dict = {
            'filter_size':3,
            'filter_stride':2,
            'filter_pad':1,
        }

    # For each i/o dimension pair, build:
    # Input -> Conv2d -> ReLU -> BatchNorm2d 
    def _init_layers(self):
        io_tuple_list = zip(self.layer_dim_list[:-1], self.layer_dim_list[1:])
        for i, (in_dim, out_dim) in enumerate(io_tuple_list):
            # defaults imply 'same' convs
            self.module_dict['conv'+str(i)] = nn.Conv2d(in_channels=in_dim,
                                                        out_channels=out_dim, 
                                                        kernel_size=self.hparams_dict['filter_size'],
                                                        stride=self.hparams_dict['filter_stride'],
                                                        padding=self.hparams_dict['filter_pad'])
            self.module_dict['conv'+str(i)+'_bn'] = nn.BatchNorm2d(out_dim)


    def _compute_output_dims(self, img_ch_dim, img_height, img_width):

        '''
        IN:
        OUT:
        '''

        x = torch.rand(1, img_ch_dim, img_height, img_width)

        # don't call relu or batchnorm since they never affect dimensions
        for i, _ in enumerate(self.layer_dim_list[:-1]):
            x = self.module_dict['conv'+str(i)](x)
        
        return x.size(-3), x.size(-2), x.size(-1)

    def forward(self, x):
        for i, _ in enumerate(self.layer_dim_list[:-1]):
            x = self.module_dict['conv'+str(i)](x)
            x = F.relu(x)
            x = self.module_dict['conv'+str(i)+'_bn'](x)

        #x = x.view(x.size(0), -1)
        return x 


class UpConvBlock(nn.Module):

    """
    In: vector representation of dimension LATENT_DIM 
    Out: C x H x W Image
    """
    # TODO: write a piece of code that reverse whatever encoder is being used?

    def __init__(self, img_dims, hidden_layer_dim_list, hparams_dict=None):
        super().__init__()
        self.module_dict = nn.ModuleDict() 
        self.hparams = hparams_dict
        if self.hparams is None: self._set_default_hparams()

        if len(hidden_layer_dim_list) < 1:
            #TODO: simplify this to check existance?
            raise ValueError('len(hidden_layer_dim_list) must be >= 1')

        self.layer_dim_list = hidden_layer_dim_list + [img_dims[0]] # last layer dim is img_ch dim
        self._init_layers()
        self.in_dims = self._compute_upconv_input_dims(*img_dims)

    # TODO: possibly calculate output_pad so that it results in final '[img_ch_dim, img_height, img_width]' output
    # These defaults effectively pad to nearest power of 2 and take [H, W] to [2H, 2W] at each layer
    def _set_default_hparams(self):
        self.hparams_dict = {
            'filter_size':3,
            'filter_stride':2,
            'filter_pad':1,
            'output_pad':1,
        }

    def _init_layers(self):
        io_tuple_list = zip(self.layer_dim_list[:-1], self.layer_dim_list[1:])
        for i, (in_dim, out_dim) in enumerate(io_tuple_list):
            self.module_dict['upconv'+str(i)] = nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, 
                                                                   kernel_size=self.hparams_dict['filter_size'],
                                                                   stride=self.hparams_dict['filter_stride'],
                                                                   padding=self.hparams_dict['filter_pad'],
                                                                   output_padding=self.hparams_dict['output_pad'])
            self.module_dict['upconv'+str(i)+'_bn'] = nn.BatchNorm2d(out_dim)


    #TODO: in the vae code this is the required input size of deconv block
    #TODO: remind myself where i need this
    def _compute_upconv_input_dims(self, img_ch_dim, img_height, img_width):
        x = torch.rand(1, img_ch_dim, img_height, img_width)

        for i, in range(len(self.layer_dim_list),0):
            x = self.module_dict['upconv'+str(i)](x)

        return x.size(-3), x.size(-2), x.size(-1)

    def forward(self, x):
        #x = x.view(x.size(0), self.in_dim_ch, self.in_dim_h, self.in_dim_w)
        # Note this loop goes to second to last element. Last element handled below
        for i, _ in enumerate(self.layer_dim_list[:-2]):
            x = self.module_dict['upconv'+str(i)](x)
            x = F.relu(x)
            x = self.module_dict['upconv'+str(i)+'_bn'](x) #TODO: BN on final layer?

        #TODO: decide if this should go here or in higher level function
        #Or if I should 'pass' activation function to this module
        #Final layer
        x = self.module_dict['upconv'+str(i+1)](x)
        x = F.sigmoid(x)

        #x = x.view(x.size(0), -1)
        return x 
