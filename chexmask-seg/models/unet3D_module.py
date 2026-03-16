import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10


###############################################################################
# 3D Unet
# convolution - normalization - activation - dropout
###############################################################################


class Unet3D(nn.Module):
    def __init__(self,
                 n_layers: int = 2, 
                 n_scales: int = 3,  
                 init_ker: int = 4,
                 inp_chan: int = 1,
                 out_chan: int = 1,
                 dropout_rate: float = 0.05, 
                 norm: str = 'in',
                 activ: str = 'lrelu',
                 pad_type: str = 'zero',
                 ds_type: str = 'max',
                 us_type: str = 'transpose',
                 merge_type: str = 'sum',
                 last_layer: str = "ssn"):
        
        '''
        Args:
                n_layers (2): number of layers at scale. 
                n_scales (3): number of scales in U-Net
                init_ker (4): initial number of kerns, they are doubled at each scale
                inp_chan (1): input number of channels
                out_chan (1): output number of channels
                dropout_rate (0.05): dropout rate - if 0 no dropout is applied
                norm ('in'): normalization  'none', batchnorm - 'bn', instancenorm - 'in', layernorm - 'ln', groupnorm - 'gn'
                activ ('lrelu'): activation 'none', 'relu', 'lrelu', 'selu', 'tanh', 'prelu'
                pad_type ('zero'): padding type to keep input and output shape same in conv, 'zero' or 'replicate'
                ds_type ('max'): downsampling type in U-Net. 'max' or 'avg'
                us_type ('transpose'): upsampling type in U-Net. if 'transpose' then transpose convolution is used otherwise nearest neighbour upsampling followed by convolution
                merge_type ('sum'): merging of same scales in encoder and decodeer. 'sum' does summation of two path, 'mul' performs multiplication or two paths, while concate does concatination of two path (this increases parameters slights)
                last_layer ('ssn'): last layer of U-net. if 'ssn', last layer is identity and feature maps are passed, if 'aleatoric' then both mean and variance conv are calcuated and their output are passed, if anything else then only mean convolution is passed
        '''

        super(Unet3D, self).__init__()

        self.n_layers = n_layers
        self.n_scales = n_scales
        self.init_ker = init_ker
        self.inp_chan = inp_chan
        self.out_chan = out_chan
        self.dropout_rate = dropout_rate
        self.norm = norm
        self.activ = activ
        self.pad_type = pad_type
        self.ds_type = ds_type
        self.us_type = us_type
        self.merge_type = merge_type
        self.last_layer = last_layer

        if self.ds_type == 'avg':
            self.downsample = nn.AvgPool3d(kernel_size=2,
                                           stride=2,
                                           padding=0)
        elif self.ds_type == 'max':
            self.downsample = nn.MaxPool3d(kernel_size = 2,
                                           stride = 2,
                                           padding = 0)
        else:
            assert 0, "Unsupported pooling type: {}".format(ds_type)

        if self.last_layer == "aleatoric":
            # to map to desired output channel size - mu
            self.final = nn.Conv3d(in_channels = self.init_ker, 
                                   out_channels = self.out_chan, 
                                   kernel_size = 1, 
                                   stride = 1, 
                                   bias = True)   
            # to map to desired output channel size - var
            self.final_var = nn.Conv3d(in_channels = self.init_ker, 
                                   out_channels = self.out_chan, 
                                   kernel_size = 1, 
                                   stride = 1, 
                                   bias = True)   
        elif self.last_layer == "ssn":            
            self.final = nn.Identity() # if SSN then don't pass through last layer just pass the feature maps - they will be processed through individual conv functions for covariance matrics
        else:            
            self.final = nn.Conv3d(in_channels = self.init_ker, 
                                   out_channels = self.out_chan, 
                                   kernel_size = 1, 
                                   stride = 1, 
                                   bias = True)  # to map to desired output channel size

        self.enc = self.encoder()
        self.dec = self.decoder()

    def encoder(self):

        inp_dim = self.inp_chan
        out_dim = self.init_ker

        model = nn.ModuleList()

        model.append(convBlocks(inp_chan = inp_dim, 
                                out_chan = out_dim, 
                                kern_size = 3, 
                                norm = self.norm, 
                                activ = self.activ, 
                                pad_type = self.pad_type, 
                                dropout_rate = 0, 
                                n_layers = self.n_layers))

        for j in range(self.n_scales-1):

            inp_dim = out_dim
            out_dim *= 2

            model.append(convBlocks(inp_chan = inp_dim, 
                                    out_chan = out_dim, 
                                    kern_size = 3, 
                                    norm = self.norm, 
                                    activ = self.activ, 
                                    pad_type = self.pad_type, 
                                    dropout_rate = self.dropout_rate, 
                                    n_layers = self.n_layers))


        return model

    def decoder(self):

        inp_dim = self.init_ker * (2 ** (self.n_scales - 1))
        out_dim = self.init_ker * (2 ** (self.n_scales - 2))

        model = nn.ModuleList()

        for _ in range(self.n_scales - 2):

            model.append(convTransBlocks(inp_chan = inp_dim, 
                                         out_chan = out_dim, 
                                         kern_size = 3, 
                                         norm = self.norm, 
                                         activ = self.activ, 
                                         pad_type = self.pad_type, 
                                         dropout_rate = self.dropout_rate, 
                                         n_layers = self.n_layers, 
                                         us_type = self.us_type, 
                                         merge_type = self.merge_type))

            inp_dim //= 2
            out_dim //= 2

        model.append(convTransBlocks(inp_chan = inp_dim, 
                                     out_chan = out_dim, 
                                     kern_size = 3, 
                                     norm = self.norm, 
                                     activ = self.activ, 
                                     pad_type = self.pad_type, 
                                     dropout_rate = 0, 
                                     n_layers = self.n_layers, 
                                     us_type = self.us_type, 
                                     merge_type = self.merge_type))

        return model

    def forward(self, 
                x: torch.tensor):

        enc_output = []

        for i in range(self.n_scales):
            x = self.enc[i](x)          # take each scale encoder and pass input (x) through it
            enc_output += [x]           # store output for decoder part
            if i < self.n_scales-1:     # if not last scale than apply downsampling
                x = self.downsample(x)

        for i in range(self.n_scales-1):
            x = self.dec[i](x, enc_output[abs(i-self.n_scales+2)])  # pass through deocoder scale with its corresponding encoder output

        output = self.final(x)    # pass through last layer
        
        if self.last_layer == "aleatoric":
            output_var = self.final_var(x)
            return [output, output_var]  # return
        else: 
            return output



#####################################################################################
## convBlocks and convTransBlocks
######################################################################################

class convBlocks(nn.Module):
    def __init__(self,
                 inp_chan: int = 4,
                 out_chan: int = 1,
                 kern_size: int = 3,
                 norm: str = 'bn',
                 activ: str = 'relu',
                 pad_type: str = 'zero',
                 dropout_rate: int = 0,
                 n_layers: int = 2):

        '''
        Args:
                inp_chan (4): input number of channels
                out_chan (1): output number of channels
                kern_size (3): size of convolution layer
                norm ('in'): normalization  'none', batchnorm - 'bn', instancenorm - 'in', layernorm - 'ln', groupnorm - 'gn'
                activ ('lrelu'): activation 'none', 'relu', 'lrelu', 'selu', 'tanh', 'prelu'
                pad_type ('zero'): padding type to keep input and output shape same in conv, 'zero' or 'replicate'
                dropout_rate (0.05): dropout rate - if 0 no dropout is applied
                n_layers (2): number of layers in the convblocks. 
        '''

        super(convBlocks, self).__init__()

        self.model = []

        self.model += [conv3dBlock(inp_chan = inp_chan, 
                                   out_chan = out_chan, 
                                   kern_size = kern_size,
                                   stride = 1, 
                                   padding = 1, 
                                   norm = norm, 
                                   activ = activ, 
                                   pad_type = pad_type, 
                                   dropout_rate = 0)]

        for _ in range(n_layers-2):
            self.model += [conv3dBlock(inp_chan = out_chan, 
                                       out_chan = out_chan, 
                                       kern_size = kern_size, 
                                       stride = 1, 
                                       padding = 1, 
                                       norm = 'none', 
                                       activ = activ, 
                                       pad_type = pad_type, 
                                       dropout_rate = 0)]

        self.model += [conv3dBlock(inp_chan = out_chan, 
                                   out_chan = out_chan, 
                                   kern_size = kern_size, 
                                   stride = 1, 
                                   padding = 1, 
                                   norm='none', 
                                   activ = activ, 
                                   pad_type = pad_type, 
                                   dropout_rate = dropout_rate)]

        self.model = nn.Sequential(*self.model)

    def forward(self, 
                x: torch.tensor):

        return self.model(x)


class convTransBlocks(nn.Module):
    def __init__(self,
                 inp_chan: int = 4,
                 out_chan: int = 1,
                 kern_size: int = 3,
                 norm: str = 'bn',
                 activ: str = 'relu',
                 pad_type: str = 'zero',
                 dropout_rate: float = 0,
                 n_layers: int = 2,
                 us_type: str = 'transpose',
                 merge_type: str = 'sum'):

        '''
        Args:
                init_ker (4): initial number of kerns, they are doubled at each scale
                inp_chan (1): input number of channels
                out_chan (1): output number of channels
                kern_size (3): size of convolution kernel
                norm ('in'): normalization  'none', batchnorm - 'bn', instancenorm - 'in', layernorm - 'ln', groupnorm - 'gn'
                activ ('lrelu'): activation 'none', 'relu', 'lrelu', 'selu', 'tanh', 'prelu'
                pad_type ('zero'): padding type to keep input and output shape same in conv, 'zero' or 'replicate'
                dropout_rate (0.05): dropout rate - if 0 no dropout is applied
                n_layers (2): number of layers in the convTransblocks. 
                us_type ('transpose'): upsampling type in U-Net. if 'transpose' then transpose convolution is used otherwise nearest neighbour upsampling followed by convolution
                merge_type ('sum'): merging of same scales in encoder and decodeer. 'sum' does summation of two path, 'mul' performs multiplication or two paths, while concate does concatination of two path (this increases parameters slights)
        '''


        super(convTransBlocks, self).__init__()

        self.merge_type = merge_type

        if us_type == 'transpose':
            self.upsamp = convTranspose3dBlock(inp_chan = inp_chan, 
                                               out_chan = out_chan, 
                                               kern_size = 3, 
                                               stride = 2, 
                                               padding = 1, 
                                               output_padding = 1, 
                                               norm = 'none', 
                                               activ = activ, 
                                               dropout_rate = 0)
        else:
            m = []
            m += [nn.Upsample(scale_factor = 2, 
                              mode = 'nearest')]
            m += [conv3dBlock(inp_chan = inp_chan, 
                              out_chan = out_chan, 
                              kern_size = 3, 
                              stride = 1, 
                              padding = 1, 
                              activ = activ, 
                              pad_type=pad_type)]
            self.upsamp = nn.Sequential(*m)

        if self.merge_type == 'sum' or self.merge_type == 'mul':
            inp_chan = out_chan

        self.model = []

        self.model += [conv3dBlock(inp_chan = inp_chan, 
                                   out_chan = out_chan, 
                                   kern_size = kern_size, 
                                   stride = 1, 
                                   padding = 1, 
                                   norm = norm, 
                                   activ = activ, 
                                   pad_type = pad_type, 
                                   dropout_rate = 0)]

        for _ in range(n_layers - 2):
            self.model += [conv3dBlock(inp_chan = out_chan, 
                                       out_chan = out_chan, 
                                       kern_size = kern_size, 
                                       stride = 1, 
                                       padding = 1, 
                                       norm = 'none', 
                                       activ = activ, 
                                       pad_type = pad_type, 
                                       dropout_rate = 0)]

        self.model += [conv3dBlock(inp_chan = out_chan, 
                                   out_chan = out_chan, 
                                   kern_size = kern_size, 
                                   stride = 1, 
                                   padding = 1, 
                                   norm = 'none', 
                                   activ = activ, 
                                   pad_type = pad_type, 
                                   dropout_rate = dropout_rate)]

        self.model = nn.Sequential(*self.model)

    def forward(self, 
                x1: torch.tensor, 
                x2: torch.tensor):

        x1 = self.upsamp(x1)

        if self.merge_type == 'sum':
            x = x1+x2
        elif self.merge_type == 'mul':
            x = x1*x2
        else:
            x = torch.cat((x1, x2), 1)

        return self.model(x)


class conv3dBlock(nn.Module):
    def __init__(self,
                 inp_chan: int = 4,
                 out_chan: int = 1,
                 kern_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 norm: str = 'bn',
                 activ: str = 'relu',
                 pad_type: str = 'zero',
                 dropout_rate: float = 0):

        '''
        Args:
                inp_chan (4): input number of channels
                out_chan (1): output number of channels
                kern_size (3): size of convolution layer
                stride (1): convolution stride
                padding (1): convolution padding size
                norm ('in'): normalization  'none', batchnorm - 'bn', instancenorm - 'in', layernorm - 'ln', groupnorm - 'gn'
                activ ('lrelu'): activation 'none', 'relu', 'lrelu', 'selu', 'tanh', 'prelu'
                pad_type ('zero'): padding type to keep input and output shape same in conv, 'zero' or 'replicate'
                dropout_rate (0.05): dropout rate - if 0 no dropout is applied
        '''

        super(conv3dBlock, self).__init__()

        self.use_bias = True
        self.dropout_rate = dropout_rate

        # initialize padding
        if pad_type == 'replicate':
            self.pad = nn.ReplicationPad3d(padding=padding)
        elif pad_type == 'zero':
            self.pad = nn.ConstantPad3d(padding=padding, value=0)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = inp_chan
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(norm_dim//2,norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        elif activ == 'selu':
            self.activ = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'none':
            self.activ = None
        else:
            assert 0, "Unsupported activation: {}".format(activ)

        if dropout_rate > 0:
            self.dp = nn.Dropout3d(dropout_rate)
        else:
            self.dp = None

        # initialize convolution
        self.conv = nn.Conv3d(in_channels = inp_chan, 
                              out_channels = out_chan, 
                              kernel_size = kern_size, 
                              stride = stride, 
                              bias = self.use_bias)

    def forward(self, 
                x: torch.tensor):

        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        if self.dp:
            x = self.dp(x)

        return x


class convTranspose3dBlock(nn.Module):
    def __init__(self, 
                 inp_chan: int,
                 out_chan: int, 
                 kern_size: int, 
                 stride: int, 
                 padding: int = 0, 
                 output_padding: int = 0, 
                 norm: str = 'none', 
                 activ: str = 'relu', 
                 dropout_rate: float = 0.0):

        '''
        Args:
                inp_chan (4): input number of channels
                out_chan (1): output number of channels
                kern_size (3): size of convolution layer
                stride (1): convolution stride
                padding (1): convolution padding size
                output_padding (1): transpose convolution output padding size
                norm ('in'): normalization  'none', batchnorm - 'bn', instancenorm - 'in', layernorm - 'ln', groupnorm - 'gn'
                activ ('lrelu'): activation 'none', 'relu', 'lrelu', 'selu', 'tanh', 'prelu'
                dropout_rate (0.05): dropout rate - if 0 no dropout is applied
        '''

        super(convTranspose3dBlock, self).__init__()

        self.use_bias = True
        self.dropout_rate = dropout_rate

        # initialize normalization
        norm_dim = out_chan
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(norm_dim//2,norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        elif activ == 'selu':
            self.activ = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'none':
            self.activ = None
        else:
            assert 0, "Unsupported activation: {}".format(activ)

        if dropout_rate > 0:
            self.dp = nn.Dropout3d(dropout_rate)
        else:
            self.dp = None

        # initialize convolution
        self.convtp = nn.ConvTranspose3d(in_channels = inp_chan, 
                                         out_channels = out_chan, 
                                         kernel_size = kern_size, 
                                         stride = stride, 
                                         padding = padding, 
                                         output_padding = output_padding, 
                                         bias = self.use_bias)

    def forward(self, 
                x: torch.tensor):

        x = self.convtp(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        if self.dp:
            x = self.dp(x)

        return x