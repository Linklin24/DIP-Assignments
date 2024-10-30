import torch.nn as nn
from torch.nn import init

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        ngf = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 64
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        ### FILL: add more CONV Layers

        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, 2*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 128
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*ngf, 4*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 256
            nn.BatchNorm2d(4*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(8*ngf, 4*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 256
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 128
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(2*ngf, ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 64
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.deconv8 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 3
            nn.Tanh()
        )

        self.init_weights()

    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x = self.conv8(x7)

        x = self.deconv1(x)
        x = x + x7
        x = self.deconv2(x)
        x = x + x6
        x = self.deconv3(x)
        x = x + x5
        x = self.deconv4(x)
        x = x + x4
        x = self.deconv5(x)
        x = x + x3
        x = self.deconv6(x)
        x = x + x2
        x = self.deconv7(x)
        x = x + x1
        output = self.deconv8(x)
        
        return output
    