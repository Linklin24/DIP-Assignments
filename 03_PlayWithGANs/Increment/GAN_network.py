import torch
import torch.nn as nn
from torch.nn import init

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

class GeneratorNetwork(nn.Module):

    def __init__(self, ngf=64):
        super().__init__()

        self.downconv1 = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 64
        )
        self.downconv2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, 2*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 128
            nn.BatchNorm2d(2*ngf),
        )
        self.downconv3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*ngf, 4*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 256
            nn.BatchNorm2d(4*ngf),
        )
        self.downconv4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 512
            nn.BatchNorm2d(8*ngf),
        )
        self.downconv5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
        )
        self.downconv6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
        )
        self.downconv7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
        )
        self.downconv8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
        )
        self.upconv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 512
            nn.BatchNorm2d(8*ngf),
        )
        self.upconv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 1024, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.Dropout(0.5),
        )
        self.upconv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 1024, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.Dropout(0.5),
        )
        self.upconv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 1024, Output channels: 512
            nn.BatchNorm2d(8*ngf),
            nn.Dropout(0.5),
        )
        self.upconv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16*ngf, 4*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 1024, Output channels: 256
            nn.BatchNorm2d(4*ngf),
        )
        self.upconv6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8*ngf, 2*ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 512, Output channels: 128
            nn.BatchNorm2d(2*ngf),
        )
        self.upconv7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4*ngf, ngf, kernel_size=4, stride=2, padding=1),  # Input channels: 256, Output channels: 64
            nn.BatchNorm2d(ngf),
        )
        self.upconv8 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*ngf, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 128, Output channels: 3
            nn.Tanh(),
        )

        init_weights(self)
    
    def forward(self, x):
        x1 = self.downconv1(x)
        x2 = self.downconv2(x1)
        x3 = self.downconv3(x2)
        x4 = self.downconv4(x3)
        x5 = self.downconv5(x4)
        x6 = self.downconv6(x5)
        x7 = self.downconv7(x6)
        x = self.downconv8(x7)

        x = self.upconv1(x)
        x = self.upconv2(torch.cat([x, x7], 1))
        x = self.upconv3(torch.cat([x, x6], 1))
        x = self.upconv4(torch.cat([x, x5], 1))
        x = self.upconv5(torch.cat([x, x4], 1))
        x = self.upconv6(torch.cat([x, x3], 1))
        x = self.upconv7(torch.cat([x, x2], 1))
        x = self.upconv8(torch.cat([x, x1], 1))
        
        return x

class DiscriminatorNetwork(nn.Module):

    def __init__(self, ndf=64):
        super().__init__()

        sequence = [nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        sequence += [
                nn.Conv2d(ndf, 2*ndf, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(2*ndf),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [
                nn.Conv2d(2*ndf, 4*ndf, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(4*ndf),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [
                nn.Conv2d(4*ndf, 8*ndf, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm2d(8*ndf),
                nn.LeakyReLU(0.2, True)
            ]
        sequence += [nn.Conv2d(8*ndf, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

        init_weights(self)

    def forward(self, x):
        return self.model(x)
    
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    