import torch.nn as nn
from torch.nn import functional as F

from torch import nn as nn
import torch
# from until.tools import init_network_weights

def init_network_weights(model, init_type='normal', gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method {} is not implemented'.
                    format(init_type)
                )
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('InstanceNorm') != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        EPSILON = 1e-8 
        out = x.div(norm+EPSILON)  #加上EPSILON防止Normalize的时候除0
        return out
class ConvNet(nn.Module):

    def __init__(self, c_hidden=64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 4* c_hidden

        self.embedding = nn.Sequential(nn.Linear(256, 128))
        self.classifier = nn.Linear(128, 10)
        self.l2norm = Normalize(2)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        # x_output = x
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x_output = x

        x = F.max_pool2d(x, 2)
        x =  x.view(x.size(0), -1)

        return self.classifier(self.l2norm(self.embedding(x))), self.l2norm(self.embedding(x)), x,x_output



def cnn_digitsdg(init_type='kaiming'):
    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type=init_type)
    return model