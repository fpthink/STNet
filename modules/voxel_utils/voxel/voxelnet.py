from .torch_util import Conv2d,Conv3d
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from collections import OrderedDict
import logging
logger = logging.getLogger('global')

class FCN(nn.Module):
    def __init__(self, inplanes, planes):
        super(FCN, self).__init__()
        planes = int(planes/2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,  stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class VFE(nn.Module):
    def __init__(self, inplanes, planes):
        super(VFE, self).__init__()
        self.fcn1 = FCN(inplanes, planes)

    def forward(self, x):
        batch, channel, voxels, num_T = x.size()
        out = self.fcn1(x)
        point_wise_feature = F.max_pool2d(out, kernel_size=[1, num_T], stride=[1, num_T])
        logger.debug('point_wise_feature size: {}'.format(point_wise_feature.size()))
        out = torch.cat((out, point_wise_feature.repeat(1, 1, 1, num_T)), 1)
        logger.debug('VFE size: {}'.format(out.size()))
        return out

class Conv_Middle_layers(nn.Module):
    def __init__(self,inplanes):
        super(Conv_Middle_layers, self).__init__()
        self.conv1 = Conv3d(inplanes, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv2 = Conv3d(64, 64, stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = Conv3d(64, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv3d',nn.Conv3d(64,128,kernel_size=(3,1,1),stride=(1,1,1),padding=(0,0,0))),
            ('bn',nn.BatchNorm3d(128)),
            ('relu',nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out=self.conv4(out)
        shape = out.size()
        
        out=F.max_pool3d(out,kernel_size=[shape[2], 1, 1])
        out=out.squeeze(2)
        
        return out

