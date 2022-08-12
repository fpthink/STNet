import torch
import torch.nn as nn
from .torch_util import Conv2d
import logging
from collections import OrderedDict
from modules.voxel_utils.voxel.block import GroupCompletion
from utils.loss.utils import _tranpose_and_gather_feat
logger =logging.getLogger('global')
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class NaiveRpnHead(nn.Module):
    def __init__(self, inplanes, num_classes):
        '''
        Args:
            inplanes: input channel
            num_classes: as the name implies
            num_anchors: as the name implies
        '''
        super(NaiveRpnHead, self).__init__()
        self.num_classes = num_classes
        self.cls = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1))
        self.loc = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1))
        self.z_axis = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1))
        self.cls[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.loc)
        fill_fc_weights(self.z_axis)
    def forward(self, x):
        '''
        Args:
            x: [B, inplanes, h, w], input feature
        Return:
            pred_cls: [B, num_anchors, h, w]
            pred_loc: [B, num_anchors*4, h, w]
        '''
        pred_cls = self.cls(x)
        pred_loc = self.loc(x)
        pred_z_axis = self.z_axis(x)
        #(B,9,C)
        return pred_cls, pred_loc,pred_z_axis

class RPN(nn.Module):
    def __init__(self, num_classes=1):
        super(RPN, self).__init__()
        self.conv1_1 = Conv2d(128, 128, 2, padding=(1, 1))
        self.conv1_2 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv1_3 = Conv2d(128, 128, 1, padding=(1, 1))

        self.deconv1 = nn.Sequential(OrderedDict([('ConvTranspose',nn.ConvTranspose2d(128, 128, 2, 2, 0)),('bn',nn.BatchNorm2d(128)),('relu',nn.ReLU(inplace=True))]))
        self.deconv = nn.Sequential(OrderedDict([('ConvTranspose',nn.Conv2d(128, 128, 1, 1, 0)),('bn',nn.BatchNorm2d(128)),('relu',nn.ReLU(inplace=True))]))
        self.conv_final= nn.Sequential(OrderedDict([('Conv',nn.Conv2d(128*2,128,1,1)),('bn',nn.BatchNorm2d(128)),('relu',nn.ReLU(inplace=True))]))

        self.rpn_head = NaiveRpnHead(128, num_classes)

    def forward(self, x):
        deconv=self.deconv(x)
        out=self.conv1_1(x)
        out=self.conv1_2(out)
        out=self.conv1_3(out)
        out = self.deconv1(out)
        out=torch.cat([deconv,out],dim=1)
        out=self.conv_final(out)
        rpn_pred_cls, rpn_pred_loc,pred_z_axis = self.rpn_head(out)
        return rpn_pred_cls, rpn_pred_loc,pred_z_axis