"""
Reimplementation of DeepCov and DeepCon contact prediction models

DeepCov:
Fully convolutional neural networks for protein residue-residue contact prediction
David T. Jones and Shaun M. Kandathil - University College London
https://github.com/psipred/DeepCov

DeepCon:
Dilated convolution network with dropout (best reported performing model, Fig.3d)
https://github.com/ba-lab/DEEPCON/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


RAW_CHANNELS = 441


# define conv layer block for ResNet-based models
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)


# Bottleneck block
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=1, dilation=1, dropout=0.3):
        super(BottleNeck, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3(out_channels, out_channels, padding=padding,
                             dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, stride=1, padding=2, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DeepCov(nn.Module):
    def __init__(self, block, layers, in_channels=441):
        super(DeepCov, self).__init__()
        self.in_channels = in_channels
        self.conv1x1 = nn.Conv2d(441, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2d = nn.MaxPool3d((2, 1, 1))
        self.conv5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.init_weights()
        self.layer1 = self.make_layers(block, 64, layers[0])

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1x1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv5x5.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.last_conv.weight, gain=nn.init.calculate_gain('relu'))

    def make_layers(self, block, out_channels, blocks):
        layers = []
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # maxout layer
        out = self.conv1x1(x)
        out = self.bn1(out)
        out = self.maxpool2d(out)

        # multiple layers
        out = self.layer1(out)
        # last layer needs to be of kernel size (1,1) and sigmoid activation
        out = self.bn2(out)
        out = self.last_conv(out)
        out = torch.sigmoid(out)

        return out


class DeepCon(nn.Module):
    def __init__(self, block, layers, in_channels=441):
        super(DeepCon, self).__init__()
        self.in_channels = in_channels
        self.conv1 = conv1x1(self.in_channels, 128)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 1, 1))
        self.bn1 = nn.BatchNorm2d(441)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1])
        self.conv_last = conv3x3(64, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv_last.weight,
                                gain=nn.init.calculate_gain('relu'))

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(64, out_channels, stride))
        self.in_channels = out_channels
        d_rate = 1
        for i in range(blocks):
            layers.append(block(out_channels, out_channels))
            layers.append(block(out_channels, out_channels, dilation=d_rate, padding=d_rate))
            if d_rate == 1:
                d_rate = 2
            elif d_rate == 2:
                d_rate = 4
            else:
                d_rate = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        # maxout layer
        out = self.bn1(x)  # (441, L, L) -> (441, L, L)
        out = self.relu(out)
        out = self.conv1(out)  # (441, L, L) -> (128, L, L)
        out = self.maxpool3d(out)  # (128, L, L) -> (64, L, L)
        # end of maxout layer

        # residue layers
        out = self.layer1(out)  # (64, L, L) -> (64, L, L)
        out = self.layer2(out)

        # last layers
        out = self.bn2(out)  # (64, L, L) -> (64, L, L)
        out = self.relu(out)
        out = self.conv_last(out)  # (64, L, L) -> (1, L, L)
        out = torch.sigmoid(out)
        return out
