# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
# import torch
# import torch.nn as nn
# from torch.optim import SGD

import MinkowskiEngine as ME
import sys
# from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from models_ME.resnet import ResNetBase
from models_ME.residual_block import get_block, get_norm


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    NORM_TYPE = 'BN'
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    KERNEL_SIZE = (2, 2, 2, 2, 2, 2, 2, 2)  # non block
    STRIDE = (2, 2, 2, 2, 2, 2, 2, 2)  # non block
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, config, D=3):
        self.BLOCK = get_block(self.NORM_TYPE)
        ResNetBase.__init__(self, in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        # check if the dilations by block have the length equals the number of layers
        check_dilations = [len(self.DILATIONS[i]) == self.LAYERS[i] if type(self.DILATIONS[i]) is list else True
                           for i in range(len(self.DILATIONS))]
        if not all(check_dilations):
            print('LAYERS:', self.LAYERS)
            print('DILATIONS:', self.DILATIONS)
            sys.exit("ERROR: wrong number of dilation by block. "+
                     "It should match the number of layers of each block or be a single int value.")
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=config.conv1_kernel_size, dimension=D)

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, bn_momentum=config.bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=self.KERNEL_SIZE[0], stride=self.STRIDE[0], dimension=D)
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, bn_momentum=config.bn_momentum)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0], dilation=self.DILATIONS[0], bn_momentum=config.bn_momentum)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=self.KERNEL_SIZE[1], stride=self.STRIDE[1], dimension=D)
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, bn_momentum=config.bn_momentum)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1], dilation=self.DILATIONS[1], bn_momentum=config.bn_momentum)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=self.KERNEL_SIZE[2], stride=self.STRIDE[2], dimension=D)

        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, bn_momentum=config.bn_momentum)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2], dilation=self.DILATIONS[2], bn_momentum=config.bn_momentum)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=self.KERNEL_SIZE[3], stride=self.STRIDE[3], dimension=D)
        # self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=config.bn_momentum)
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, bn_momentum=config.bn_momentum)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3], dilation=self.DILATIONS[3], bn_momentum=config.bn_momentum)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=self.KERNEL_SIZE[4], stride=self.STRIDE[4], dimension=D)
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], bn_momentum=config.bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4], dilation=self.DILATIONS[4], bn_momentum=config.bn_momentum)
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=self.KERNEL_SIZE[5], stride=self.STRIDE[5], dimension=D)
        self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], bn_momentum=config.bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5], dilation=self.DILATIONS[5], bn_momentum=config.bn_momentum)
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=self.KERNEL_SIZE[6], stride=self.STRIDE[6], dimension=D)
        self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], bn_momentum=config.bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6], dilation=self.DILATIONS[6], bn_momentum=config.bn_momentum)
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=self.KERNEL_SIZE[7], stride=self.STRIDE[7], dimension=D)
        self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], bn_momentum=config.bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7], dilation=self.DILATIONS[7], bn_momentum=config.bn_momentum)

        if ME.__version__.split('.')[1] < '5':
            self.final = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                has_bias=True,
                dimension=D)
        else:
            self.final = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                bias=True,
                dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)


class MinkUNet14(MinkUNetBase):
    NORM_TYPE = 'BN'
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    NORM_TYPE = 'BN'
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    NORM_TYPE = 'BN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    NORM_TYPE = 'BN'
    LAYERS = (2, 3, 4, 22, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    NORM_TYPE = 'BN'
    LAYERS = (4, 6, 9, 44, 4, 4, 4, 4)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):

    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class MinkUNet34CIN(MinkUNet34):
    NORM_TYPE = 'IN'
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class MinkUNet34CIN_CONVATROUS_HYBRID(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34C_CONVATROUS_HYBRID(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'BN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet34CIN_CONVATROUS_HYBRID_SMALL(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = (1, [1,2,3], [1,2,3,5], [1,2,3,5,9,13], [1,13], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_HYBRID32(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 32, 32, 32, 32, 32, 32, 32)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_HYBRID128(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (128, 128, 128, 128, 128, 128, 128, 128)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_HYBRID2(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_BLOCK(MinkUNet34):
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = (1, 1, 2, 4, 8, 16, 1, 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_HYBRID_HIGHRESO(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = (1, [1,5,11], [1,5,9,19], [1,3,5,11,19,43], [1,19], [1,9], [1,5], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet50IN_CONVATROUS_HYBRID64(MinkUNet50):
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)
    PLANES = (64, 64, 64, 64, 64, 64, 64, 64)
    DILATIONS = ([1,2], [1, 2, 9], [1, 2, 5, 11], list(range(1,24)), [1, 11], [1, 5], [1, 2], 1)
    # KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CIN_CONVATROUS_HYBRID_HIGHRESO32(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'IN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 32, 32, 32, 32, 32, 32, 32)
    DILATIONS = (1, [1,5,11], [1,5,9,19], [1,3,5,11,19,43], [1,19], [1,9], [1,5], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet34CBN_CONVATROUS_HYBRID32(MinkUNet34):
    # apply dilation increasing from 1 to a given value inside each block by layer
    # sawtooth wave /|/|/|
    NORM_TYPE = 'BN'
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 32, 32, 32, 32, 32, 32, 32)
    DILATIONS = (1, [1,2,5], [1,2,5,9], [1,2,3,5,9,17], [1,9], [1,5], [1,2], 1)
    #KERNEL_SIZE = (3, 3, 3, 3, 3, 3, 3, 3)  # non block
    STRIDE = (1, 1, 1, 1, 1, 1, 1, 1)

