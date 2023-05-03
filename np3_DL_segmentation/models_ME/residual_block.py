import torch.nn as nn

import MinkowskiEngine as ME

def get_norm(norm_type, n_channels, bn_momentum=0.1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(n_channels)
  elif norm_type == 'INBN':
    return nn.Sequential(
        ME.MinkowskiInstanceNorm(n_channels),
        ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
  else:
    raise ValueError(f'Norm type: {norm_type} not supported')

class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               dimension=3,
               bn_momentum=0.1):
    super(BasicBlockBase, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum)
    self.conv2 = ME.MinkowskiConvolution(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        dimension=dimension)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'

class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type):
  if norm_type == 'BN':
    return BasicBlockBN
  elif norm_type == 'IN':
    return BasicBlockIN
  else:
    raise ValueError(f'Type {norm_type}, not defined')

# def get_block(norm_type,
#             inplanes,
#             planes,
#             stride=1,
#             dilation=1,
#             downsample=None,
#             bn_momentum=0.1,
#             D=3):
#   if norm_type == 'BN':
#       return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
#   elif norm_type == 'IN':
#       return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
#   else:
#       raise ValueError(f'Type {norm_type}, not defined')

