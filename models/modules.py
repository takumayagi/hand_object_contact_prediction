#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import torch

import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
  def forward(self, input):
    '''
    Note that input.size(0) is usually the batch size.
    So what it does is that given any input with input.size(0) # of batches,
    will flatten to be 1 * nb_elements.
    '''
    batch_size = input.size(0)
    out = input.contiguous().view(batch_size,-1)
    return out # (batch_size, *size)


class FeatureBlock(nn.Module):
  def __init__(self, n_in, n_hid, n_out):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d((2, 2))
    )

  def forward(self, x):
    return self.layers(x)


class FeatureBranch(nn.Module):
  """
  Follow [Xie+, CVPR'19]
  """
  def __init__(self, in_channels=3, out_channels=128):
    super().__init__()
    self.layers = nn.ModuleList([
      FeatureBlock(in_channels, 32, 32),  # 1->2x
      FeatureBlock(32, 64, 64),  # 2->4x
      FeatureBlock(64, 128, 128),  # 4->8x
      FeatureBlock(128, 128, out_channels),  # 8->16x
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class FeatureBranch2(nn.Module):
  """
  Follow [Xie+, CVPR'19]
  """
  def __init__(self, in_channels=3, out_channels=128):
    super().__init__()
    self.layers = nn.ModuleList([
      FeatureBlock(in_channels, 32, 32),  # 1->2x
      FeatureBlock(32, 64, 64),  # 2->4x
      FeatureBlock(64, 128, out_channels),  # 4->8x
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class CNNEncoder(nn.Module):
  def __init__(self, n_in, n_hid):
    super().__init__()

    # Backbones
    self.cnn = FeatureBranch(n_in, n_hid)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, hs):
    hs = self.cnn(hs)
    hs = torch.flatten(self.avgpool(hs), 1)

    return hs


class FeatureBlockGroupNorm(nn.Module):
  def __init__(self, n_in, n_hid, n_out, kernel_size=3, padding=1, pool_size=2, no_pool=False, lrelu=False):
    super().__init__()
    if no_pool:
      self.layers = nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(1, n_out),
        nn.LeakyReLU() if lrelu else nn.ReLU(),
      )
    else:
      self.layers = nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(1, n_out),
        nn.LeakyReLU() if lrelu else nn.ReLU(),
        nn.MaxPool2d((pool_size, pool_size))
      )

  def forward(self, x):
    return self.layers(x)


class CNNEncoderGroupNorm(nn.Module):
  def __init__(self, n_in):
    super().__init__()

    # Backbones
    self.layers = nn.ModuleList([
      FeatureBlockGroupNorm(n_in, 32, 32),  # 1->2x
      FeatureBlockGroupNorm(32, 64, 64),  # 2->4x
    ])

  def forward(self, hs):
    for layer in self.layers:
      hs = layer(hs)

    return hs


class CNNEncoderGroupNorm2(nn.Module):
  def __init__(self, n_in):
    super().__init__()

    # Backbones
    self.layers = nn.ModuleList([
      FeatureBlockGroupNorm(n_in, 32, 32),  # 1->2x
      FeatureBlockGroupNorm(32, 64, 64, no_pool=True),  # 2->4x
    ])

  def forward(self, hs):
    for layer in self.layers:
      hs = layer(hs)

    return hs


class CNNEncoderGroupNorm3(nn.Module):
  def __init__(self, n_in):
    super().__init__()

    # Backbones
    self.layers = nn.ModuleList([
      FeatureBlockGroupNorm(n_in, 128, 128),  # 4->8x
      FeatureBlockGroupNorm(128, 128, 256),  # 8->16x
    ])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, hs):
    for layer in self.layers:
      hs = layer(hs)
    hs = torch.flatten(self.avgpool(hs), 1)
    return hs
