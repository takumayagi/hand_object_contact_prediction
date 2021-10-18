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

import torchvision.models as models
import torchvision.ops as ops

from models.resnet import resnet50_backbone
from models.modules import Flatten, FeatureBranch, CNNEncoder, FeatureBranch2, CNNEncoderGroupNorm, CNNEncoderGroupNorm2, CNNEncoderGroupNorm3, FeatureBlockGroupNorm


class Fixed(nn.Module):

  def __init__(self, args):
    super().__init__()

  def forward(self, batch):
    return torch.ones_like(batch["labels"][0], device=batch["labels"].device).float()


class IoU(nn.Module):

  def __init__(self, args):
    super().__init__()

  def forward(self, batch):
    return (batch["mask_ious"][0] > 0).float()


class UnionLSTMHO(nn.Module):

  def __init__(self, args, lrelu=False):
    super().__init__()

    self.rgb_encoder = nn.Sequential(
      FeatureBlockGroupNorm(3, 32, 32, no_pool=True, lrelu=lrelu),  # 1x
      FeatureBlockGroupNorm(32, 64, 64, lrelu=lrelu),  # 1x->2x
      FeatureBlockGroupNorm(64, 128, 128, no_pool=True, lrelu=lrelu),  # 2x
    )
    self.flow_encoder = nn.Sequential(
      FeatureBlockGroupNorm(3, 32, 32, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(32, 128, 128, no_pool=True, lrelu=lrelu),  # 2x
    )
    self.fusion_encoder = nn.Sequential(
      FeatureBlockGroupNorm(256, 256, 256, lrelu=lrelu),  # 2->4x
      FeatureBlockGroupNorm(256, 256, 256, lrelu=lrelu),  # 4->8x
      nn.AdaptiveAvgPool2d((1, 1)),
      Flatten()
    )

    self.spatial_module = nn.Sequential(
      nn.Conv2d(4, 96, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(96, 128, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(128, 64, kernel_size=8)
    )
    self.fc1 = nn.Sequential(
      nn.Linear(256 + 64, 128),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
    )
    self.lstm = nn.LSTM(128, 64, num_layers=args.nb_layers, bidirectional=True)
    self.fc2 = nn.Sequential(
      nn.Linear(128, 64),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(64, 32),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(32, 1)
    )

    h_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    c_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    self.h_0 = nn.Parameter(h_0, requires_grad=True)
    self.c_0 = nn.Parameter(c_0, requires_grad=True)

  def forward(self, batch):
    h_rgb = self.rgb_encoder(batch["union_imgs"][0])
    h_flow = self.flow_encoder(batch["union_flows"][0])

    hs = self.fusion_encoder(torch.cat((h_rgb, h_flow), dim=1))
    dual_masks = torch.stack((
      batch["hand_masks"][0],
      batch["obj_bbox_masks"][0],
      batch["other_hand_masks"][0],
      batch["other_bbox_masks"][0]
    ), dim=1)
    h_spa = torch.flatten(self.spatial_module(dual_masks), 1)
    hs = torch.cat((hs, h_spa), dim=1)

    hs = self.fc1(hs)
    hs, _ = self.lstm(hs.unsqueeze(1), (self.h_0, self.c_0))
    hs = self.fc2(hs.squeeze(1))

    return torch.sigmoid(hs)[..., 0]


class UnionLSTMHORGB(nn.Module):

  def __init__(self, args, lrelu=False):
    super().__init__()

    self.rgb_encoder = nn.Sequential(
      FeatureBlockGroupNorm(3, 32, 32, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(32, 64, 64, no_pool=True, lrelu=lrelu),  # 2->4x
      FeatureBlockGroupNorm(64, 128, 128, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(128, 128, 128, no_pool=True, lrelu=lrelu),  # 2->4x
    )
    self.fusion_encoder = nn.Sequential(
      FeatureBlockGroupNorm(128, 256, 256, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(256, 256, 256, lrelu=lrelu),  # 1->2x
      nn.AdaptiveAvgPool2d((1, 1)),
      Flatten()
    )

    self.spatial_module = nn.Sequential(
      nn.Conv2d(4, 96, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(96, 128, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(128, 64, kernel_size=8)
    )
    self.fc1 = nn.Sequential(
      nn.Linear(256 + 64, 128),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
    )
    self.lstm = nn.LSTM(128, 64, num_layers=args.nb_layers, bidirectional=True)
    self.fc2 = nn.Sequential(
      nn.Linear(128, 64),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(64, 32),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(32, 1)
    )

    h_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    c_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    self.h_0 = nn.Parameter(h_0, requires_grad=True)
    self.c_0 = nn.Parameter(c_0, requires_grad=True)

  def forward(self, batch):
    h_rgb = self.rgb_encoder(batch["union_imgs"][0])
    hs = self.fusion_encoder(h_rgb)
    dual_masks = torch.stack((
      batch["hand_masks"][0],
      batch["obj_bbox_masks"][0],
      batch["other_hand_masks"][0],
      batch["other_bbox_masks"][0]
    ), dim=1)
    h_spa = torch.flatten(self.spatial_module(dual_masks), 1)
    hs = torch.cat((hs, h_spa), dim=1)

    hs = self.fc1(hs)
    hs, _ = self.lstm(hs.unsqueeze(1), (self.h_0, self.c_0))
    hs = self.fc2(hs.squeeze(1))

    return torch.sigmoid(hs)[..., 0]


class UnionLSTMHOFlow(nn.Module):

  def __init__(self, args, lrelu=False):
    super().__init__()

    self.flow_encoder = nn.Sequential(
      FeatureBlockGroupNorm(3, 32, 32, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(32, 128, 128, no_pool=True, lrelu=lrelu),  # 2->4x
    )
    self.fusion_encoder = nn.Sequential(
      FeatureBlockGroupNorm(128, 256, 256, lrelu=lrelu),  # 1->2x
      FeatureBlockGroupNorm(256, 256, 256, lrelu=lrelu),  # 1->2x
      nn.AdaptiveAvgPool2d((1, 1)),
      Flatten()
    )

    self.spatial_module = nn.Sequential(
      nn.Conv2d(4, 96, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(96, 128, kernel_size=5, padding=2, stride=2),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Conv2d(128, 64, kernel_size=8)
    )
    self.fc1 = nn.Sequential(
      nn.Linear(256 + 64, 128),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
    )
    self.lstm = nn.LSTM(128, 64, num_layers=args.nb_layers, bidirectional=True)
    self.fc2 = nn.Sequential(
      nn.Linear(128, 64),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(64, 32),
      nn.LeakyReLU() if lrelu else nn.ReLU(),
      nn.Linear(32, 1)
    )

    h_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    c_0 = torch.zeros((args.nb_layers * 2, 1, 64), dtype=torch.float32)
    self.h_0 = nn.Parameter(h_0, requires_grad=True)
    self.c_0 = nn.Parameter(c_0, requires_grad=True)

  def forward(self, batch):
    h_flow = self.flow_encoder(batch["union_flows"][0])
    hs = self.fusion_encoder(h_flow)
    dual_masks = torch.stack((
      batch["hand_masks"][0],
      batch["obj_bbox_masks"][0],
      batch["other_hand_masks"][0],
      batch["other_bbox_masks"][0]
    ), dim=1)
    h_spa = torch.flatten(self.spatial_module(dual_masks), 1)
    hs = torch.cat((hs, h_spa), dim=1)

    hs = self.fc1(hs)
    hs, _ = self.lstm(hs.unsqueeze(1), (self.h_0, self.c_0))
    hs = self.fc2(hs.squeeze(1))

    return torch.sigmoid(hs)[..., 0]
