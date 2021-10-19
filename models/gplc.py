#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp

import torch
import torch.nn as nn

from models.baseline import UnionLSTMHO, UnionLSTMHORGB, UnionLSTMHOFlow


class CrUnionLSTMHO(nn.Module):

  def __init__(self, args):
    super().__init__()

    self.noisy_net = UnionLSTMHO(args)
    self.clean_net = UnionLSTMHO(args)

    noisy_path = "pretrained/training/UnionLSTMHO_noisy_pretrain.pth"
    clean_path = "pretrained/training/UnionLSTMHO_clean_pretrain.pth"

    if osp.exists(noisy_path):
      print(f"Noisy Net: {noisy_path}")
      self.noisy_net.load_state_dict(torch.load(noisy_path))
    else:
      print(f"Noisy Net not found!!!: {noisy_path}")
    if osp.exists(clean_path):
      print(f"Clean Net: {clean_path}")
      self.clean_net.load_state_dict(torch.load(clean_path))
    else:
      print(f"Clean Net not found!!!: {clean_path}")

  def forward(self):
    pass


class CrUnionLSTMHORGB(nn.Module):

  def __init__(self, args):
    super().__init__()

    self.noisy_net = UnionLSTMHORGB(args)
    self.clean_net = UnionLSTMHORGB(args)

    noisy_path = "pretrained/training/UnionLSTMHORGB_noisy_pretrain.pth"
    clean_path = "pretrained/training/UnionLSTMHORGB_clean_pretrain.pth"

    if osp.exists(noisy_path):
      print(f"Noisy Net: {noisy_path}")
      self.noisy_net.load_state_dict(torch.load(noisy_path))
    else:
      print(f"Noisy Net not found!!!: {noisy_path}")
    if osp.exists(clean_path):
      print(f"Clean Net: {clean_path}")
      self.clean_net.load_state_dict(torch.load(clean_path))
    else:
      print(f"Clean Net not found!!!: {clean_path}")

  def forward(self):
    pass


class CrUnionLSTMHOFlow(nn.Module):

  def __init__(self, args):
    super().__init__()

    self.noisy_net = UnionLSTMHOFlow(args)
    self.clean_net = UnionLSTMHOFlow(args)

    noisy_path = "pretrained/training/UnionLSTMHOFlow_noisy_pretrain.pth"
    clean_path = "pretrained/training/UnionLSTMHOFlow_clean_pretrain.pth"

    if osp.exists(noisy_path):
      print(f"Noisy Net: {noisy_path}")
      self.noisy_net.load_state_dict(torch.load(noisy_path))
    else:
      print(f"Noisy Net not found!!!: {noisy_path}")
    if osp.exists(clean_path):
      print(f"Clean Net: {clean_path}")
      self.clean_net.load_state_dict(torch.load(clean_path))
    else:
      print(f"Clean Net not found!!!: {clean_path}")
