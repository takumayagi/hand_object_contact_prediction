#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import argparse

from models.baseline import Fixed, IoU, UnionLSTMHO, UnionLSTMHORGB, UnionLSTMHOFlow
from models.gplc import CrUnionLSTMHO, CrUnionLSTMHORGB, CrUnionLSTMHOFlow


def get_args():
  parser = argparse.ArgumentParser()

  # Directory
  parser.add_argument('--out_dir', type=str, default="outputs")
  parser.add_argument('--dir_name', type=str, default="tmp")
  parser.add_argument('--nb_workers', type=int, default=16)
  parser.add_argument('--nb_eval_workers', type=int, default=8)

  # Dataset
  parser.add_argument('--image_dir', type=str, default="data/rgb_frames")
  parser.add_argument('--flow_dir', type=str, default="data/flow_frames")
  parser.add_argument('--data_dir', type=str, default="data/annotations")
  parser.add_argument('--train_vids', type=str, default="configs/noisy_train_vids.txt")
  parser.add_argument('--clean_vids', type=str, default="configs/trusted_train_vids.txt")
  parser.add_argument('--valid_vids', type=str, default="configs/valid_vids.txt")
  parser.add_argument('--test_vids', type=str, default="configs/test_vids.txt")
  parser.add_argument('--bound_th', type=int, default=6)
  parser.add_argument('--supervised', action='store_true')
  parser.add_argument('--modality', type=str, nargs="*", default=["rgb", "flow"])
  parser.add_argument('--anno_version', type=str, default="v5")

  # Model
  parser.add_argument('--model', type=str, default="")
  parser.add_argument('--nb_layers', type=int, default=4)

  # Training
  parser.add_argument('--nb_iters', type=int, default=25000)
  parser.add_argument('--iter_evaluation', type=int, default=5000)
  parser.add_argument('--iter_snapshot', type=int, default=5000)
  parser.add_argument('--iter_display', type=int, default=1000)
  parser.add_argument('--iter_visualize', type=int, default=1000)

  # Optimization
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--optimizer', type=str, default="adam")
  parser.add_argument('--min_lr', type=float, default=3e-8)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--lr_step_list', type=float, nargs="*", default=[15000, 20000])
  parser.add_argument('--momentum', type=float, default=0.99)

  # Data preprocessing
  parser.add_argument('--min_label_ratio', type=float, default=0.0)
  parser.add_argument('--only_boundary', action='store_true')
  parser.add_argument('--min_clean_label_ratio', type=float, default=0.25)

  # gPLC
  parser.add_argument('--plc', action='store_true')
  parser.add_argument('--iter_supervision', type=int, default=2500)
  parser.add_argument('--iter_warmup', type=int, default=2500)
  parser.add_argument('--init_delta', type=float, default=0.05)
  parser.add_argument('--max_delta', type=float, default=0.25)
  parser.add_argument('--delta_increment', type=float, default=0.025)
  parser.add_argument('--delta_th', type=float, default=0.005)
  parser.add_argument('--no_filter', action='store_true')
  parser.add_argument('--pl', action='store_true')
  parser.add_argument('--semisupervised', action='store_true')
  parser.add_argument('--update_clean', action='store_true')
  parser.add_argument('--confidence_penalty', type=float, default=0.0)
  parser.add_argument('--finetune_noisy_net', action='store_true')
  parser.add_argument('--asymp_labeled_flip', action='store_true')

  # Test
  parser.add_argument('--eval_set', type=str, default="valid")
  parser.add_argument('--resume', type=str, default="")

  # Visualization
  parser.add_argument('--vis', action='store_true')
  parser.add_argument('--vis_dir_name', type=str, default="vis_predictions")
  parser.add_argument('--fps', type=float, default=25.0)
  parser.add_argument('--ratio', type=int, default=2)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--save_model', action='store_true')
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--seed', type=int, default=1701)  # XXX
  args = parser.parse_args()

  return args


def get_model(args):
  base_model = eval(args.model)(args)
  return base_model
