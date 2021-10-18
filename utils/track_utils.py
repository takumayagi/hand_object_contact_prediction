#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np


def fix_bb(bb, minv):
  return [x + minv if idx != 4 else x for idx, x in enumerate(bb)]


def calc_iou(bb_test, bb_gt):
  """Computes IoU between two bboxes in the form [x1,y1,x2,y2]
  """
  min_value = np.min([np.min(bb_test[:4]), np.min(bb_gt[:4])])
  bb_test = fix_bb(bb_test, min_value)
  bb_gt = fix_bb(bb_gt, min_value)

  x1 = np.maximum(bb_test[0], bb_gt[0])
  y1 = np.maximum(bb_test[1], bb_gt[1])
  x2 = np.minimum(bb_test[2], bb_gt[2])
  y2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., x2 - x1)
  h = np.maximum(0., y2 - y1)
  wh = w * h
  return wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
               (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)


def calc_hand_iou(hand_mask, bb):
  """Computes IoU between a mask and a bounding box
  """
  hand_mask = hand_mask.astype(np.bool)
  bb_mask = np.zeros_like(hand_mask)

  height, width = hand_mask.shape

  x1, y1, x2, y2 = int(max(bb[0], 0)), int(max(bb[1], 0)), int(min(bb[2], width)), int(min(bb[3], height))

  bb_mask[y1:y2, x1:x2] = 1

  iou = np.sum(hand_mask * bb_mask) / np.sum(hand_mask + bb_mask)
  return iou


def calc_bbox_mask_iou(mask, bb):
  """Computes IoU between a mask and a bounding box
  """
  mask = mask.astype(np.bool)
  bb_mask = np.zeros_like(mask)

  height, width = mask.shape

  x1, y1, x2, y2 = int(max(bb[0], 0)), int(max(bb[1], 0)), int(min(bb[2], width)), int(min(bb[3], height))

  bb_mask[y1:y2, x1:x2] = 1

  iou = np.sum(mask * bb_mask) / np.sum(mask + bb_mask)
  return iou
