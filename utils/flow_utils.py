#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader


def get_fg_mask(detections, im_height, im_width, pad_ratio=0.05):

  fg_mask = np.zeros((im_height, im_width), dtype=np.uint8)

  # Filter out both hands and objects
  for box in detections:

    hr, wr = box[3] - box[1], box[2] - box[0]

    if hr > im_height * 3 / 4 or wr > im_width * 3 / 4:
      continue

    ax1, ax2 = int(max(0, box[0] - wr * pad_ratio)), int(min(im_width, box[2] + wr * pad_ratio))
    ay1, ay2 = int(max(0, box[1] - hr * pad_ratio)), int(min(im_height, box[3] + hr * pad_ratio))

    fg_mask[ay1:ay2, ax1:ax2] = 1

  return fg_mask


def propagate_flow(target, flow):

  fl_height, fl_width = flow.shape[:2]

  ndim = target.ndim

  if ndim == 2:
    target = target[..., None]

  # Track mask
  # Track pixels within bounding boxes
  xx, yy = np.meshgrid(np.arange(fl_width), np.arange(fl_height))

  # calculate x + dx
  px = xx + flow[..., 0]
  py = yy + flow[..., 1]
  px = px.reshape((-1))
  py = py.reshape((-1))

  # Calc weights
  x1 = np.clip(np.floor(px), 0, fl_width - 1).astype(np.int)
  x2 = np.clip(np.floor(px) + 1, 0, fl_width - 1).astype(np.int)
  y1 = np.clip(np.floor(py), 0, fl_height - 1).astype(np.int)
  y2 = np.clip(np.floor(py) + 1, 0, fl_height - 1).astype(np.int)

  a = np.expand_dims((np.floor(px) + 1 - px) * (np.floor(py) + 1 - py), 1)
  b = np.expand_dims((px - np.floor(px)) * (np.floor(py) + 1 - py), 1)
  c = np.expand_dims((np.floor(px) + 1 - px) * (py - np.floor(py)), 1)
  d = np.expand_dims((px - np.floor(px)) * (py - np.floor(py)), 1)

  result = target[y1, x1] * a + target[y1, x2] * b + target[y2, x1] * c + target[y2, x2] * d

  if ndim == 2:
    return result.reshape((fl_height, fl_width))
  else:
    return result.reshape((fl_height, fl_width, -1))


def apply_homography(pts, H):
  """
  pts (N, 2)
  """
  ndim = pts.ndim
  if ndim == 1:
    pts = pts[None]
  pts = np.concatenate((pts, np.ones((len(pts), 1))), axis=1)
  pts = np.dot(pts, H.T)
  pts /= pts[:, 2:3]

  if ndim == 1:
    return pts[0, :2]
  else:
    return pts[:, :2]


def calc_consistency(fw_flow, bk_flow, ratio=0.01, offset=0.5):
  """
  fw_flow: Ft->t+1
  bg_flow: Ft+1->t
  """
  fl_height, fl_width = fw_flow.shape[:2]

  warped_bk_flow = propagate_flow(bk_flow, fw_flow)

  loop_flow = fw_flow + warped_bk_flow
  loop_mag = np.sqrt(loop_flow[..., 0] ** 2 + loop_flow[..., 1] ** 2)

  fw_mag = np.sqrt(fw_flow[..., 0] ** 2 + fw_flow[..., 1] ** 2)
  bk_mag = np.sqrt(warped_bk_flow[..., 0] ** 2 + warped_bk_flow[..., 1] ** 2)

  consistency_map = loop_mag <= ratio * (fw_mag + bk_mag) + offset

  return consistency_map


def correct_flow(flow, im_width, im_height, detections, valid_ratio=0.01):
  """
  Image shuld be original detection size
  """
  height, width = flow.shape[:2]

  fg_mask = cv2.resize(get_fg_mask(detections, im_height, im_width), (width, height))

  bg_mask = (1 - fg_mask).astype(np.bool)

  if np.sum(bg_mask) < height * width * valid_ratio:
    return flow  # No correction due to less background mask

  bg_flows = flow[bg_mask].reshape((-1, 2))

  # Estimate homography given optical flows
  stride = 4
  xx, yy = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
  sgrid = np.stack((xx, yy), axis=2)
  match0 = sgrid[bg_mask[::stride, ::stride]].reshape((-1, 2))
  bg_flows = flow[::stride, ::stride][bg_mask[::stride, ::stride]].reshape((-1, 2))
  match1 = match0 + bg_flows

  if len(match1) < 100:
    H = np.eye(3)
  else:
    # Backward homography
    cv2.setNumThreads(4)
    H, masks = cv2.findHomography(match1, match0, cv2.RANSAC, 10)

  xx, yy = np.meshgrid(np.arange(width), np.arange(height))
  grid = np.stack((xx, yy), axis=2)
  new_flow = cv2.perspectiveTransform((grid + flow).reshape((1, -1, 2)), H).reshape((height, width, 2)) - grid

  # XXX Fix errors in peripheral region. Not sure
  new_flow[0, 0] = 0.0
  new_flow[0, -1] = 0.0
  new_flow[-1, 0] = 0.0
  new_flow[-1, -1] = 0.0

  return new_flow


def calc_homography(flow, detections, im_size, valid_ratio=0.01):
  """
  Image shuld be original detection size
  """
  im_width, im_height = im_size

  height, width = flow.shape[:2]

  fg_mask = cv2.resize(get_fg_mask(detections, im_height, im_width), (width, height))
  bg_mask = (1 - fg_mask).astype(np.bool)

  if np.sum(bg_mask) < height * width * valid_ratio:
    return np.eye(3)  # No correction due to less background mask

  bg_flows = flow[bg_mask].reshape((-1, 2))

  # Estimate homography given optical flows
  stride = 4
  xx, yy = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
  sgrid = np.stack((xx, yy), axis=2)
  match0 = sgrid[bg_mask[::stride, ::stride]].reshape((-1, 2))
  bg_flows = flow[::stride, ::stride][bg_mask[::stride, ::stride]].reshape((-1, 2))
  match1 = match0 + bg_flows

  if len(match1) < 100:
    H = np.eye(3)
  else:
    # Backward homography
    cv2.setNumThreads(4)
    H, masks = cv2.findHomography(match1, match0, cv2.RANSAC, 10)

  return H


def prepare_flows(args, im_width, im_height, frame_nums, info_dict):
  # XXX Should be parallelizable
  imgs, fg_mags = [], []
  fw_flows, bk_flows, fg_flows = [], [], []
  for frame_num in frame_nums:
    impath = osp.join(args.image_dir, f"frame_{frame_num:010d}.jpg")
    fw_flow_path = osp.join(args.flow_dir, f"fw_{frame_num:010d}.flo")
    bk_flow_path = osp.join(args.flow_dir, f"bk_{frame_num:010d}.flo")

    if not osp.exists(fw_flow_path):
      print(f"Optical flow not found: {fw_flow_path}")
      sys.exit(1)

    if not osp.exists(bk_flow_path):
      print(f"Optical flow not found: {bk_flow_path}")
      sys.exit(1)

    # Read images
    img = cv2.imread(impath)
    img = cv2.resize(img, (im_width, im_height))

    fw_flow = read_png_flow(fw_flow_path)
    bk_flow = read_png_flow(bk_flow_path)
    fl_height, fl_width = fw_flow.shape[:2]

    refined_fw_flow = correct_flow(img, fw_flow, info_dict["detections"][f"frame_{frame_num:010d}"])
    refined_fw_flow = cv2.resize(refined_fw_flow, (im_width, im_height))
    refined_fw_flow *= np.array([im_width / fl_width, im_height / fl_height])
    mag = np.sqrt(refined_fw_flow[..., 0] ** 2 + refined_fw_flow[..., 1] ** 2)

    imgs.append(img)
    fw_flows.append(fw_flow)
    bk_flows.append(bk_flow)
    fg_flows.append(refined_fw_flow)
    fg_mags.append(mag)

  # Calculate consistency maps
  consistency_maps = [calc_consistency(fw_flows[t], bk_flows[t+1]) for t in range(len(fw_flows)-1)]
  consistency_maps.append(consistency_maps[-1])

  return imgs, fw_flows, bk_flows, fg_flows, fg_mags, consistency_maps



class DualFlowDataset(Dataset):

  def __init__(self, args, frame_nums, info_dict):
    self.image_dir = args.image_dir
    self.flow_dir = args.flow_dir
    self.im_width = info_dict["video_info"]["width"]
    self.im_height = info_dict["video_info"]["height"]
    self.frame_nums = frame_nums
    self.dets = info_dict["detections"]

  def __len__(self):
    return len(self.frame_nums)

  def __getitem__(self, i):
    frame_num = self.frame_nums[i]

    impath = osp.join(self.image_dir, f"frame_{frame_num:010d}.jpg")
    fw_flow_path = osp.join(self.flow_dir, f"fw_{frame_num:010d}.flo")
    bk_flow_path = osp.join(self.flow_dir, f"bk_{frame_num:010d}.flo")

    if not osp.exists(fw_flow_path):
      print(f"Optical flow not found: {fw_flow_path}")

    if not osp.exists(bk_flow_path):
      print(f"Optical flow not found: {bk_flow_path}")

    # Read images
    img = cv2.resize(cv2.imread(impath), (self.im_width, self.im_height))
    fw_flow = read_png_flow(fw_flow_path)
    bk_flow = read_png_flow(bk_flow_path)
    fl_height, fl_width = fw_flow.shape[:2]

    refined_fw_flow = correct_flow(img, fw_flow, self.dets[f"frame_{frame_num:010d}"])
    mag = np.sqrt(refined_fw_flow[..., 0] ** 2 + refined_fw_flow[..., 1] ** 2)

    return img, fw_flow, bk_flow, refined_fw_flow, mag


def write_png_flow(name, flow, threshold=65.535):
  uint16_max = 65535
  flow = np.clip(flow + threshold, 0.0, threshold * 2) * (uint16_max / threshold / 2)
  flow = flow.astype(np.uint16)

  f = open(name, 'wb')
  th = np.array([threshold], dtype=np.float32)
  th.tofile(f)
  bytes1 = cv2.imencode('.png', flow[..., 0])[1]
  bytes2 = cv2.imencode('.png', flow[..., 1])[1]
  len1 = np.array([len(bytes1)], dtype=np.int32)
  len2 = np.array([len(bytes2)], dtype=np.int32)
  len1.tofile(f)
  len2.tofile(f)
  bytes1.tofile(f)
  bytes2.tofile(f)
  f.close()


def read_png_flow(name):
  uint16_max = 65535
  f = open(name, 'rb')
  th = np.empty(1, np.float32)
  len1 = np.empty(1, np.int32)
  len2 = np.empty(1, np.int32)
  f.readinto(th)
  f.readinto(len1)
  f.readinto(len2)

  u = np.empty(int(len1[0]), np.uint8)
  v = np.empty(int(len2[0]), np.uint8)
  f.readinto(u)
  f.readinto(v)
  u = cv2.imdecode(u, -1)
  v = cv2.imdecode(v, -1)

  f.close()

  flow = np.stack((u, v), axis=2)
  return flow.astype(np.float32) / (uint16_max / th[0] / 2) - th[0]
