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
import glob
import time
import re
import json
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from flownet2.models import FlowNet2  # the path is depended on where you create this module


def filter_impath_list(impath_list, skip):
  if skip == 1:
    return impath_list

  frame_nums = [int(re.sub("\\D", "", osp.basename(x))) for x in impath_list]

  return [impath for impath, frame_num in zip(impath_list, frame_nums) if frame_num % skip == 1]


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


class PairedImageDataset(Dataset):

  def __init__(self, args, image_dir, valid_frame_nums=None, frame_skip=None):
    if valid_frame_nums is None:
      impath_list = list(sorted(glob.glob(osp.join(image_dir, "*.jpg"))))
      impath_list = filter_impath_list(impath_list, args.skip)
      self.impath_list1 = impath_list[:-1]
      self.impath_list2 = impath_list[1:]

      filter_idxs = [idx for idx, (x1, x2) in enumerate(zip(self.impath_list1, self.impath_list2)) if
          not osp.exists(osp.join(args.track_dir, "flows", f"fw_{osp.basename(x1)[6:16]}.flo")) or
          not osp.exists(osp.join(args.track_dir, "flows", f"bk_{osp.basename(x2)[6:16]}.flo"))]
      self.impath_list1 = [x for idx, x in enumerate(self.impath_list1) if idx in filter_idxs]
      self.impath_list2 = [x for idx, x in enumerate(self.impath_list2) if idx in filter_idxs]
    else:
      self.impath_list1 = [osp.join(image_dir, f"frame_{frame_num:010d}.jpg") for frame_num in valid_frame_nums]
      self.impath_list2 = [osp.join(image_dir, f"frame_{frame_num+frame_skip:010d}.jpg") for frame_num in valid_frame_nums]
    self.in_width = args.in_width
    self.in_height = args.in_height

  def __len__(self):
    return len(self.impath_list1)

  def __getitem__(self, i):
    impath1 = self.impath_list1[i]
    impath2 = self.impath_list2[i]
    pim1 = Image.open(impath1)
    pim2 = Image.open(impath2)
    im1 = np.array(pim1.resize((self.in_width, self.in_height), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1)
    im2 = np.array(pim2.resize((self.in_width, self.in_height), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1)

    return {
      "impath1": osp.basename(impath1),
      "impath2": osp.basename(impath2),
      "im1": im1,
      "im2": im2
    }


if __name__ == '__main__':
  """
  Extract optical flows for training
  """

  # obtain the necessary args for construct the flownet framework
  parser = argparse.ArgumentParser()
  parser.add_argument('vid', help="Video ID (e.g. P01_01, P01_101")
  parser.add_argument('--image_dir', default="data/rgb_frames/")
  parser.add_argument('--anno_dir', default="data/annotations/")
  parser.add_argument('--out_dir', default="data/flow_frames/")
  parser.add_argument('--flownet_path', default="pretrained/FlowNet2_checkpoint.pth.tar")
  parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
  parser.add_argument("--rgb_max", type=float, default=255.)
  parser.add_argument("--in_height", type=int, default=384)
  parser.add_argument("--in_width", type=int, default=640)
  parser.add_argument("--out_ratio", type=int, default=2)
  parser.add_argument("--batch_size", type=int, default=24)
  parser.add_argument("--nb_workers", type=int, default=8)
  parser.add_argument("--frame_skip", type=int, default=2)
  parser.add_argument('--gt', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--extract_backward_flow', action='store_true')
  parser.add_argument('--no_filter', action='store_true')
  args = parser.parse_args()

  start_time = time.time()

  pid = args.vid[:3]

  in_height, in_width = args.in_height, args.in_width
  assert in_height % 64 == 0 and in_width % 64 == 0
  out_height, out_width = in_height // args.out_ratio, in_width // args.out_ratio

  valid_frame_nums = []

  if not args.gt:
    pseudo_anno_path = osp.join(args.anno_dir, pid, args.vid, "anno_v5_processed.json")
    with open(pseudo_anno_path) as f:
      anno_dict = json.load(f)
    for anno_track in anno_dict:
      valid_frame_nums.extend(anno_track["annos"]["frame_nums"])
  else:
    gt_anno_path = osp.join(args.anno_dir, pid, args.vid, f"dense_anno_{args.vid}_v3.json")
    with open(gt_anno_path) as f:
      gt_anno_dict = json.load(f)
    for anno_track in gt_anno_dict:
      valid_frame_nums.extend(anno_track["annos"]["frame_nums"])

  valid_frame_nums = np.sort(np.unique(valid_frame_nums))

  image_dir = osp.join(args.image_dir, pid, args.vid)
  last_frame_num = int(re.sub("\\D", "", osp.basename(list(sorted(glob.glob(osp.join(image_dir, "*.jpg"))))[-1])))

  add_nums = []
  for frame_num in valid_frame_nums:
    if frame_num + args.frame_skip <= last_frame_num - args.frame_skip and frame_num + args.frame_skip not in valid_frame_nums:
      add_nums.append(frame_num + args.frame_skip)
    if frame_num != 1 and frame_num - args.frame_skip not in valid_frame_nums:
      add_nums.append(frame_num - args.frame_skip)

  valid_frame_nums = np.sort(np.concatenate((valid_frame_nums, np.unique(add_nums))))
  valid_frame_nums = valid_frame_nums[valid_frame_nums <= last_frame_num - args.frame_skip]

  # Initialize network
  net = FlowNet2(args).cuda()
  dict = torch.load(args.flownet_path)
  net.load_state_dict(dict["state_dict"])

  out_dir = osp.join(args.out_dir, pid, args.vid)

  dataset = PairedImageDataset(args, image_dir, valid_frame_nums, args.frame_skip)
  data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.nb_workers)

  print(f"{args.vid}: Number of frames={len(dataset)}")

  if len(dataset) == 0:
    print("No more frames to be processed")
    sys.exit(1)

  os.makedirs(out_dir, exist_ok=True)

  with torch.no_grad():
    wh = torch.tensor([out_width / in_width, out_height / in_height]).cuda()

    for batch in data_loader:
      st = time.time()

      # Forward flow
      im_fw = torch.stack([batch["im1"], batch["im2"]], dim=2).cuda()
      raw_results = net(im_fw)
      results = F.interpolate(raw_results, size=(out_height, out_width), mode='bilinear', align_corners=True) * wh.view(1, 2, 1, 1)
      results_numpy = results.data.cpu().numpy().transpose(0, 2, 3, 1)

      for idx, flow in enumerate(results_numpy):
        frame_num = re.sub("\\D", "", batch["impath1"][idx])
        write_png_flow(osp.join(out_dir, f"fw_{frame_num}.flo"), flow)

      # Backward flow (only required for pseudo-label generation)
      if args.extract_backward_flow:
        im_bk = torch.stack([batch["im2"], batch["im1"]], dim=2).cuda()
        raw_results = net(im_bk)
        results = F.interpolate(raw_results, size=(out_height, out_width), mode='bilinear', align_corners=True) * wh.view(1, 2, 1, 1)
        results_numpy = results.data.cpu().numpy().transpose(0, 2, 3, 1)

        for idx, flow in enumerate(results_numpy):
          frame_num = re.sub("\\D", "", batch["impath2"][idx])
          write_png_flow(osp.join(out_dir, f"bk_{frame_num}.flo"), flow)

  print("Done. Elapsed time: {} (s)".format(time.time()-start_time))
