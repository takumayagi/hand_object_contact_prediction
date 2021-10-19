#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os.path as osp
import sys
import json
from collections import Counter

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from pycocotools import mask as mask_util

from utils.track_utils import calc_iou, calc_bbox_mask_iou
from utils.flow_utils import read_png_flow, calc_homography


overlap_iou = 0.5


class EPICDatasetBase(Dataset):
  def __init__(self, root_image_dir, root_flow_dir, root_data_dir, vid_list, train, modality, debug=False):
    self.root_image_dir = root_image_dir
    self.root_flow_dir = root_flow_dir
    self.root_data_dir = root_data_dir
    self.train = train
    self.rgb_width, self.rgb_height = 112, 112
    self.fl_width, self.fl_height = 112, 112
    self.mask_width, self.mask_height = 32, 32
    self.max_length = 105
    self.modality = modality
    self.debug = debug
    self.data = {}

    self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

  def get_interval(self, info):
    nb_frames = len(info["labels"])
    if self.train and nb_frames > self.max_length:
      for trial in range(100):
        st = torch.randint(0, nb_frames - self.max_length, (1,)).item()
        en = st + self.max_length
        if np.max(np.array(info["labels"][st:en])) >= 0:
          break
    else:
      st, en = 0, nb_frames

    return st, en

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    info = self.data[i]
    im_width, im_height = info["im_width"], info["im_height"]

    st, en = self.get_interval(info)

    # labels
    labels = np.array(info["labels"][st:en]).astype(np.float32)

    image_dir = osp.join(self.root_image_dir, info["pid"], info["vid"])
    flow_dir = osp.join(self.root_flow_dir, info["pid"], info["vid"])

    # Hand appearance
    union_imgs, union_flows = [], []
    hand_masks, other_hand_masks, obj_bbox_masks, other_bbox_masks = [], [], [], []

    for idx, ((frame_num, hx1, hy1, hx2, hy2, _, _), (_, ox1, oy1, ox2, oy2, _, _), rle, other_rle, other_boxes) in \
        enumerate(zip(info["hand_track"][st:en], info["obj_track"][st:en], info["hand_rles"][st:en], info["other_hand_rles"][st:en], info["other_obj_boxes"][st:en])):

      ox1, oy1, ox2, oy2 = max(0, int(ox1)), max(0, int(oy1)), min(im_width, int(ox2)), min(im_height, int(oy2))
      hx1, hy1, hx2, hy2 = max(0, int(hx1)), max(0, int(hy1)), min(im_width, int(hx2)), min(im_height, int(hy2))
      ux1, uy1, ux2, uy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)

      # Hand mask map
      hand_mask = cv2.resize(mask_util.decode(rle), (im_width, im_height))

      # Other hand mask
      other_hand_mask = cv2.resize(mask_util.decode(other_rle), (im_width, im_height))
      other_hand_mask[hand_mask == 1] = 0

      # Object bbox
      obj_bbox_mask = np.zeros_like(hand_mask)
      obj_bbox_mask[oy1:oy2, ox1:ox2] = 1
      obj_bbox_mask[(hand_mask == 1) + (other_hand_mask == 1)] = 0

      # Other object bbox
      other_bbox_mask = np.zeros_like(hand_mask)
      for bx1, by1, bx2, by2 in other_boxes:
        bx1, by1, bx2, by2 = max(0, int(bx1)), max(0, int(by1)), min(im_width, int(bx2)), min(im_height, int(by2))
        if calc_iou([bx1, by1, bx2, by2], [ox1, oy1, ox2, oy2]) < overlap_iou:
          other_bbox_mask[by1:by2, bx1:bx2] = 1

      other_bbox_mask[(hand_mask == 1) + (other_hand_mask == 1)] = 0

      hand_masks.append(cv2.resize(hand_mask[uy1:uy2, ux1:ux2], (self.mask_width, self.mask_height)).astype(np.float32))
      other_hand_masks.append(cv2.resize(other_hand_mask[uy1:uy2, ux1:ux2], (self.mask_width, self.mask_height)).astype(np.float32))
      obj_bbox_masks.append(cv2.resize(obj_bbox_mask[uy1:uy2, ux1:ux2], (self.mask_width, self.mask_height)).astype(np.float32))
      other_bbox_masks.append(cv2.resize(other_bbox_mask[uy1:uy2, ux1:ux2], (self.mask_width, self.mask_height)).astype(np.float32))
      all_boxes = other_boxes + [[hx1, hy1, hx2, hy2], [ox1, oy1, ox2, oy2]]

      if "rgb" in self.modality:
        impath = osp.join(image_dir, f"frame_{int(frame_num):010d}.jpg")
        img = cv2.resize(cv2.imread(impath), (im_width, im_height))
        union_imgs.append(cv2.resize(img[uy1:uy2, ux1:ux2], (self.rgb_width, self.rgb_height)).astype(np.float32))

      if "flow" in self.modality:
        flow_path = osp.join(flow_dir, f"fw_{int(frame_num):010d}.flo")
        # flow_path = osp.join(flow_dir, f"fw_{int(frame_num):010d}.flo")
        flow = read_png_flow(flow_path)

        fl_height, fl_width = flow.shape[:2]
        w_ratio, h_ratio = fl_width / im_width, fl_height / im_height

        hx1, hy1, hx2, hy2 = max(0, int(hx1 * w_ratio)), max(0, int(hy1 * h_ratio)), \
            min(fl_width, int(hx2 * w_ratio)), min(fl_height, int(hy2 * h_ratio))
        ox1, oy1, ox2, oy2 = max(0, int(ox1 * w_ratio)), max(0, int(oy1 * h_ratio)), \
            min(fl_width, int(ox2 * w_ratio)), min(fl_height, int(oy2 * h_ratio))
        ux1, uy1, ux2, uy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)

        # Calculate homography matrix and cancel background motion from the forward flows
        H = calc_homography(flow, all_boxes, (im_width, im_height))
        grid = np.stack(np.meshgrid(np.arange(ux1, ux2, 1), np.arange(uy1, uy2, 1)), axis=2)
        union_flow = cv2.perspectiveTransform((grid + flow[uy1:uy2, ux1:ux2]).reshape((1, -1, 2)), H).reshape((uy2 - uy1, ux2 - ux1, 2)) - grid
        mag = np.sqrt(np.sum(union_flow ** 2, axis=2, keepdims=True))
        union_flow = np.concatenate((union_flow, mag), axis=2)
        union_flow = cv2.resize(union_flow, (self.fl_width, self.fl_height)).astype(np.float32)
        union_flows.append(union_flow)

    out_dict = {
      "idx": i,
      "st": st,
      "en": en,
      "track_id": info["track_id"],
      "hand_masks": np.array(hand_masks),
      "other_hand_masks": np.array(other_hand_masks),
      "obj_bbox_masks": np.array(obj_bbox_masks),
      "other_bbox_masks": np.array(other_bbox_masks),
      "labels": labels,
    }

    if "gt_labels" in info and info["gt_labels"] is not None:
      out_dict.update({
        "gt_labels": np.array(info["gt_labels"][st:en]).astype(np.float32)
      })

    if "clean_labels" in info:
      out_dict.update({
        "clean_labels": np.array(info["clean_labels"][st:en]).astype(np.float32)
      })

    if "rgb" in self.modality:
      union_imgs = (np.stack(union_imgs) / 255. - self.mean) / self.std
      out_dict.update({
        "union_imgs": union_imgs.transpose((0, 3, 1, 2)),
      })

    if "flow" in self.modality:
      out_dict.update({
        "union_flows":np.stack(union_flows).transpose((0, 3, 1, 2))
      })

    return out_dict


class EPICDatasetPL(EPICDatasetBase):
  def __init__(self, root_image_dir, root_flow_dir, root_data_dir, vid_list, train, only_boundary=False, modality=["rgb", "flow"], version="v5", no_filter=False, min_ratio=0.0, debug=False):
    super().__init__(root_image_dir, root_flow_dir, root_data_dir, vid_list, train, modality, debug)

    self.filter = not no_filter

    idx = 0
    counter = 0

    # Open annotations
    for vid in vid_list:
      pid = vid.split("_")[0]
      anno_path = osp.join(root_data_dir, pid, vid, f"anno_{version}_processed.json")

      if not osp.exists(anno_path):
        print(f"Annotation not found: skip {vid} {anno_path}")
        continue

      print(f"{vid} ", end="")
      sys.stdout.flush()

      with open(anno_path) as f:
        raw_annos = json.load(f)

      for anno_dict in raw_annos:
        anno = anno_dict["annos"]

        pseudo_labels = np.array(anno["pseudo_labels"])

        # XXX Skip non-label samples
        if self.filter and np.max(pseudo_labels) == -1:
          continue

        # Skip single-label examples
        if self.filter and only_boundary and (0 not in pseudo_labels or 1 not in pseudo_labels):
          continue

        label_ratio = np.sum(pseudo_labels >= 0) / len(pseudo_labels)
        if label_ratio < min_ratio:
          continue

        # XXX Filter small bboxes
        obj_track = np.array(anno["obj_track"])
        obj_diags = np.sqrt((obj_track[:, 3] - obj_track[:, 1]) ** 2 + (obj_track[:, 4] - obj_track[:, 2]) ** 2)
        if self.filter and np.mean(obj_diags) < 80:
          continue

        if self.filter and only_boundary:
          last_label, dist_cnt = -1, 0
          st, en = -1, -1
          for lidx, label in enumerate(pseudo_labels):
            if st == -1 and label != -1:
              last_label, st = label, lidx
            if last_label != -1 and label == last_label and dist_cnt <= 5:
              en, dist_cnt = lidx, 0
            if label != -1 and label != last_label:
              last_label, en, dist_cnt = label, lidx, 0
            if last_label != -1:
              dist_cnt += 1
          if st == -1 or en == -1 or en - st < 15:
            continue
        else:
          st, en = 0, len(pseudo_labels)

        counter += 1

        self.data[idx] = {
          "pid": pid,
          "vid": vid,
          "track_id": "{}_{}_{}".format(vid, anno_dict["hk"], anno_dict["ok"]),
          "im_width": anno_dict["im_width"],
          "im_height": anno_dict["im_height"],
          "obj_track": anno["obj_track"][st:en],
          "hand_track": anno["hand_track"][st:en],
          "hand_rles": anno["hand_rles"][st:en],
          "other_hand_rles": anno["other_hand_rles"][st:en],
          "other_obj_boxes": anno["other_obj_boxes"][st:en],
          "labels": anno["pseudo_labels"][st:en],
          "gt_labels": anno["gt_annos"][st:en] if "gt_annos" in anno else None,
          "clean_labels": [-1 for x in range(en - st)],
        }
        idx += 1

    print("")
    # Print length
    print(len(self.data))

    all_labels = np.concatenate([x["labels"] for x in self.data.values()])
    print(Counter(all_labels.tolist()))


class EPICDataset(EPICDatasetBase):
  def __init__(self, root_image_dir, root_flow_dir, root_data_dir, vid_list, train, modality=["rgb", "flow"], ignore_null=False, debug=False, filter_invalid=True):
    super().__init__(root_image_dir, root_flow_dir, root_data_dir, vid_list, train, modality, debug)

    idx = 0
    total_cnt = 0
    for vid in vid_list:
      nb_cnt = 0
      pid = vid.split("_")[0]

      print(f"{vid} ", end="")
      sys.stdout.flush()

      anno_path = osp.join(root_data_dir, pid, vid, f"dense_anno_{vid}_v3.json")
      with open(anno_path) as f:
        all_anno_dict = json.load(f)

      for anno_dict in all_anno_dict:

        anno = anno_dict["annos"]

        # XXX Skip non-label samples
        gt_labels = np.array(anno["annos"])
        if not ignore_null and np.max(gt_labels) == -1:
          continue

        valid_mask = gt_labels >= 0
        flag = not ignore_null and np.min(gt_labels) == -1 and filter_invalid

        self.data[idx] = {
          "pid": pid,
          "vid": vid,
          "hk": anno_dict["hk"],
          "ok": anno_dict["ok"],
          "track_id": "{}_{}_{}".format(vid, anno_dict["hk"], anno_dict["ok"]),
          "im_width": anno_dict["im_width"],
          "im_height": anno_dict["im_height"],
          "obj_track": [x for x, m in zip(anno["obj_track"], valid_mask) if m] if flag else anno["obj_track"],
          "hand_track": [x for x, m in zip(anno["hand_track"], valid_mask) if m] if flag else anno["hand_track"],
          "hand_rles": [x for x, m in zip(anno["hand_rles"], valid_mask) if m] if flag else anno["hand_rles"],
          "other_hand_rles": [x for x, m in zip(anno["other_hand_rles"], valid_mask) if m] if flag else anno["other_hand_rles"],
          "other_obj_boxes": [x for x, m in zip(anno["other_obj_boxes"], valid_mask) if m] if flag else anno["other_obj_boxes"],
          "labels": [x for x, m in zip(anno["annos"], valid_mask) if m] if flag else anno["annos"]
        }
        nb_cnt += 1
        idx += 1
      total_cnt += nb_cnt

    # Print length
    print("")
    print(len(self.data))
    print(np.mean([len(x["labels"]) for x in self.data.values()]))
    print(np.std([len(x["labels"]) for x in self.data.values()]))

    all_labels = np.concatenate([x["labels"] for x in self.data.values()])
    print(Counter(all_labels.tolist()))


class EPICDatasetDummy(EPICDataset):

  def __getitem__(self, i):
    info = self.data[i]
    im_width, im_height = info["im_width"], info["im_height"]

    st, en = self.get_interval(info)

    # labels
    labels = np.array(info["labels"][st:en]).astype(np.float32)

    mask_ious = []
    for idx, ((frame_num, hx1, hy1, hx2, hy2, _, _), (_, ox1, oy1, ox2, oy2, _, _), rle) in \
        enumerate(zip(info["hand_track"][st:en], info["obj_track"][st:en], info["hand_rles"][st:en])):

      ox1, oy1, ox2, oy2 = max(0, int(ox1)), max(0, int(oy1)), min(im_width, int(ox2)), min(im_height, int(oy2))
      hx1, hy1, hx2, hy2 = max(0, int(hx1)), max(0, int(hy1)), min(im_width, int(hx2)), min(im_height, int(hy2))

      # Hand mask map
      hand_mask = cv2.resize(mask_util.decode(rle), (im_width, im_height))
      mask_ious.append(calc_bbox_mask_iou(hand_mask, (ox1, oy1, ox2, oy2)))

    return {
      "track_id": info["track_id"],
      "labels": labels,
      "mask_ious": np.array(mask_ious).astype(np.float32)
    }
