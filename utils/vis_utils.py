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
import json
import glob
import argparse
import re

import numpy as np
import cv2


def vis_predictions(vis_dir, dataset, data_idx, pred_dict, fps, ratio):

  info = dataset.data[data_idx]

  out_size = (info["im_width"] // ratio, info["im_height"] // ratio)
  track_id = info["track_id"]

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(osp.join(vis_dir, f"pred_{track_id}.mp4"), fourcc, fps, out_size)

  for idx, ((frame_num, hx1, hy1, hx2, hy2, _, _), (_, ox1, oy1, ox2, oy2, _, _)) \
      in enumerate(zip(info["hand_track"], info["obj_track"])):
    impath = osp.join(dataset.root_image_dir, info["pid"], info["vid"], f"frame_{int(frame_num):010d}.jpg")
    img = cv2.resize(cv2.imread(impath), (info["im_width"], info["im_height"]))

    hx1, hy1, hx2, hy2 = max(0, int(hx1)), max(0, int(hy1)), min(info["im_width"], int(hx2)), min(info["im_height"], int(hy2))
    ox1, oy1, ox2, oy2 = max(0, int(ox1)), max(0, int(oy1)), min(info["im_width"], int(ox2)), min(info["im_height"], int(oy2))

    cv2.putText(img, f"{int(frame_num)}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4, cv2.LINE_AA)
    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 1)
    cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (255, 0, 0), 1)

    if hy1 > 30:
      cv2.putText(img, str(int(pred_dict["y_pred"][idx])), (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(img, str(int(pred_dict["y_true"][idx])), (hx1 + 35, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    else:
      cv2.putText(img, str(int(pred_dict["y_pred"][idx])), (hx1, hy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(img, str(int(pred_dict["y_true"][idx])), (hx1 + 35, hy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    writer.write(cv2.resize(img, out_size))
  writer.release()
