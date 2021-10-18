#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np
from scipy.ndimage.morphology import binary_dilation


def calc_boundary(arr):

  arr = arr.astype(np.bool)
  arr[arr > 0] = 1

  nb_frames = len(arr)

  e = np.zeros_like(arr)
  se = np.zeros_like(arr)

  e[:-1] = arr[:-1]
  se[:-1] = arr[1:]

  b = arr^e | arr^se
  b[-1] = 0

  return b


def calc_boundary_accuracy(y_pred, y_true, bound_th=6):

  pr_boundary = calc_boundary(y_pred)
  gt_boundary = calc_boundary(y_true)

  pr_dil = binary_dilation(pr_boundary, np.ones(bound_th*2+1))
  gt_dil = binary_dilation(gt_boundary, np.ones(bound_th*2+1))

  gt_match = gt_boundary * pr_dil
  pr_match = pr_boundary * gt_dil

  n_pr = np.sum(pr_boundary)
  n_gt = np.sum(gt_boundary)

  # print(gt_match, pr_match)

  if n_pr == 0 and n_gt > 0:
    precision, recall = 1, 0
  elif n_pr > 0 and n_gt == 0:
    precision, recall = 0, 1
  elif n_pr == 0 and n_gt == 0:
    precision, recall = 1, 1
  else:
    precision = np.sum(pr_match) / n_pr
    recall = np.sum(gt_match) / n_gt

  if precision + recall == 0:
    F = 0
  else:
    F = 2 * precision * recall / (precision + recall)

  return F


def calc_peripheral_accuracy(y_pred, y_true, bound_th=6):

  gt_boundary = calc_boundary(y_true)
  gt_dil = binary_dilation(gt_boundary, np.ones(bound_th*2+1))

  if np.sum(gt_dil) == 0:
    return -1.0
  else:
    return np.sum(y_pred[gt_dil == 1] == y_true[gt_dil == 1]) / np.sum(gt_dil)


if __name__ == "__main__":
  print(calc_boundary(np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])))
  print(calc_boundary(np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 1])))
  print(calc_boundary(np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 0])))
  print(calc_boundary(np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])))

  print(calc_boundary_accuracy(
    np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
    np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1]),
    bound_th=2
  ))

  print(calc_boundary_accuracy(
    # np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
    # np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]),
    np.array([1, 1, 1, 1, 1]),
    np.array([0, 0, 0, 0, 0]),
    bound_th=1
  ))
  print(calc_peripheral_accuracy(
    np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1]),
    np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]),
    bound_th=1
  ))
