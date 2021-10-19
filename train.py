#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import datetime
import time
import json
from collections import Counter
from itertools import groupby

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage.morphology import binary_dilation
import Levenshtein

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import EPICDataset, EPICDatasetPL, EPICDatasetDummy
from utils.argument import get_args, get_model
from utils.vis_utils import vis_predictions
from utils.eval_utils import calc_boundary, calc_boundary_accuracy, calc_peripheral_accuracy

from mllogger import MLLogger
logger = MLLogger(init=False)


def eval_net(args, device, save_dir, model, valid_dataset):

  predictions = []

  if args.vis:
    vis_dir = osp.join(save_dir, args.vis_dir_name)
    os.makedirs(vis_dir, exist_ok=True)

  with torch.no_grad():
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.nb_workers if args.eval else args.nb_eval_workers)

    loss, total_acc, total_cnt = 0.0, 0.0, 0.0
    criterion = nn.BCELoss()
    y_trues, y_preds = [], []
    accs, boundary_accs, peripheral_accs, edit_scores = [], [], [], []
    correct_cnt = 0

    # XXX Batch size is 1
    for valid_cnt, batch in enumerate(valid_loader):

      batch = dict([(k, v.to(device)) if type(v) != list else (k, v) for k, v in batch.items()])
      labels = batch["labels"][0]

      if "CrUnionLSTM" in args.model:
        out = model.noisy_net(batch)
      else:
        out = model(batch)

      cnt = torch.sum(labels != -1).item()
      loss += criterion(out[labels > -1], labels[labels > -1]).item() * cnt

      y_pred = (out > 0.5).long()
      acc = torch.sum(y_pred[labels != -1] == labels[labels != -1]).item() / cnt
      total_acc += acc * cnt
      total_cnt += cnt
      y_trues.extend(labels[labels != -1].tolist())
      y_preds.extend(y_pred[labels != -1].tolist())
      accs.append(acc)

      # Boundary precision/recall
      F = calc_boundary_accuracy(y_pred.cpu().numpy(), labels.cpu().numpy(), bound_th=args.bound_th)
      boundary_accs.append(F)
      pacc = calc_peripheral_accuracy(y_pred.cpu().numpy(), labels.cpu().numpy(), bound_th=args.bound_th)
      if pacc != -1.0:
        peripheral_accs.append(pacc)

      pred_seq = "".join([str(x[0]) for x in groupby(y_pred.tolist())])
      gt_seq = "".join([str(x[0]) for x in groupby(labels.cpu().numpy().astype(int).tolist())])
      edit_score = 1 - Levenshtein.distance(pred_seq, gt_seq) / max(len(pred_seq), len(gt_seq))
      edit_scores.append(edit_score)

      if acc > 0.9 and F == 1.0:
        correct_cnt += 1

      pred_dict = {
        "track_id": batch["track_id"][0],
        "y_logit": out.tolist(),
        "y_pred": y_pred.tolist(),
        "y_true": labels.tolist(),
        "acc": acc,
        "bacc": F,
        "pacc": pacc,
        "edit_score": edit_score
      }
      predictions.append(pred_dict)

      if args.debug:
        print("{}\t{}\t{}\t{}\t{}".format(batch["track_id"][0][1:], np.round(acc, 2), np.round(F, 2), np.round(pacc, 2), np.round(edit_score, 2)))

      if args.vis:
        offset = valid_cnt * args.batch_size
        vis_predictions(vis_dir, valid_dataset, offset, pred_dict, args.fps, args.ratio)

  logger.info("Loss: {}".format(loss / total_cnt))
  logger.info("Balanced accuracy: {}".format(balanced_accuracy_score(y_trues, y_preds)))
  logger.info("Boundary accuracy: {}".format(np.mean(boundary_accs)))
  logger.info("Peripheral accuracy: {}".format(np.mean(peripheral_accs)))
  logger.info("Track accuracy: {}".format(correct_cnt / len(valid_dataset)))
  logger.info("Edit score: {}".format(np.mean(edit_scores)))

  return predictions


def main():

  args = get_args()
  total_start_time = time.time()

  if args.eval and args.model == "":
    args.model = osp.basename(osp.dirname(args.resume)).split("_")[0]
  if args.eval and args.dir_name == "tmp":
    args.dir_name = "tmp"

  dir_name = "{}_{}_{}".format(args.model, args.dir_name, datetime.datetime.now().strftime('%y%m%d'))
  if args.eval and args.out_dir == "outputs":
    args.out_dir = "predictions"

  logger.initialize(args.out_dir, dir_name)
  logger.info(vars(args))
  save_dir = logger.get_savedir()
  logger.info("Written to {}".format(save_dir))

  device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device("cpu")

  base_model = get_model(args)

  logger.info("Model: {}".format(base_model.__class__.__name__))
  logger.info("Output dir: {}".format(save_dir))
  if not args.eval and not args.save_model:
    logger.info("Caution: NOT SAVING MODEL! Attach --save_model to export the trained model.")

  if args.resume != "":
    base_model.load_state_dict(torch.load(args.resume))

  base_model.to(device)
  model = base_model  # XXX Batch size more than one is not supported

  with open(args.train_vids) as f:
    vid_list = [x.strip("\n") for x in f.readlines()]
  logger.info("Train vids: {}".format(args.train_vids))

  with open(args.clean_vids) as f:
    clean_vid_list = [x.strip("\n") for x in f.readlines()]
    logger.info("Clean vids: {}".format(args.clean_vids))

  valid_vids = args.test_vids if args.eval else args.valid_vids
  with open(valid_vids) as f:
    valid_vid_list = [x.strip("\n") for x in f.readlines()]
  if args.eval:
    logger.info("Test vids: {}".format(valid_vids))
  else:
    logger.info("Validation vids: {}".format(valid_vids))

  if args.debug:
    vid_list = ["P14_08"]
    valid_vid_list = ["P14_08"]

  if args.model in ["Fixed", "IoU"]:
    valid_dataset = EPICDatasetDummy(args.image_dir, args.flow_dir, args.data_dir, valid_vid_list, False)
  else:
    valid_dataset = EPICDataset(args.image_dir, args.flow_dir, args.data_dir, valid_vid_list, False, modality=args.modality)

  if not args.eval:
    if args.semisupervised:
      clean_dataset = EPICDataset(args.image_dir, args.flow_dir, args.data_dir, clean_vid_list, True, modality=args.modality)

    if args.supervised:
      train_dataset = EPICDataset(args.image_dir, args.flow_dir, args.data_dir, vid_list, True, modality=args.modality)
    else:
      train_dataset = EPICDatasetPL(args.image_dir, args.flow_dir, args.data_dir, vid_list, True, only_boundary=args.only_boundary, modality=args.modality, version=args.anno_version, no_filter=args.no_filter, min_ratio=args.min_label_ratio)

    criterion = nn.BCELoss(reduction='none')
    if args.semisupervised:
      clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_eval_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers, drop_last=True, prefetch_factor=2)

    if args.optimizer == "adam":
      if "CrUnionLSTM" in args.model:
        optimizer = optim.Adam(model.noisy_net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        optimizer_clean = optim.Adam(model.clean_net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
      else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    elif args.optimizer == "sgd":
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=1e-4)
    else:
      raise NotImplementedError()

    scheduler = MultiStepLR(optimizer, args.lr_step_list, 0.1)
    if "CrUnionLSTM" in args.model:
      scheduler_clean = MultiStepLR(optimizer_clean, args.lr_step_list, 0.1)

    epoch_cnt, iter_cnt = 1, 0
    loss_elapsed = []
    current_delta = args.init_delta
    corrected_list = []

    refined_label_dict = {}
    for didx in range(len(train_dataset)):
      refined_label_dict[train_dataset.data[didx]["track_id"]] = train_dataset.data[didx]["labels"]

    refined_label_path = osp.join(save_dir, "labels_init.json")
    logger.info(f"Label saved: {refined_label_path}")
    with open(refined_label_path, "w") as f:
      json.dump(refined_label_dict, f)

    start_time = time.time()
    while iter_cnt != args.nb_iters and optimizer.param_groups[0]['lr'] > args.min_lr:
      print("")
      if args.plc:
        logger.info("Epoch {} ( delta = {} )".format(epoch_cnt, current_delta))
      else:
        logger.info("Epoch {}".format(epoch_cnt))
      all_labels = np.concatenate([x["labels"] for x in train_dataset.data.values()])

      if (args.plc and iter_cnt > args.iter_warmup) or epoch_cnt == 1:
        label_ratio = np.sum(all_labels != -1) / len(all_labels)
        logger.info("Label ratio: {}".format(label_ratio))
        logger.info(Counter(all_labels.tolist()))

      if args.update_clean and (args.plc and iter_cnt > args.iter_warmup):
        clean_all_labels = np.concatenate([x["clean_labels"] for x in train_dataset.data.values()])
        clean_label_ratio = np.sum(clean_all_labels != -1) / len(clean_all_labels)
        logger.info("Clean label ratio: {}".format(clean_label_ratio))
        logger.info(Counter(clean_all_labels.tolist()))
        clean_label_weight = np.sum(clean_all_labels == 1) / np.sum(clean_all_labels == 0)

      label_weight = np.sum(all_labels == 1) / np.sum(all_labels == 0)

      nb_flipped, total_frame = 0, 0
      nb_correct, nb_wrong, nb_total = 0, 0, 0
      nb_correct_pe, nb_wrong_pe, nb_total_pe = 0, 0, 0

      for cnt, batch in enumerate(train_loader):
        if iter_cnt == args.nb_iters:
          break

        batch = dict([(k, v.to(device)) if type(v) != list else (k, v) for k, v in batch.items()])
        labels = batch["labels"][0]
        clean_labels = batch["clean_labels"][0] if "clean_labels" in batch else None

        if args.plc and "gt_labels" in batch:
          gt_labels = batch["gt_labels"][0]
          nb_correct += torch.sum(labels == gt_labels).item()
          nb_wrong += torch.sum((labels != gt_labels) * (labels != -1)).item()
          nb_total += len(labels)

          gt_boundary = calc_boundary(gt_labels.cpu().numpy())
          gt_dil = binary_dilation(gt_boundary, np.ones(args.bound_th*2+1))
          nb_correct_pe += np.sum((labels.cpu().numpy() == gt_labels.cpu().numpy()) * (gt_dil == 1)).item()
          nb_wrong_pe += np.sum((labels.cpu().numpy() != gt_labels.cpu().numpy()) * (labels.cpu().numpy() != -1) * (gt_dil == 1)).item()
          nb_total_pe += np.sum(gt_dil)

        if iter_cnt < args.iter_warmup and ((0 not in labels.tolist()) or (1 not in labels.tolist())):
          print("\rBad batch: skip", end="")
        elif torch.sum(labels > -1) > 0:

          if "CrUnionLSTM" in args.model:
            if args.update_clean and torch.sum(clean_labels >= 0) >= len(clean_labels) * args.min_clean_label_ratio:
              model.zero_grad()
              y_corrected = model.clean_net(batch)
              loss = criterion(y_corrected[clean_labels > -1], clean_labels[clean_labels > -1])

              weight = torch.ones(len(loss), dtype=torch.float32, device=loss.device)
              weight[clean_labels[clean_labels > -1] == 0.0] = clean_label_weight

              loss = torch.mean(weight * loss)
              loss.backward()
              nn.utils.clip_grad_norm_(model.clean_net.parameters(), max_norm=1.0, norm_type=2)
              optimizer_clean.step()
              scheduler_clean.step()
              print(f" {batch['track_id'][0]} {np.round(loss.item(), 4)}", end="")
            else:
              with torch.no_grad():
                y_corrected = model.clean_net(batch)

          # Inference
          model.zero_grad()
          if "CrUnionLSTM" in args.model:
            out = model.noisy_net(batch)
          else:
            out = model(batch)

          # Label flipping
          if args.plc and iter_cnt >= args.iter_warmup:
            if "CrUnionLSTM" in args.model:  # gPLC
              flip_mask = (out.detach() > 1 - current_delta) * (y_corrected > 0.5) + \
                  (out.detach() < current_delta) * (y_corrected < 0.5)
            else:  # Naive PLC
              flip_mask = (out.detach() > 1 - current_delta) + (out.detach() < current_delta)

            new_labels = labels.clone()
            new_clean_labels = batch["clean_labels"][0].clone()

            labeled_mask = (labels != -1)
            if args.asymp_labeled_flip:
              new_labels[flip_mask * labeled_mask] = (out > 0.5).float()[flip_mask * labeled_mask]
              new_clean_labels[flip_mask * labeled_mask] = (out > 0.5).float()[flip_mask * labeled_mask]
              corrected_list.append(torch.sum((labels != new_labels)).item() / torch.sum(labels != -1).item())
            else:
              new_labels[flip_mask] = (out > 0.5).float()[flip_mask]
              new_clean_labels[flip_mask] = (out > 0.5).float()[flip_mask]
              corrected_list.append(torch.sum((labels != new_labels)).item() / len(labels))

            bidx, st, en = batch["idx"].item(), batch["st"].item(), batch["en"].item()
            train_dataset.data[bidx]["labels"][st:en] = new_labels.cpu().numpy().astype(int).tolist()
            train_dataset.data[bidx]["clean_labels"][st:en] = new_clean_labels.cpu().numpy().astype(int).tolist()

            if len(corrected_list) > len(train_dataset):
              corrected_list.pop(0)

            if (iter_cnt + 1) % len(train_dataset) == 0:
              print("Corrected ratio: {}".format(np.mean(corrected_list)))
            if len(corrected_list) >= len(train_dataset) and np.mean(corrected_list) <= args.delta_th and current_delta < args.max_delta:
              new_delta = min(current_delta + args.delta_increment, args.max_delta)
              logger.info("New delta: {} -> {}".format(current_delta, new_delta))
              current_delta = new_delta
              corrected_list = []

            nb_flipped += torch.sum((labels != new_labels) * (labels != -1)).item()
            total_frame += torch.sum((labels != -1)).item()

            labels = new_labels

          valid_mask = (labels > -1)
          if torch.sum(valid_mask) > 0:
            loss = criterion(out[valid_mask], labels[valid_mask])
            weight = torch.ones(len(loss), dtype=torch.float32, device=loss.device)
            weight[labels[valid_mask] == 0.0] = label_weight
          else:
            loss = criterion(out, labels)

          if torch.sum(valid_mask) > 0:
            loss = torch.mean(weight * loss)
            total_loss = loss

            if args.confidence_penalty > 0.0:
              confidence_penalty = torch.mean(out * torch.log(torch.clamp(out, min=1e-10)) + (1 - out) * torch.log(torch.clamp(1 - out, min=1e-10)))
              total_loss = total_loss + confidence_penalty * args.confidence_penalty

            total_loss.backward()
            if "CrUnionLSTM" in args.model:
              nn.utils.clip_grad_norm_(model.noisy_net.parameters(), max_norm=1.0, norm_type=2)
            else:
              nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            scheduler.step()
            loss_elapsed.append(loss.item())
            print(f"\r{batch['track_id'][0]} {np.round(loss.item(), 4)}", end="")
          else:
            torch.mean(loss).backward()
        else:
          print("\rBad batch: skip", end="")

        if args.semisupervised and (iter_cnt + 1) % args.iter_supervision == 0:
          print("")
          logger.info("Tune by clean label...")
          # Train clean set
          total_noisy_loss, total_correct_loss = 0.0, 0.0
          for clean_cnt, clean_batch in enumerate(clean_loader):
            clean_batch = dict([(k, v.to(device)) if type(v) != list else (k, v) for k, v in clean_batch.items()])
            labels = clean_batch["labels"][0]
            if "CrUnionLSTM" in args.model:
              model.zero_grad()
              y_corrected = model.clean_net(clean_batch)
              correct_loss = torch.mean(criterion(y_corrected[labels > -1], labels[labels > -1]))
              loss = correct_loss
              total_correct_loss += correct_loss.item()
              loss.backward()
              nn.utils.clip_grad_norm_(model.clean_net.parameters(), max_norm=1.0, norm_type=2)
              optimizer_clean.step()

              model.zero_grad()
              if args.finetune_noisy_net:
                y_tilde = model.noisy_net(clean_batch)
                noisy_loss = torch.mean(criterion(y_tilde[labels > -1], labels[labels > -1]))
                noisy_loss.backward()
                nn.utils.clip_grad_norm_(model.noisy_net.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()
              else:
                with torch.no_grad():
                  y_tilde = model.noisy_net(clean_batch)
                  noisy_loss = torch.mean(criterion(y_tilde[labels > -1], labels[labels > -1]))

              total_noisy_loss += noisy_loss.item()
            else:  # Normal network
              model.zero_grad()
              out = model(clean_batch)
              loss = criterion(out[labels > -1], labels[labels > -1])
              loss = torch.mean(loss)
              total_noisy_loss += loss.item()
              loss.backward()
              nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
              optimizer.step()
            print(f"\r{clean_batch['track_id'][0]} {np.round(loss.item(), 4)}", end="")
          print("")
          logger.info("Noisy loss: {}".format(total_noisy_loss / len(clean_dataset)))
          logger.info("Correct loss: {}".format(total_correct_loss / len(clean_dataset)))

        if (iter_cnt + 1) % args.iter_display == 0:
          logger.info("Iter {}: {}, {} iter / s".format(iter_cnt + 1, np.mean(loss_elapsed), args.iter_display / (time.time() - start_time)))
          loss_elapsed = []
          start_time = time.time()

        if args.save_model and (iter_cnt + 1) % args.iter_snapshot == 0:
          model_path = osp.join(save_dir, "model_{:06d}.pth".format(iter_cnt + 1))
          logger.info("Checkpoint: {}".format(model_path))
          torch.save(base_model.state_dict(), model_path)

        if args.iter_evaluation != -1 and (iter_cnt + 1) % args.iter_evaluation == 0:
          logger.info("Validation...")
          logger.info("Iter {} accuracy:".format(iter_cnt + 1))
          model.eval()
          eval_net(args, device, save_dir, model, valid_dataset)
          model.train()
          start_time = time.time()

        iter_cnt += 1

      if args.plc and total_frame > 0 and iter_cnt > args.iter_warmup:
        logger.info("Flipped: {} / {} ( {} % )".format(nb_flipped, total_frame, nb_flipped / total_frame * 100))
      if args.plc and nb_total > 0 and (iter_cnt > args.iter_warmup or epoch_cnt == 1):
        logger.info("{} {} {}".format(nb_correct, nb_wrong, nb_total))
        logger.info("{} {} {}".format(nb_correct / nb_total, nb_wrong / nb_total, (nb_correct + nb_wrong) / nb_total))
        logger.info("{} {} {}".format(nb_correct_pe, nb_wrong_pe, nb_total_pe))
        logger.info("{} {} {}".format(nb_correct_pe / nb_total_pe, nb_wrong_pe / nb_total_pe, (nb_correct_pe + nb_wrong_pe) / nb_total_pe))

      if args.plc:  # Export updated pseudo-labels
        refined_label_dict = {}
        for didx in range(len(train_dataset)):
          refined_label_dict[train_dataset.data[didx]["track_id"]] = train_dataset.data[didx]["labels"]

        refined_label_path = osp.join(save_dir, f"labels_{epoch_cnt:02d}.json")
        logger.info(f"Label saved: {refined_label_path}")
        with open(refined_label_path, "w") as f:
          json.dump(refined_label_dict, f)

        refined_label_dict = {}
        for didx in range(len(train_dataset)):
          refined_label_dict[train_dataset.data[didx]["track_id"]] = train_dataset.data[didx]["clean_labels"]

        refined_label_path = osp.join(save_dir, f"clean_labels_{epoch_cnt:02d}.json")
        logger.info(f"Clean label saved: {refined_label_path}")
        with open(refined_label_path, "w") as f:
          json.dump(refined_label_dict, f)

      epoch_cnt += 1

  else:  # Evaluation
    predictions = eval_net(args, device, save_dir, model, valid_dataset)
    pred_path = osp.join(save_dir, "predictions_{}.json".format(osp.splitext(osp.basename(valid_vids))[0]))
    with open(pred_path, "w") as f:
      json.dump(predictions, f)
  logger.info("Done. Elapsed time: {} (s)".format(time.time()-total_start_time))


if __name__ == "__main__":
  main()
