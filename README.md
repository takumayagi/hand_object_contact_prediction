# Hand-Object Contact Prediction (BMVC2021)
This repository contains the code and data for the paper "Hand-Object Contact Prediction via Motion-Based Pseudo-Labeling and Guided Progressive Label Correction" by Takuma Yagi, Md. Tasnimul Hasan and Yoichi Sato.

**Under Construction: We will publish the code and data as soon as possible.**

## Requirements
* Python 3.6+
* ffmpeg
* numpy
* opencv-python
* pillow
* scikit-learn
* python-Levenshtein
* pycocotools
* torch (1.8.1, 1.4.0- for flow generation)
* torchvision (0.9.1)
* [mllogger](https://github.com/takumayagi/mllogger)
* [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)https://github.com/NVIDIA/flownet2-pytorch

Caution: You will need ~100GB space for testing and ~2TB space for training.

## Getting Started
### Download the data
1. Download EPIC-KITCHENS-100 videos from the [official site](https://github.com/epic-kitchens/epic-kitchens-download-scripts). Since this dataset uses 480p frames and optical flows for training and testing you need to download the original videos. Place them to data/videos/PXX/PXX_XX.MP4.
2. Download and extract the [ground truth label]() and [pseudo-label (large, only required for training)]() to data/.

Required videos are listed in configs/\*_vids.txt.

### Clone repository
```
git clone  --recursive https://github.com/takumayagi/hand_object_contact_prediction.git
```

### Install FlowNet2 submodule
See the [official repo](https://github.com/NVIDIA/flownet2-pytorch) to install the custom components.  
Note that flownet2-pytorch won't work on latest pytorch version (confirmed working in 1.4.0).

### Extract RGB frames
The following code will extract 480p rgb frames to data/rgb_frames.

#### Validation & test set
```
for vid in `cat configs/valid_vids.txt`; do bash preprocessing/extract_rgb_frames.bash $vid; done
for vid in `cat configs/test_vids.txt`; do bash preprocessing/extract_rgb_frames.bash $vid; done
```

#### Trusted training set
```
for vid in `cat configs/trusted_train_vids.txt`; do bash preprocessing/extract_rgb_frames.bash $vid; done
```

#### Noisy training set
```
# Caution: take up large space (~400GBs)
for vid in `cat configs/noisy_train_vids.txt`; do bash preprocessing/extract_rgb_frames.bash $vid; done
```

### Extract Flow frames
Similar to above, we extract flow images (in 16-bit png).
This requires the annotation files since we only extract flows used in training/test to save space.

```
# Same for test, trusted_train, and noisy_train
# Extracting flows for noisy_train will take up large space
for vid in `cat configs/valid_vids.txt`; do python preprocessing/extract_flow_frames.py $vid; done
```

## Demo (WIP)
Currently, we only have evaluation code against pre-processed input sequences (& bounding boxes).
We're planning to release a demo code with track generation.

## Test
Download the [pretrained models]() and place them to pretrained/.

Evaluation by test set:
```
python train.py --model CrUnionLSTMHO --eval --resume pretrained/proposed_model_180000.pth
```

### Visualization
```
python train.py --model CrUnionLSTMHO --eval --resume pretrained/proposed_model_180000.pth --vis
```

## Training
Download the [initialization models]() and place them to pretrained/training/.

```
python train.py --model CrUnionLSTMHO --dir_name proposed --semisupervised --iter_supervision 5000 --iter_warmup 0 --plc --update_clean --init_delta 0.05  --asymp_labeled_flip --nb_iters 800000 --lr_step_list 40000 --save_model --finetune_noisy_net --delta_th 0.01 --iter_snapshot 20000 --iter_evaluation 20000 --min_clean_label_ratio 0.25
```

### Training initialization models
To train the proposed model (CrUnionLSTMHO), we first train a noisy/clean network before applying gPLC.
```
python train_v1.py --model UnionLSTMHO --dir_name noisy_pretrain --train_vids annotations/train_vids_v2_55.txt --nb_iters 40000 --save_model --only_boundary
python train_v1.py --model UnionLSTMHO --dir_name clean_pretrain --train_vids annotations/dev_train_vids_v2.txt --nb_iters 25000 --save_model --iter_warmup 2500 --supervised
```

### Tips
- Set larger --nb_workers an --nb_eval_workers if you have enough number of CPUs.
- You can set --modality to either rgb or flow if training single-modality models.
