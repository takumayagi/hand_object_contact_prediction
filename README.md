# Hand-Object Contact Prediction (BMVC2021)
This repository contains the code and data for the paper "Hand-Object Contact Prediction via Motion-Based Pseudo-Labeling and Guided Progressive Label Correction" by Takuma Yagi, Md. Tasnimul Hasan and Yoichi Sato.

**Under Construction: We will publish the code and data as soon as possible.**

## Requirements
* Python 3.6+
* numpy
* opencv-python
* pillow
* scikit-learn
* python-Levenshtein
* pycocotools
* torch (1.8.1, 1.4.0 for flow generation)
* torchvision (0.9.1)
* [mllogger](https://github.com/takumayagi/mllogger)
* [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)https://github.com/NVIDIA/flownet2-pytorch

## Getting Started
### Download the data
1. Download EPIC-KITCHENS-100 videos from the [official site](https://github.com/epic-kitchens/epic-kitchens-download-scripts). Since this dataset uses 480p frames and optical flows for training and testing you need to download the original videos.
2. Download and extract the [ground truth label]() and [pseudo-label (large, only required for training)]() to data/.

### Clone repository
```
git clone https://github.com/takumayagi/hand_object_contact_prediction.git --recursive
```

### Extract RGB frames

### Extract Flow frames


## Test

## Training
