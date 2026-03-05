'''ADE20K and COCO segmentation training and evaluation script.'''

import argparse 
import copy 
import os 
import os.path as osp 
import time 
import warnings 

import mmcv 
import torch 
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version 
from mmseg.apis import set_random_seed, train_segmentor, init_random_seed 
from mmseg.datasets import build_dataset 
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes



"""
Semantic Segmentation (ADE20K)
- Task: Assign a class label to every pixel in an image. 
- ADE20K: 150 semantic classes, ~20K training images, 2K validation images, 3K testing images.
- Output: (B, num_classes, H, W) - one class prediction per pixel. 

Framework - mmsegmentation (OpenMMLab)
Head - UPerNet or Linear Decoder 
"""

"""
Object Detection (COCO)
- Task: Predict bounding boxes + class labels for every object in an image. 
- COCO: 80 object classes, ~118K training images, 5K validation images
- Output: A set of bounding boxes (x, y, w, h), class labels, and confidence scores for each detected object.

Framework - mmdetection (OpenMMLab)
Head - Mask R-CNN or Deformable DETR
"""