"""
Make ViT a Feature Pyramid Network (FPN) by extracting features from multiple layers.

Stage 1 (layers 0-2):   H/4 × W/4,  dim=C      → "low-level features"
Stage 2 (layers 3-5):   H/8 × W/8,  dim=C      → "mid-level features"  
Stage 3 (layers 6-8):   H/16 × W/16, dim=C     → "high-level features"
Stage 4 (layers 9-11):  H/16 × W/16, dim=C     → "highest-level features"
"""

"""ViT Backbone that outputs multi-scale features for dense prediction tasks (semantic segmentation and object detection)."""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from functools import partial 

class ViTBackbone(nn.Module):
    """
    Wrap your existing ViT to extract multi-scale feature maps. 

    For a 12-layer ViT with patch_size = 16 and input image size 512x512:
    - After patch embeddings: (B, 32*32, C) = (B, 1024, C)
    - Stage features are reshaped to 2D spatial maps 
    - FPN-like neck produces {1/4, 1/8, 1/16, 1/32} resolution feature maps
    """

    def __init__(self, vit_model, emb_dim, img_size=512, patch_size=16, out_indices=(2, 5, 8, 11), fpn_out_dim=256):
        super(ViTBackbone, self).__init__() 

        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_dim = emb_dim
        self.out_indices = out_indices
        self.num_patches_h = img_size // patch_size # 512/16 = 32
        