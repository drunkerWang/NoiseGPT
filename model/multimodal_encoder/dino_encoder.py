import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoImageProcessor, AutoModel, AutoConfig

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_dino_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

parser = transformers.HfArgumentParser(
    (ModelArguments))
model_args = parser.parse_args_into_dataclasses()

class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()


        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        #self.load_model()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        print("I used DINO model!")
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        #self.config = AutoConfig.from_pretrained('/home/simonzhai/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/cba6c51934b2e3f33842c4bcf922dc9e7b15083f')
        
        #self.vision_tower = AutoModel.from_pretrained('/home/simonzhai/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/cba6c51934b2e3f33842c4bcf922dc9e7b15083f', config=self.config)
        
        self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs["x_prenorm"]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.clip_vision_tower.dtype

    @property
    def device(self):
        return self.clip_vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.clip_vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 1024

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 256

