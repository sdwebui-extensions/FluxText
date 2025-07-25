import argparse
import math
import os
import os.path as osp
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from diffusers import FluxTransformer2DModel
import lightning as L
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from tqdm import tqdm
import yaml

from fluxtext.condition import Condition
from fluxtext.generate_fill import generate_fill, generate_fill_basic
from fluxtext.model import OminiModelFIll
from fluxtext.model_lora import OminiModelLORAFIll
import folder_paths

ASPECT_RATIO_LD_LIST = [  # width:height
    "2.39:1",  # cinemascope, 2.39
    "2:1",  # rare, 2
    "16:9",  # rare, 1.89
    "1.85:1",  # american widescreen, 1.85
    "9:16",  # popular, 1.78
    "5:8",  # rare, 1.6
    "3:2",  # rare, 1.5
    "4:3",  # classic, 1.33
    "1:1",  # square
]

RESOLUTIONS = [512, 768, 1024]
PIXELS = [512 * 512, 768 * 768, 1024 * 1024]

def get_ratio(name: str) -> float:
    width, height = map(float, name.split(":"))
    return height / width

def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    aspect_ratio = height / width
    closest_ratio = min(
        ratios, key=lambda ratio: abs(aspect_ratio - get_ratio(ratio))
    )
    return closest_ratio

def get_aspect_ratios_dict(
    total_pixels: int = 256 * 256, training: bool = True
) -> dict[str, tuple[int, int]]:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    aspect_ratios_dict = {}
    aspect_ratios_vertical_dict = {}
    for ratio in ASPECT_RATIO_LD_LIST:
        width_ratio, height_ratio = map(float, ratio.split(":"))
        width = int(math.sqrt(total_pixels * (width_ratio / height_ratio)) // D) * D
        height = int((total_pixels / width) // D) * D

        if training:
            # adjust aspect ratio to match total pixels
            diff = abs(height * width - total_pixels)
            candidate = [
                (height - D, width),
                (height + D, width),
                (height, width - D),
                (height, width + D),
            ]
            for h, w in candidate:
                if abs(h * w - total_pixels) < diff:
                    height, width = h, w
                    diff = abs(h * w - total_pixels)

        # remove duplicated aspect ratio
        if (height, width) not in aspect_ratios_dict.values() or not training:
            aspect_ratios_dict[ratio] = (height, width)
            vertial_ratios = ":".join(ratio.split(":")[::-1])
            aspect_ratios_vertical_dict[vertial_ratios] = (width, height)

    aspect_ratios_dict.update(aspect_ratios_vertical_dict)

    return aspect_ratios_dict


def init_pipeline(model_path, config):
    training_config = config["train"]

    trainable_model = OminiModelFIll(
            flux_pipe_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )
    
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    state_dict_new = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
    trainable_model.transformer.load_state_dict(state_dict_new, strict=False)

    pipe = trainable_model.flux_pipe
    
    return pipe, trainable_model

def init_pipeline_lora(model, model_path, config):
    training_config = config["train"]

    trainable_model = OminiModelLORAFIll(
            fill_model=model,
            flux_pipe_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )
    
    from safetensors.torch import load_file
    state_dict = load_file(model_path)
    state_dict_new = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
    trainable_model.transformer.load_state_dict(state_dict_new, strict=False)

    pipe = trainable_model.flux_pipe
    
    return pipe, trainable_model

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class FLUXTextLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_path":  ("STRING", {"default": "flux-text/flux-text.safetensors", "multiline": True, "label": "lora_path"}), }}
    
    RETURN_TYPES = ("FLUXText_PIPE", "FLUXText_Config")
    FUNCTION = "load_model"
    CATEGORY = "FLUXText"

    def load_model(self, model_path):
        _dirname = osp.dirname(model_path)
        config_path = osp.join(_dirname, 'config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pipe, model = init_pipeline(model_path, config)
        
        return (pipe, config)
    
class FLUXFillTransformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "transformer_name": (folder_paths.get_filename_list("diffusion_models"), ),
                            #   "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
        }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_transformer"

    CATEGORY = "FLUXText"

    def load_transformer(self, transformer_name):
        transformer_name = folder_paths.get_full_path_or_raise("diffusion_models", transformer_name)
        transformer = FluxTransformer2DModel.from_single_file(transformer_name)
        
        return (transformer,)
    

class FLUXTextLORALoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "model": ("MODEL", ),
                    "lora_path": ("STRING", {"default": "flux-text/flux-text.safetensors", "multiline": True, "label": "lora_path"}), 
                    }
                }
    
    RETURN_TYPES = ("FLUXText_PIPE", "FLUXText_Config")
    FUNCTION = "load_model"
    CATEGORY = "FLUXText"

    def load_model(self, model, lora_path):
        _dirname = osp.dirname(lora_path)
        config_path = osp.join(_dirname, 'config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pipe, model = init_pipeline_lora(model, lora_path, config)
        
        return (pipe, config)
    
    
class FLUXTextGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("FLUXText_PIPE", ),
                "config": ("FLUXText_Config", ),
                "img": ("IMAGE",),
                "glyph_img": ("IMAGE",),
                "mask_img": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "FLUXText"

    def generate(self, pipe, config, img, glyph_img, mask_img, prompt, num_inference_steps, seed):
        img = tensor2pil(img)
        glyph_img = tensor2pil(glyph_img)
        mask_img = tensor2pil(mask_img)

        ori_width, ori_height = img.size
        num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
        aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
        close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
        tgt_height, tgt_width = aspect_ratio_dict[close_ratio]
        
        hint = mask_img.resize((tgt_width, tgt_height)).convert('RGB')
        img = img.resize((tgt_width, tgt_height))
        condition_img = glyph_img.resize((tgt_width, tgt_height)).convert('RGB')
        hint = np.array(hint) / 255
        condition_img = np.array(condition_img)
        condition_img = (255 - condition_img) / 255
        condition_img = [condition_img, hint, img]
        position_delta = [0, 0]
        condition = Condition(
                        condition_type='word_fill',
                        condition=condition_img,
                        position_delta=position_delta,
                    )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        res = generate_fill(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=tgt_height,
            width=tgt_width,
            generator=generator,
            model_config=config.get("model", {}),
            default_lora=True,
            num_inference_steps=num_inference_steps,
        )
        
        image = res.images[0]
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).float()
        # Add batch dimension to make it [batch, height, width, channels]
        if image.dim() == 3:  # [height, width, channels]
            image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
            
        return (image,)
        return (res,)
    
class FLUXTextMaskImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mask": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked_image",)
    FUNCTION = "generate"
    CATEGORY = "FLUXText"

    def generate(self, img, mask):
        masked_image = img * (1 - mask)
        return (masked_image, )
    
class FLUXTextAutoSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE", ),
                "glyph_img": ("IMAGE",),
                "mask": ("IMAGE", ),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("img", "glyph_img", "mask", "tgt_height", "tgt_width")
    FUNCTION = "generate"
    CATEGORY = "FLUXText"

    def generate(self, img, glyph_img, mask):
        img_pil = tensor2pil(img)

        ori_width, ori_height = img_pil.size
        num_pixel = min(PIXELS, key=lambda x: abs(x - ori_width * ori_height))
        aspect_ratio_dict = get_aspect_ratios_dict(num_pixel)
        close_ratio = get_closest_ratio(ori_height, ori_width, ASPECT_RATIO_LD_LIST)
        tgt_height, tgt_width = aspect_ratio_dict[close_ratio]

        img = img.permute(0, 3, 1, 2)
        glyph_img = glyph_img.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=(tgt_height, tgt_width), mode='bilinear', align_corners=False)
        glyph_img = F.interpolate(glyph_img, size=(tgt_height, tgt_width), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(tgt_height, tgt_width), mode='bilinear', align_corners=False)

        img = img.permute(0, 2, 3, 1)
        glyph_img = glyph_img.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        return img, glyph_img, mask, tgt_height, tgt_width

class FLUXTextGenerateBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("FLUXText_PIPE", ),
                "config": ("FLUXText_Config", ),
                "CONDITIONING": ("CONDITIONING",),
                "img_latent": ("LATENT",),
                "glyph_img_latent": ("LATENT",),
                "mask": ("IMAGE",),
                "tgt_height": ("INT",),
                "tgt_width": ("INT",),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "FLUXText"

    def generate(self, pipe, config, CONDITIONING, img_latent, glyph_img_latent, mask, num_inference_steps, tgt_height, tgt_width, seed):
        mask = mask[:,:,:,0]
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        res = generate_fill_basic(
            pipe,
            CONDITIONING=CONDITIONING,
            img_latent=img_latent,
            glyph_img_latent=glyph_img_latent,
            mask=mask,
            height=tgt_height,
            width=tgt_width,
            generator=generator,
            model_config=config.get("model", {}),
            default_lora=True,
            num_inference_steps=num_inference_steps,
            output_type="latent",
        )
        
        # image = res.images[0]
        # image = np.array(image) / 255.0
        # image = torch.from_numpy(image).float()
        # # Add batch dimension to make it [batch, height, width, channels]
        # if image.dim() == 3:  # [height, width, channels]
        #     image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
            
        # return (image,)
        res = {"samples": res.images}
        return (res,)