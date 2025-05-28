# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderConfig, AutoModel, AutoModelForCausalLM
from torch import amp
from tqdm import tqdm
from modules.surya_order.config import MBartOrderConfig, VariableDonutSwinConfig
from modules.surya_order.decoder import MBartOrder
from modules.surya_order.encoder import VariableDonutSwinModel
from modules.surya_order.encoderdecoder import OrderVisionEncoderDecoderModel
from modules.surya_order.processor import OrderImageProcessor
from modules.surya_order.settings import settings

def load_model_and_processor(checkpoint_path,train_mode=False,dtype=torch.float32):
    """加载模型与处理器（强制使用float32）"""
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint_path)

    # 自定义编解码器配置
    decoder = MBartOrderConfig(**vars(config.decoder))
    encoder = VariableDonutSwinConfig(**vars(config.encoder))
    config.decoder = decoder
    config.encoder = encoder

    # 注册自定义模型
    AutoModel.register(MBartOrderConfig, MBartOrder)
    AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    # 强制使用float32加载模型
    model = OrderVisionEncoderDecoderModel.from_pretrained(
        checkpoint_path, 
        config=config, 
        torch_dtype=dtype  # 始终使用float32
    )
    if train_mode:
        model = model.train()
    else:
        model = model.eval()

    # 初始化处理器
    processor = OrderImageProcessor.from_pretrained(checkpoint_path)
    processor.size = settings.ORDER_IMAGE_SIZE
    box_size = 1024
    max_tokens = 256
    processor.token_sep_id = max_tokens + box_size + 1
    processor.token_pad_id = max_tokens + box_size + 2
    processor.max_boxes = settings.ORDER_MAX_BOXES - 1
    processor.box_size = {"height": box_size, "width": box_size}

    return model, processor

def load_model(checkpoint_path,train_mode=False,dtype=torch.float32):
    """加载模型与处理器（强制使用float32）"""
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint_path)

    # 自定义编解码器配置
    decoder = MBartOrderConfig(**vars(config.decoder))
    encoder = VariableDonutSwinConfig(**vars(config.encoder))
    config.decoder = decoder
    config.encoder = encoder

    # 注册自定义模型
    AutoModel.register(MBartOrderConfig, MBartOrder)
    AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    # 强制使用float32加载模型
    model = OrderVisionEncoderDecoderModel.from_pretrained(
        checkpoint_path, 
        config=config, 
        torch_dtype=dtype  # 始终使用float32
    )
    if train_mode:
        model = model.train()
    else:
        model = model.eval()

    return model

def load_processor(checkpoint_path):
    # 初始化处理器
    processor = OrderImageProcessor.from_pretrained(checkpoint_path)
    processor.size = settings.ORDER_IMAGE_SIZE
    box_size = 1024
    max_tokens = 256
    processor.token_sep_id = max_tokens + box_size + 1
    processor.token_pad_id = max_tokens + box_size + 2
    processor.max_boxes = settings.ORDER_MAX_BOXES - 1
    processor.box_size = {"height": box_size, "width": box_size}
    return processor
