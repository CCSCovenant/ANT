# -*- coding: utf-8 -*-

"""
该文件定义了用于处理不同文本编码器的策略工厂。
它将不同文本编码器的处理逻辑（获取原始嵌入、投影嵌入）封装在各自的策略类中，
并通过一个中心工厂函数来创建相应的策略实例。

This file defines the strategy factory for processing different text encoders.
It encapsulates the processing logic (getting raw embeddings, projecting embeddings) for different
text encoders into their respective strategy classes, and creates corresponding strategy
instances through a central factory function.
"""

import torch
import torch.nn as nn
import clip
from abc import ABC, abstractmethod


# =================================================================================
# 1. 策略接口定义 (Strategy Interface Definition)
# =================================================================================

class TextProcessorStrategy(ABC):
    """
    文本处理策略的抽象基类。
    定义了所有具体策略类必须实现的通用接口。
    
    Abstract base class for text processing strategies.
    Defines the common interface that all concrete strategy classes must implement.
    """
    def __init__(self, text_encoder: nn.Module, text_proj: nn.Module, text_ln: nn.Module, device: torch.device, text_transformer: nn.Module = None):
        """
        初始化策略。
        Initializes the strategy.

        Args:
            text_encoder (nn.Module): 冻结的基础文本编码器 (The frozen base text encoder).
            text_proj (nn.Module): 可训练的投影层 (The trainable projection layer).
            text_ln (nn.Module): 可训练的最终 LayerNorm 层 (The trainable final LayerNorm layer).
            device (torch.device): 计算设备 (The computation device).
            text_transformer (nn.Module, optional): 可训练的 Transformer 层，主要为 CLIP 使用 (The trainable Transformer layer, mainly for CLIP). Defaults to None.
        """
        self.text_encoder = text_encoder
        self.text_proj = text_proj
        self.text_ln = text_ln
        self.text_transformer = text_transformer
        self.device = device

    @abstractmethod
    def get_raw_embeds(self, text: list[str]) -> torch.Tensor:
        """
        调用冻结的基础编码器获取原始嵌入。此过程不计算梯度。
        Calls the frozen base encoder to get raw embeddings. This process does not compute gradients.

        Args:
            text (list[str]): 待编码的文本字符串列表 (A list of text strings to be encoded).
        
        Returns:
            torch.Tensor: 原始的文本嵌入张量 (The raw text embedding tensor).
        """
        pass

    @abstractmethod
    def project_embeds(self, raw_embeds: torch.Tensor) -> torch.Tensor:
        """
        将原始嵌入通过可训练的层进行投影，以适配U-Net。
        Projects the raw embeddings through trainable layers to adapt them for the U-Net.

        Args:
            raw_embeds (torch.Tensor): 从 get_raw_embeds 方法获得的原始嵌入 (The raw embeddings from the get_raw_embeds method).

        Returns:
            torch.Tensor: 最终投影完成的文本条件张量 (The final projected text condition tensor).
        """
        pass


# =================================================================================
# 2. 具体策略实现 (Concrete Strategy Implementations)
# =================================================================================

class T5Strategy(TextProcessorStrategy):
    """
    处理 T5 编码器输出的具体策略。
    The concrete strategy for handling T5 encoder outputs.
    """
    def get_raw_embeds(self, text: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder(text)
            
    def project_embeds(self, raw_embeds: torch.Tensor) -> torch.Tensor:
        if raw_embeds.ndim == 2:
            raw_embeds = raw_embeds.unsqueeze(1)
        projected = self.text_proj(raw_embeds.float())
        return self.text_ln(projected)

class ClipStrategy(TextProcessorStrategy):
    """
    处理 CLIP 编码器输出的具体策略。
    该实现严格遵循了手动执行 CLIP 底层编码流程的方式。
    
    The concrete strategy for handling CLIP encoder outputs.
    This implementation strictly follows the manual, low-level CLIP encoding process.
    """
    def get_raw_embeds(self, text: list[str]) -> torch.Tensor:
        """
        手动执行 CLIP 的底层编码流程，以获取原始词元嵌入。
        Manually executes CLIP's low-level encoding process to get raw token embeddings.
        
        Returns:
            torch.Tensor: 形状为 [序列长度, 批次大小, 维度] 的张量 (A tensor of shape [Seq_Len, Batch_Size, Dim]).
        """
        with torch.no_grad():
            tokens = clip.tokenize(text, truncate=True).to(self.device)
            
            x = self.text_encoder.token_embedding(tokens).type(self.text_encoder.dtype)
            x = x + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_encoder.transformer(x)
            x = self.text_encoder.ln_final(x).type(self.text_encoder.dtype)
            return x

    def project_embeds(self, raw_embeds: torch.Tensor) -> torch.Tensor:
        """
        执行可训练的投影流程，包括一个线性层和一个 Transformer 层。
        Executes the trainable projection process, including a linear layer and a Transformer layer.
        
        Args:
            raw_embeds (torch.Tensor): 形状为 [序列长度, 批次大小, 维度] 的原始嵌入 (Raw embeddings of shape [Seq_Len, Batch_Size, Dim]).

        Returns:
            torch.Tensor: 形状为 [批次大小, 序列长度, 维度] 的最终条件张量 (The final condition tensor of shape [Batch_Size, Seq_Len, Dim]).
        """
        x = self.text_proj(raw_embeds.float())
        x = self.text_transformer(x)
        x = self.text_ln(x)
        return x.permute(1, 0, 2)  # LND -> NLD

class GenericStrategy(TextProcessorStrategy):
    """
    适用于 BERT, MoClip, LongClip 等编码器的通用策略。
    A generic strategy applicable to encoders like BERT, MoClip, LongClip, etc.
    """
    def get_raw_embeds(self, text: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.text_encoder(text)

    def project_embeds(self, raw_embeds: torch.Tensor) -> torch.Tensor:
        if raw_embeds.ndim == 2:
            raw_embeds = raw_embeds.unsqueeze(1)
        projected = self.text_proj(raw_embeds.float())
        return self.text_ln(projected)


# =================================================================================
# 3. 策略工厂函数 (Strategy Factory Function)
# =================================================================================

def create_text_processor(config, text_encoder, text_proj, text_ln, text_transformer=None) -> TextProcessorStrategy:
    """
    根据配置创建并返回一个具体的文本处理策略实例。
    Creates and returns a concrete text processing strategy instance based on the configuration.

    Args:
        config: 包含 `text_encoder_type` 和 `device` 的配置对象 (A config object containing `text_encoder_type` and `device`).
        text_encoder (nn.Module): 冻结的基础文本编码器 (The frozen base text encoder).
        text_proj (nn.Module): 可训练的投影层 (The trainable projection layer).
        text_ln (nn.Module): 可训练的最终 LayerNorm 层 (The trainable final LayerNorm layer).
        text_transformer (nn.Module, optional): 可训练的 Transformer 层 (The trainable Transformer layer). Defaults to None.

    Returns:
        TextProcessorStrategy: 一个配置好的文本处理器策略实例 (A configured text processor strategy instance).
    """
    encoder_type = config.text_encoder_type.lower()
    device = config.device
    
    strategy_map = {
        't5': T5Strategy,
        'clip': ClipStrategy,
        'bert': GenericStrategy,
        'moclip': GenericStrategy,
        'longclip': GenericStrategy
    }
    
    strategy_class = strategy_map.get(encoder_type)
    if not strategy_class:
        raise NotImplementedError(f"Strategy for text encoder type '{encoder_type}' is not implemented.")
        
    return strategy_class(text_encoder, text_proj, text_ln, device, text_transformer)