# ./models/submodules/text_encoders.py

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
import clip
from models.BERT.BERT_encoder import load_bert 
from models.CLIPS.Moclip.moclip import EvalWarperMoClip 
from models.CLIPS.LongCLIP.model import longclip 

# 核心改动：不再导入 json 和 os，而是导入我们自己的配置类
from .encoder_config import EncoderPaths


class T5TextEmbedder(nn.Module):
    """
    T5 Text Encoder.
    T5文本编码器。
    """
    def __init__(self, pretrained_path, max_length=77):
        super().__init__()
        print(f"Loading T5 model from: {pretrained_path}")
        self.model = T5EncoderModel.from_pretrained(pretrained_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        self.max_length = max_length

    def forward(
        self, caption, text_input_ids=None, attention_mask=None, max_length=None
    ):
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(
                    caption,
                    padding=True,  # 启用填充
                    truncation=True,  # 启用截断
                    max_length=self.max_length,  # 设置最大长度
                    return_tensors="pt"  # 返回 PyTorch 张量
                )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings

def get_text_encoder(config, device='cuda'):
    """
    Factory function to get a text encoder based on config.
    根据配置获取文本编码器的工厂函数。
    """
    encoder_type = config.text_encoder_type.lower()
    
    # 现在直接从导入的 EncoderPaths 类中访问路径
    
    if encoder_type == "t5":
        max_len = getattr(config, "laten_size", 77)
        encoder_dim = getattr(config, "t5_dim", 2048)
        # 直接使用类属性，非常清晰
        encoder = T5TextEmbedder(pretrained_path=EncoderPaths.T5, max_length=max_len).to(device, dtype=torch.float32)

    elif encoder_type == "clip":
        encoder_dim = getattr(config, "clip_dim", 512)
        model, _ = clip.load(EncoderPaths.CLIP_VERSION, device='cpu', jit=False)
        encoder = model.to(device)

    elif encoder_type == "bert":
        encoder_dim = getattr(config, "clip_dim", 768)
        encoder = load_bert(EncoderPaths.BERT)
        
    elif encoder_type == "longclip":
        model, _ = longclip.load(EncoderPaths.LONGCLIP, device=device)
        encoder = model
        encoder_dim = 512
        
    elif encoder_type == "moclip":
        encoder = EvalWarperMoClip(model_path=EncoderPaths.MOCLIP)
        encoder_dim = 768
        
    else:
        raise ValueError(f"Unsupported text encoder type: {encoder_type}")

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    return encoder, encoder_dim