# openANT/models/submodules/encoder_config.py

class EncoderPaths:
    """
    一个用于存储所有文本编码器模型路径的配置类。
    所有路径都作为类属性进行管理，方便导入和使用。
    """
    
    # --- 请在这里配置你的模型路径 ---

    # T5 模型路径
    T5 = "./T5"

    # BERT 模型路径或HuggingFace标识符
    BERT = "distilbert/distilbert-base-uncased"

    # LongCLIP 模型文件的绝对路径
    LONGCLIP = "/path/to/your/longclip_model.pth"

    # MoClip 模型文件的绝对路径
    MOCLIP = "/path/to/your/moclip_model.pth"

    # CLIP 版本号
    CLIP_VERSION = "ViT-B/32"