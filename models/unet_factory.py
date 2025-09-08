# ./models/unet_factory.py

from .t2m_unet import T2MUnet
from .submodules.text_encoders import get_text_encoder

class UnetFactory:
    """
    Factory for creating T2M-Unet models with different configurations.
    用于创建不同配置的 T2M-Unet 模型的工厂。
    """
    @staticmethod
    def create_unet(config):
        """
        Creates and returns a T2MUnet model based on the provided configuration.
        根据提供的配置创建并返回一个 T2MUnet 模型。

        Args:
            config (object or dict): A configuration object containing all necessary parameters.
                                     一个包含所有必要参数的配置对象。
                                     Expected attributes:
                                     - text_encoder_type (str)
                                     - device (str)
                                     - input_feats (int)
                                     - text_latent_dim (int)
                                     - base_dim (int)
                                     - dim_mults (tuple)
                                     - ... (all other params for T2MUnet and its submodules)

        Returns:
            T2MUnet: An initialized T2M-Unet model.
                     一个已初始化的 T2M-Unet 模型。
        """
        print(f"Creating UNet with text encoder: '{config.text_encoder_type}'")
        
        # 1. Get the pre-trained text encoder and its feature dimension
        # 1. 获取预训练的文本编码器及其特征维度
        text_encoder, text_encoder_dim = get_text_encoder(config, config.device)
        
        # 2. Create the main T2MUnet model, injecting the text encoder
        # 2. 创建主 T2MUnet 模型，并注入文本编码器
        model = T2MUnet(
            config=config,
            text_encoder=text_encoder,
            text_encoder_dim=text_encoder_dim
        )
        
        print("T2M-Unet model created successfully.")
        return model.to(config.device)