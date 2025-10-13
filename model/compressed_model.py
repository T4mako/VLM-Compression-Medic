# model/compressed_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoConfig
from logger import logger


class CompressedHuatuoVisionModel(nn.Module):
    def __init__(self, config_path, device="cuda"):
        super().__init__()
        self.device = device

        # 加载原始配置
        self.config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)

        # 创建原始模型结构
        self.model = AutoModelForVision2Seq.from_config(
            self.config,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

        logger.info("压缩模型类初始化完成")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_directory):
        # 保存模型权重和配置
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_path, device="cuda"):
        # 创建模型实例
        model = cls(model_path, device)

        # 加载压缩后的权重
        try:
            state_dict = torch.load(
                f"{model_path}/pytorch_model.bin",
                map_location=device
            )
            model.model.load_state_dict(state_dict, strict=False)
            logger.info("压缩模型权重加载成功")
        except Exception as e:
            logger.warning(f"加载压缩权重失败: {e}")
            logger.info("将使用随机初始化的权重")

        return model