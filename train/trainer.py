import swanlab
from config import Config
from logger import logger
from model.load_model import load_huatuo_vision_model
from model.obr import apply_obr_to_linear
from data.dataset import load_pubmed_vision
import torch

class OBRTrainer:
    def __init__(self, config: Config):
        self.config = config
        if config.training.use_swanlab:
            swanlab.init(
                project=config.training.swanlab_project,
                name=config.training.swanlab_name
            )

    # 注册 hook 采集每一层的输入激活，并返回平均激活值，为模型压缩（如 OBR、量化、低秩分解）提供统计基础
    def calibrate_activations(self, model, processor, calib_data):
        """
            激活字典，存储各层的激活值
            键：层名称，值：该层的激活值列表
        """
        activations = {}

        """ 用于保存激活的钩子函数 """
        def hook_fn(module, input, output, name):
            print(f"[DEBUG] 激活形状 {name}: {input[0].shape}")
            if isinstance(input, tuple):
                input = input[0]
            if name not in activations:
                activations[name] = []
            # 将当前层的输入激活值添加到对应列表中
            activations[name].append(input.detach().cpu())

        # 注册 hooks，保存所有注册的钩子句柄
        hooks = []
        # 遍历模型所有模块，为 Linear 层注册钩子
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 注册前向传播钩子
                hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

        # 校准循环
        for i, sample in enumerate(calib_data[:self.config.data.calib_size]):
            images = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
            inputs = processor(images, sample["input_text"], return_tensors="pt").to(model.device) # 返回PyTorch张量格式
            # 前向传播（无梯度计算）
            with torch.no_grad():
                model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Average activations
        avg_activations = {}
        for name, acts in activations.items():
            avg_activations[name] = torch.cat(acts, dim=0).mean(dim=0)  # [L, Cin] -> [Cin]

            logger.info(f"Layer {name} - avg: {avg_activations[name].shape}")

        return avg_activations

    def compress_model(self):
        model, processor = load_huatuo_vision_model(
            self.config.model.model_name,
            device=self.config.training.device,
            vit_compress_mode=self.config.model.vit_compress_mode,
            vit_target_tokens=self.config.model.vit_target_tokens
        )
        # 打印模型结构，确认模型加载正确
        logger.info(f"[Model] 已加载模型: {self.config.model.model_name}")
        calib_data = load_pubmed_vision(self.config)
        logger.info(f"[dataset] 激活数据量 {len(calib_data)} samples")
        activations = self.calibrate_activations(model, processor, calib_data)
        logger.info(f"[dataset] 激活校准已完成 {activations}")

        # 获取语言模型
        lm = getattr(model, self.config.model.language_model_name)
        logger.info(f"语言模型结构: {lm}")
        # 遍历语言模型的所有模块，为 Linear 层应用 OBR
        for name, module in lm.named_modules():
            if isinstance(module, torch.nn.Linear):
                act = activations.get(name)
                if act is None:
                    logger.warning(f"No activation found for layer {name}, using random input")
                    act = torch.randn(module.in_features)
                else:
                    logger.info(f"Compressing layer {name} with activation shape {act.shape}")
                apply_obr_to_linear(
                    module,
                    act,
                    bits=self.config.compression.lm_bits,
                    sparsity=self.config.compression.lm_sparsity,
                    alpha=self.config.compression.alpha,
                    device=model.device
                )

        # 获取视觉编码器组件
        vision_encoder = getattr(model, self.config.model.vision_encoder_name)
        for name, module in vision_encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                act = activations.get(name, torch.randn(module.in_features))
                apply_obr_to_linear(
                    module,
                    act,
                    bits=self.config.compression.vision_bits,
                    sparsity=self.config.compression.vision_sparsity,
                    alpha=self.config.compression.alpha,
                    device=model.device
                )


        logger.info("Model compression (LM + Vision) completed!")
        return model, processor