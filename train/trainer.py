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
        """Hook to collect activations for Hessian"""
        activations = {}

        def hook_fn(module, input, output, name):
            if isinstance(input, tuple):
                input = input[0]
            if name not in activations:
                activations[name] = []
            activations[name].append(input.detach().cpu())

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

        # Run calibration
        for i, sample in enumerate(calib_data[:self.config.data.calib_size]):
            images = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
            inputs = processor(images, sample["input_text"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Average activations
        avg_activations = {}
        for name, acts in activations.items():
            avg_activations[name] = torch.cat(acts, dim=0).mean(dim=0)  # [L, Cin] -> [Cin]

        return avg_activations

    def compress_model(self):
        model, processor = model, processor = load_huatuo_vision_model(
            Config.model.model_name,
            device=Config.training.device,
            vit_compress_mode=Config.model.vit_compress_mode,
            vit_target_tokens=Config.model.vit_target_tokens
        )
        calib_data = load_pubmed_vision(Config)
        activations = self.calibrate_activations(model, processor, calib_data)

        # Compress language model
        lm = getattr(model, self.config.model.language_model_name)
        for name, module in lm.named_modules():
            if isinstance(module, torch.nn.Linear):
                act = activations.get(name, torch.randn(module.in_features))
                apply_obr_to_linear(
                    module,
                    act,
                    bits=self.config.compression.lm_bits,
                    sparsity=self.config.compression.lm_sparsity,
                    alpha=self.config.compression.alpha,
                    device=model.device
                )

        # compress vision encoder similarly
        vision_encoder = getattr(model, self.config.model.vision_encoder_name)  # e.g. "vision_tower" or "vision_encoder"
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