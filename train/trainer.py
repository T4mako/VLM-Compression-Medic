import swanlab
from ..config import Config
from ..logger import logger
from ..model.load_model import load_huatuo_vision_model
from ..model.obr import apply_obr_to_linear
from ..data.dataset import load_pubmed_vision
import torch

class OBRTrainer:
    def __init__(self, config: Config):
        self.config = config
        if config.training.use_swanlab:
            swanlab.init(
                project=config.training.swanlab_project,
                name=config.training.swanlab_name
            )

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
        model, processor = load_huatuo_vision_model(self.config.model.model_name)
        calib_data = load_pubmed_vision(self.config)
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

        # TODO: compress vision encoder similarly

        logger.info("Model compression completed!")
        return model, processor