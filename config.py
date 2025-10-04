import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DataConfig:
    local_root: str = "../machine-unlearning/data/PubMedVision_repo"
    alignment_path: str = "../machine-unlearning/data/PubMedVision_Alignment_VQA"
    cache_dir: str = "./data/cache"
    debug_limit: Optional[int] = None
    dn_ratio: float = 1.0  # 保留比例
    calib_size: int = 256  # 校准集大小

@dataclass
class ModelConfig:
    model_name: str = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
    vision_encoder_name: str = "vision_tower"
    projector_name: str = "mm_projector"
    language_model_name: str = "language_model"
    use_rotation: bool = True
    rotation_type: str = "FlatQuant"  # "QuaRot", "SpinQuant", "FlatQuant"
    vit_compress_mode: str = "pooling"      # "pooling" or "token_pruning"
    vit_target_tokens: int = 64             # 推荐 64~100

@dataclass
class CompressionConfig:
    # Language branch
    lm_bits: int = 4
    lm_sparsity: float = 0.5
    lm_use_obr: bool = True
    lm_quantizer: str = "GPTQ"  # "RTN", "GPTQ"

    # Vision branch
    vision_bits: int = 4
    vision_sparsity: float = 0.5
    vision_use_obr: bool = True
    vision_pruner: str = "WANDA"

    # Projector
    projector_bits: int = 16  # 16=FP16, 8=INT8
    projector_sparsity: float = 0.0

    # OBR specific
    alpha: float = 0.5  # quantization grouping ratio

@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    seed: int = 42
    device: str = "cuda"
    use_swanlab: bool = True
    swanlab_project: str = "OBR-Med-VLM"
    swanlab_name: str = "default_run"

@dataclass
class EvalConfig:
    metrics: List[str] = field(default_factory=lambda: ["perplexity"])
    eval_batch_size: int = 1

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_args(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default=None)
        # 可扩展更多命令行参数
        args = parser.parse_args()
        if args.config:
            import json
            with open(args.config) as f:
                data = json.load(f)
                return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        return cls()