# model/load_model.py

from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from .obr import apply_obr_to_linear
from .prune import DynamicViTCompressor
from logger import logger
import torch
import os

def load_huatuo_vision_model(
    model_name: str,
    device: str = "cuda",
    vit_compress_mode: str = "pooling",      # 支持 'pooling', 'token_pruning'
    vit_target_tokens: int = 64              # 目标 token 数，如 64 = 8x8
):
    logger.info(f"Loading model: {model_name}")
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    
    
    # 加载模型
    if "Qwen2" in model_name and "VL" in model_name:
        logger.info(f"Loading Qwen VL model with AutoModelForVision2Seq from local cache")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=cache_dir
        ).to(device)
    else:
        logger.info(f"Loading model with AutoModelForCausalLM from local cache")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=cache_dir
        ).to(device)

    print("Weight dtype:", model.language_model.model.layers[0].self_attn.q_proj.weight.dtype)
        
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=cache_dir
    )

    # === 插入 Dynamic-VLM 的 ViT 压缩器 ===
    # 假设 Qwen2.5-VL 的视觉编码器为 model.vision_model
    if hasattr(model, 'vision_model'):
        logger.info(f"Inserting DynamicViTCompressor (mode={vit_compress_mode}, tokens={vit_target_tokens})")
        model.vit_compressor = DynamicViTCompressor(
            mode=vit_compress_mode,
            target_tokens=vit_target_tokens
        ).to(device)
    else:
        logger.warning("No 'vision_model' found in model. Skipping ViT compression.")

    return model, processor