from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from model.obr import apply_obr_to_linear
from logger import logger
import torch
import os

def load_huatuo_vision_model(model_name: str, device: str = "cuda"):
    logger.info(f"Loading model: {model_name}")
    
    # 从本地缓存加载模型，不下载
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # 根据模型名称选择合适的加载方式
    if "Qwen2" in model_name and "VL" in model_name:
        # Qwen2.5VL 使用 AutoModelForVision2Seq
        logger.info(f"Loading Qwen VL model with AutoModelForVision2Seq from local cache")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,  # 只使用本地文件
            cache_dir=cache_dir     # 指定缓存目录
        ).to(device)
    else:
        # 其他模型使用 AutoModelForCausalLM
        logger.info(f"Loading model with AutoModelForCausalLM from local cache")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,  # 只使用本地文件
            cache_dir=cache_dir     # 指定缓存目录
        ).to(device)
        
    # 从本地缓存加载处理器
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True,
        local_files_only=True,  # 只使用本地文件
        cache_dir=cache_dir     # 指定缓存目录
    )
    
    return model, processor