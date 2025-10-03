from transformers import AutoModelForCausalLM, AutoProcessor
from ..model.obr import apply_obr_to_linear
from logger import logger
import torch

def load_huatuo_vision_model(model_name: str, device: str = "cuda"):
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor