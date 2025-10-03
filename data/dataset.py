import os
import json
import logging
from pathlib import Path
from PIL import Image
from datasets import load_from_disk, load_dataset
from typing import Dict, List, Optional
from config import Config
from logger import logger

_DATASET_LOCAL_ROOT = None

def _intermediate_path(dn_ratio: float, debug_limit: int):
    return f"./data/cache/intermediate_dn{dn_ratio}_limit{debug_limit}.jsonl"

def _progress_path(dn_ratio: float, debug_limit: int):
    return f"./data/cache/progress_dn{dn_ratio}_limit{debug_limit}.json"

def _load_intermediate_samples(dn_ratio: float, debug_limit: int):
    path = _intermediate_path(dn_ratio, debug_limit)
    if not os.path.exists(path):
        return []
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def _process_sample(ex: Dict) -> Optional[Dict]:
    try:
        images = ex.get("image", [])
        if not images:
            return None
        # 只保留路径，不加载 PIL（推理时再加载）
        conversation = ex.get("conversation", [])
        if not conversation:
            return None
        human = conversation[0]["value"] if len(conversation) > 0 else ""
        gpt = conversation[1]["value"] if len(conversation) > 1 else ""
        return {
            "image": images,
            "human": human,
            "gpt": gpt,
            "modality": ex.get("modality", "unknown"),
            "body_part": ex.get("body_part", "unknown"),
            "id": ex.get("id", "")
        }
    except Exception as e:
        logger.warning(f"Failed to process sample: {e}")
        return None

def load_pubmed_vision(config: Config):
    cache_file = os.path.join(config.data.cache_dir, "pubmed_vision_cache.pt")
    os.makedirs(config.data.cache_dir, exist_ok=True)

    if os.path.exists(cache_file):
        logger.info("Loading from cached dataset")
        import torch
        return torch.load(cache_file)

    logger.info("Building dataset from scratch...")

    # Load alignment VQA
    try:
        ds = load_from_disk(config.data.alignment_path)
    except:
        logger.info("Fallback to HuggingFace Hub")
        ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA")
        ds.save_to_disk(config.data.alignment_path)

    global _DATASET_LOCAL_ROOT
    _DATASET_LOCAL_ROOT = os.path.abspath(config.data.local_root)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    dset = ds[split_name]

    # Process samples
    intermediate_file = _intermediate_path(config.data.dn_ratio, config.data.debug_limit or 0)
    progress_file = _progress_path(config.data.dn_ratio, config.data.debug_limit or 0)

    existing = _load_intermediate_samples(config.data.dn_ratio, config.data.debug_limit or 0)
    start_idx = len(existing)

    with open(intermediate_file, "a") as f_out:
        for i in range(start_idx, len(dset)):
            if config.data.debug_limit and len(existing) >= config.data.debug_limit:
                break
            sample = _process_sample(dset[i])
            if sample:
                f_out.write(json.dumps(sample) + "\n")
                existing.append(sample)
            if (i + 1) % 1000 == 0:
                with open(progress_file, "w") as pf:
                    json.dump({"last_index": i + 1}, pf)

    # Build final items
    modalities = sorted({s["modality"] for s in existing})
    modality2id = {m: i for i, m in enumerate(modalities)}
    items = []
    for s in existing:
        items.append({
            "image_paths": [os.path.join(_DATASET_LOCAL_ROOT, p) for p in s["image"]],
            "input_text": s["human"],
            "target_text": s["gpt"],
            "modality_label": modality2id[s["modality"]],
        })

    # Cache
    import torch
    torch.save(items, cache_file)
    logger.info(f"Dataset cached to {cache_file}, total samples: {len(items)}")
    return items