import os
import json
import logging
from pathlib import Path
from PIL import Image
from datasets import load_from_disk, load_dataset
from typing import Dict, List, Optional
from config import Config
from logger import logger
from tqdm import tqdm

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

def _process_sample(ex: Dict, local_root: str) -> Optional[Dict]:
    try:
        images = ex.get("image", [])
        if not images:
            return None

        # 检查所有图片路径是否存在（相对路径需拼接 local_root）
        full_paths = [os.path.join(local_root, p) for p in images]
        if not all(os.path.exists(fp) for fp in full_paths):
            logger.debug(f"Skipping sample due to missing image(s): {full_paths}")
            return None

        conversation = ex.get("conversations", [])
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
    if config.data.debug_limit:
        debug_limit = config.data.debug_limit
    else:
        debug_limit = None  # 表示不限制
    cache_file = os.path.join(config.data.cache_dir, f"pubmed_vision_cache{debug_limit or 'full'}.pt")
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
    logger.info(f"Dataset size: {len(dset)} samples")

    intermediate_file = _intermediate_path(config.data.dn_ratio, debug_limit or 0)
    progress_file = _progress_path(config.data.dn_ratio, debug_limit or 0)

    existing = _load_intermediate_samples(config.data.dn_ratio, debug_limit or 0)
    start_idx = len(existing)
    logger.info(f"Resuming from index {start_idx}, existing samples: {len(existing)}")

    # 目标数量
    target_count = debug_limit if debug_limit else float('inf')
    processed_idx = start_idx  # 当前处理到的原始数据集索引

    # 从 progress 文件恢复已处理到的原始索引（更准确）
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            progress = json.load(pf)
            processed_idx = progress.get("last_index", start_idx)
        logger.info(f"Resuming raw dataset index from progress file: {processed_idx}")

    with open(intermediate_file, "a") as f_out:
        pbar = tqdm(total=target_count - len(existing), desc="Loading PubMedVision", unit="valid_sample")
        while len(existing) < target_count and processed_idx < len(dset):
            ex = dset[processed_idx]

            if processed_idx < 5:  # debug log for first few
                logger.debug(f"====Processing dataset index: {processed_idx}====")
                logger.debug(f"==Raw sample: {ex}==")

            sample = _process_sample(ex, _DATASET_LOCAL_ROOT)
            if sample:
                f_out.write(json.dumps(sample) + "\n")
                existing.append(sample)
                pbar.update(1)

            processed_idx += 1

            # Save progress every 1000 raw samples
            if processed_idx % 1000 == 0:
                with open(progress_file, "w") as pf:
                    json.dump({"last_index": processed_idx}, pf)

        pbar.close()

    # Build final items
    if not existing:
        logger.warning("No valid samples found!")
        items = []
    else:
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

    logger.info(f"Final items found: {len(items)}")
    if items:
        logger.info(f"Example item: {items[0]}")

    # Cache
    import torch
    torch.save(items, cache_file)
    logger.info(f"Dataset cached to {cache_file}, total samples: {len(items)}")

    # 清理 progress 文件（可选）
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return items