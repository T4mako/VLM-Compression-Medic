import swanlab
from config import Config
from logger import logger
from model.load_model import load_huatuo_vision_model
from model.adaptive_pooling import apply_adaptive_pooling_to_linear
from model.obr import apply_obr_to_linear
from data.dataset import load_pubmed_vision
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from qwen_vl_utils import process_vision_info
from tqdm import tqdm  # 添加tqdm导入
import time
from datetime import timedelta


class CalibrationDataset(Dataset):
    """校准数据集包装器"""

    def __init__(self, calib_data):
        self.calib_data = calib_data

    def __len__(self):
        return len(self.calib_data)

    def __getitem__(self, idx):
        sample = self.calib_data[idx]
        # 验证图像文件存在
        valid_image_paths = []
        for p in sample.get("image_paths", []):
            if os.path.exists(p):
                valid_image_paths.append(p)

        return {
            "image_paths": valid_image_paths,
            "input_text": sample.get("input_text", ""),
            "target_text": sample.get("target_text", ""),
            "modality_label": sample.get("modality_label", 0),
            "id": f"sample_{idx}"  # 添加ID用于调试
        }


class OBRTrainer:
    def __init__(self, config: Config):
        self.config = config
        if config.training.use_swanlab:
            swanlab.init(
                project=config.training.swanlab_project,
                name=config.training.swanlab_name
            )

    def calibrate_activations(self, model, processor, calib_data):
        def global_hook(module, input, output):
            logger.info(f"Global hook triggered on {module.__class__.__name__}")

        # 创建 DataLoader
        calib_dataset = CalibrationDataset(calib_data)
        calib_dataloader = DataLoader(
            calib_dataset,
            batch_size=self.config.data.calib_batch_size,
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            collate_fn=lambda x: x  # 自定义collate函数，保持原始数据结构
        )

        logger.info(
            f"Created DataLoader with {len(calib_dataloader)} batches, batch_size={self.config.data.calib_batch_size}")

        # 测试钩子（只测试第一个批次）
        test_hook = model.register_forward_hook(global_hook)
        try:
            first_batch = next(iter(calib_dataloader))
            for sample in first_batch:
                image_paths = sample.get("image_paths", [])
                if not image_paths:
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": p} for p in image_paths] +
                                   [{"type": "text", "text": sample.get("input_text", "")}],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    out = model(**inputs)
                break  # 只测试一个样本
        except Exception as e:
            logger.warning(f"Test hook execution failed: {e}")
        finally:
            test_hook.remove()

        """分别收集文本分支和图像分支的激活值"""
        # 确保缓存目录存在
        os.makedirs(self.config.data.activation_cache_dir, exist_ok=True)
        text_cache_dir = os.path.join(self.config.data.activation_cache_dir, "text")
        vision_cache_dir = os.path.join(self.config.data.activation_cache_dir, "vision")
        os.makedirs(text_cache_dir, exist_ok=True)
        os.makedirs(vision_cache_dir, exist_ok=True)

        # 清理之前的缓存文件
        for cache_dir in [text_cache_dir, vision_cache_dir]:
            for f in os.listdir(cache_dir):
                if f.endswith('.pt'):
                    os.remove(os.path.join(cache_dir, f))

        def hook_fn(module, input, output, name, branch_type):
            """
            钩子收集 Linear 层的输入激活（论文OBR所需）
            """
            if input is None or len(input) == 0 or not torch.is_tensor(input[0]):
                return

            try:
                inp = input[0].detach().cpu()
                out_cpu = process_activation_for_storage(inp)
                if out_cpu is None:
                    return

                cache_dir = text_cache_dir if branch_type == "text" else vision_cache_dir
                layer_path = os.path.join(cache_dir, f"{name}.pt")

                if os.path.exists(layer_path):
                    existing = torch.load(layer_path)
                    if existing.shape[1:] == out_cpu.shape[1:]:
                        combined = torch.cat([existing, out_cpu], dim=0)
                        torch.save(combined, layer_path)
                        del existing
                    else:
                        logger.warning(f"Shape mismatch for {name}: {existing.shape} vs {out_cpu.shape}")
                        mean_existing = existing.mean(dim=0, keepdim=True)
                        mean_new = out_cpu.mean(dim=0, keepdim=True)
                        torch.save((mean_existing + mean_new) / 2, layer_path)
                        del existing
                else:
                    torch.save(out_cpu, layer_path)

            except Exception as e:
                logger.error(f"Failed to save input activations for {name}: {e}")

        hooks = []

        lm = getattr(model, self.config.model.language_model_name)
        from functools import partial
        for name, module in lm.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = partial(hook_fn, name=f"text.{name}", branch_type="text")
                hooks.append(module.register_forward_hook(hook))

        # 为视觉分支注册钩子
        vision_encoder = getattr(model, self.config.model.vision_encoder_name)
        for name, module in vision_encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = partial(hook_fn, name=f"vision.{name}", branch_type="vision")
                hooks.append(module.register_forward_hook(hook))

        logger.info(f"Vision encoder type: {type(vision_encoder)}")
        logger.info(f"Vision encoder device: {next(vision_encoder.parameters()).device}")

        # 使用 DataLoader 执行校准
        calib_size = getattr(self.config.data, "calib_size", 256)
        logger.info(f"开始校准，数据量: {calib_size}，批次大小: {self.config.data.calib_batch_size}")

        processed_samples = 0
        for batch_idx, batch in enumerate(calib_dataloader):
            if processed_samples >= calib_size:
                break

            for sample in batch:
                if processed_samples >= calib_size:
                    break

                valid_image_paths = sample.get("image_paths", [])
                if not valid_image_paths:
                    logger.warning(f"⚠️ No valid images found for sample {sample.get('id', 'unknown')}, skip it.")
                    continue

                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": p} for p in valid_image_paths] +
                                       [{"type": "text", "text": sample.get("input_text", "")}],
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    logger.debug(
                        f"Processed {len(valid_image_paths)} images, sample {processed_samples + 1}/{calib_size}")

                    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}

                    with torch.no_grad():
                        out = model(**inputs)

                    processed_samples += 1

                    # 每处理一些样本输出进度
                    if processed_samples % 50 == 0:
                        logger.info(f"校准进度: {processed_samples}/{calib_size}")
                        # 定期清理GPU缓存防止内存溢出
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error processing sample {processed_samples}: {e}")
                    continue

        # 移除钩子
        for h in hooks:
            h.remove()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 计算平均激活值
        def compute_avg_activations(branch_dir, branch_name):
            avg_activations = {}

            if not os.path.exists(branch_dir):
                logger.warning(f"Cache directory {branch_dir} does not exist")
                return avg_activations

            # 获取所有缓存文件
            cache_files = [f for f in os.listdir(branch_dir) if f.endswith('.pt')]
            logger.info(f"Found {len(cache_files)} cache files for {branch_name}")

            for fname in cache_files:
                layer_name = fname[:-3]  # 移除 .pt 后缀
                cache_path = os.path.join(branch_dir, fname)

                try:
                    # 加载激活值
                    acts = torch.load(cache_path, map_location="cpu")

                    if acts.numel() == 0:
                        logger.warning(f"Empty activation file for {layer_name}")
                        continue

                    # 计算平均值
                    if acts.dim() == 1:
                        avg_activations[layer_name] = acts
                    else:
                        avg_activations[layer_name] = acts.mean(dim=0)

                    logger.debug(
                        f"{branch_name} layer {layer_name}: {acts.shape} -> {avg_activations[layer_name].shape}")

                    # 清理内存
                    del acts

                except Exception as e:
                    logger.error(f"Failed to process {branch_name} activations for {layer_name}: {e}")
                    continue

            return avg_activations

        text_avg = compute_avg_activations(text_cache_dir, "文本分支")
        vision_avg = compute_avg_activations(vision_cache_dir, "视觉分支")

        logger.info(f"校准完成，共处理 {processed_samples} 个样本")
        logger.info(f"文本分支收集了 {len(text_avg)} 层的激活值")
        logger.info(f"视觉分支收集了 {len(vision_avg)} 层的激活值")

        # 可选：清理缓存文件
        if not getattr(self.config.data, 'keep_activation_cache', False):
            for cache_dir in [text_cache_dir, vision_cache_dir]:
                for f in os.listdir(cache_dir):
                    if f.endswith('.pt'):
                        os.remove(os.path.join(cache_dir, f))

        return {"text": text_avg, "vision": vision_avg}

    def compress_model(self):
        model, processor = load_huatuo_vision_model(
            self.config.model.model_name,
            device=self.config.training.device,
            vit_compress_mode=self.config.model.vit_compress_mode,
            vit_target_tokens=self.config.model.vit_target_tokens
        )
        logger.info(f"Model loaded: {self.config.model.model_name}")

        calib_data = load_pubmed_vision(self.config)
        logger.info(f"Calibration data loaded: {len(calib_data)} samples")

        # 如果指定了校准集大小，则截取
        if hasattr(self.config.data, 'calib_size') and self.config.data.calib_size:
            calib_data = calib_data[:self.config.data.calib_size]
            logger.info(f"Using first {len(calib_data)} samples for calibration")

        activations_dict = self.calibrate_activations(model, processor, calib_data)

        text_activations = activations_dict["text"]
        vision_activations = activations_dict["vision"]

        # 文本分支：使用 OBR 压缩
        logger.info("=== 开始压缩文本分支（OBR）===")
        lm = getattr(model, self.config.model.language_model_name)

        # 先统计文本分支的线性层数量
        text_linear_layers = []
        for name, module in lm.named_modules():
            if isinstance(module, torch.nn.Linear):
                text_linear_layers.append((name, module))

        total_text_layers = len(text_linear_layers)
        logger.info(f"文本分支共有 {total_text_layers} 个线性层需要压缩")

        text_layers_processed = 0
        text_layers_failed = 0
        start_time = time.time()

        # 使用tqdm显示进度条
        with tqdm(total=total_text_layers, desc="压缩文本分支", unit="layer") as pbar:
            for idx, (name, module) in enumerate(text_linear_layers, 1):
                full_name = f"text.{name}"
                layer_device = next(module.parameters()).device

                # 计算剩余时间预估
                elapsed_time = time.time() - start_time
                if text_layers_processed > 0:
                    avg_time_per_layer = elapsed_time / text_layers_processed
                    remaining_layers = total_text_layers - idx + 1
                    remaining_time = avg_time_per_layer * remaining_layers
                    remaining_str = str(timedelta(seconds=int(remaining_time)))
                else:
                    remaining_str = "计算中..."

                logger.info(f"[文本层][{name}]开始处理({idx}/{total_text_layers}) | 剩余时间预估：{remaining_str}")

                act = text_activations.get(full_name, None)

                if act is None or act.numel() == 0:
                    logger.error(f"[文本层][{name}]处理失败({idx}/{total_text_layers}) | 原因：无激活值数据")
                    text_layers_failed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": text_layers_processed,
                        "失败": text_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })
                    continue

                # 确保激活值形状正确
                try:
                    if act.dim() == 1:
                        act = act.unsqueeze(0)

                    # 检查激活值形状是否与权重匹配
                    if act.shape[-1] != module.in_features:
                        logger.error(f"[文本层][{name}]处理失败({idx}/{total_text_layers}) | 原因：激活值形状不匹配 {act.shape} vs {module.weight.shape}")
                        text_layers_failed += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "成功": text_layers_processed,
                            "失败": text_layers_failed,
                            "当前层": name[:20] + "..." if len(name) > 20 else name
                        })
                        continue

                    act = act.to(layer_device)

                    # 应用 OBR 压缩
                    success = apply_obr_to_linear(
                        module,
                        act,
                        bits=self.config.compression.lm_bits,
                        sparsity=self.config.compression.lm_sparsity,
                        alpha=self.config.compression.alpha,
                        device=str(layer_device),
                        layer_name=name,
                        layer_idx=idx,
                        total_layers=total_text_layers
                    )

                    if success:
                        text_layers_processed += 1
                        logger.info(f"[文本层][{name}]处理完成({idx}/{total_text_layers})")
                    else:
                        text_layers_failed += 1
                        logger.error(f"[文本层][{name}]处理失败({idx}/{total_text_layers}) | 原因：OBR压缩过程中出错")

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": text_layers_processed,
                        "失败": text_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })

                except Exception as e:
                    logger.error(f"[文本层][{name}]处理失败({idx}/{total_text_layers}) | 原因：{str(e)}")
                    text_layers_failed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": text_layers_processed,
                        "失败": text_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })
                    continue

        # 视觉分支：使用 Adaptive Pooling 压缩
        logger.info("=== 开始压缩视觉分支（Adaptive Pooling）===")
        vision_encoder = getattr(model, self.config.model.vision_encoder_name)

        # 先统计视觉分支的线性层数量
        vision_linear_layers = []
        for name, module in vision_encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                vision_linear_layers.append((name, module))

        total_vision_layers = len(vision_linear_layers)
        logger.info(f"视觉分支共有 {total_vision_layers} 个线性层需要压缩")

        vision_layers_processed = 0
        vision_layers_failed = 0
        vision_start_time = time.time()

        # 使用tqdm显示进度条
        with tqdm(total=total_vision_layers, desc="压缩视觉分支", unit="layer") as pbar:
            for idx, (name, module) in enumerate(vision_linear_layers, 1):
                full_name = f"vision.{name}"
                layer_device = next(module.parameters()).device

                # 计算剩余时间预估
                elapsed_time = time.time() - vision_start_time
                if vision_layers_processed > 0:
                    avg_time_per_layer = elapsed_time / vision_layers_processed
                    remaining_layers = total_vision_layers - idx + 1
                    remaining_time = avg_time_per_layer * remaining_layers
                    remaining_str = str(timedelta(seconds=int(remaining_time)))
                else:
                    remaining_str = "计算中..."

                logger.info(f"[视觉层][{name}]开始处理({idx}/{total_vision_layers}) | 剩余时间预估：{remaining_str}")

                act = vision_activations.get(full_name, None)

                if act is None or act.numel() == 0:
                    logger.error(f"[视觉层][{name}]处理失败({idx}/{total_vision_layers}) | 原因：无激活值数据")
                    vision_layers_failed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": vision_layers_processed,
                        "失败": vision_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })
                    continue

                try:
                    # 确保激活值形状正确
                    if act.dim() == 1:
                        act = act.unsqueeze(0)

                    # 检查激活值形状是否与权重匹配
                    if act.shape[-1] != module.in_features:
                        logger.error(f"[视觉层][{name}]处理失败({idx}/{total_vision_layers}) | 原因：激活值形状不匹配")
                        vision_layers_failed += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            "成功": vision_layers_processed,
                            "失败": vision_layers_failed,
                            "当前层": name[:20] + "..." if len(name) > 20 else name
                        })
                        continue

                    act = act.to(layer_device)

                    # 应用 Adaptive Pooling 压缩
                    success = apply_adaptive_pooling_to_linear(
                        module,
                        act,
                        target_ratio=self.config.compression.vision_pooling_ratio,
                        method=self.config.compression.pooling_method,
                        device=str(layer_device),
                        layer_name=name,
                        layer_idx=idx,
                        total_layers=total_vision_layers
                    )

                    if success:
                        vision_layers_processed += 1
                        logger.info(f"[视觉层][{name}]处理完成({idx}/{total_vision_layers})")
                    else:
                        vision_layers_failed += 1
                        logger.error(
                            f"[视觉层][{name}]处理失败({idx}/{total_vision_layers}) | 原因：Adaptive Pooling压缩过程中出错")

                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": vision_layers_processed,
                        "失败": vision_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })

                except Exception as e:
                    logger.error(f"[视觉层][{name}]处理失败({idx}/{total_vision_layers}) | 原因：{str(e)}")
                    vision_layers_failed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "成功": vision_layers_processed,
                        "失败": vision_layers_failed,
                        "当前层": name[:20] + "..." if len(name) > 20 else name
                    })
                    continue

        total_time = time.time() - start_time
        logger.info(f"=== 模型压缩完成，总耗时: {str(timedelta(seconds=int(total_time)))} ===")
        logger.info(f"文本分支: 成功 {text_layers_processed} 层, 失败 {text_layers_failed} 层")
        logger.info(f"视觉分支: 成功 {vision_layers_processed} 层, 失败 {vision_layers_failed} 层")

        # 计算整体成功率
        total_layers = total_text_layers + total_vision_layers
        total_success = text_layers_processed + vision_layers_processed
        success_rate = (total_success / total_layers) * 100 if total_layers > 0 else 0
        logger.info(f"整体成功率: {success_rate:.1f}% ({total_success}/{total_layers})")

        return model, processor


def process_activation_for_storage(tensor: torch.Tensor) -> torch.Tensor:
    """
    处理激活值以便存储，确保形状一致性
    统一转换为 [batch_size, features] 形状
    """
    try:
        original_shape = tensor.shape
        if tensor.dim() == 1:
            # [H] -> [1, H]
            result = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            # [P, H] -> [1, H] 通过平均序列维度
            result = tensor.mean(dim=0, keepdim=True)
        elif tensor.dim() == 3:
            # [B, P, H] -> [B, H] 通过平均序列维度，保留批次维度
            result = tensor.mean(dim=1, keepdim=False)
            if result.dim() == 1:
                result = result.unsqueeze(0)
        elif tensor.dim() == 4:
            # [B, P, H, W] -> [B, H] 通过平均序列和宽度维度
            result = tensor.mean(dim=(1, 3), keepdim=False)
            if result.dim() == 1:
                result = result.unsqueeze(0)
        else:
            logger.warning(f"Unexpected activation shape {tensor.shape}, using global mean")
            # 对于更高维度的张量，展平后取平均
            flattened = tensor.view(tensor.size(0), -1)
            result = flattened.mean(dim=0, keepdim=True)

        # logger.debug(f"Processed activation: {original_shape} -> {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Error processing activation shape {tensor.shape}: {e}")
        # 返回一个安全的默认值
        return tensor.view(1, -1) if tensor.numel() > 0 else None


def normalize_activation_shape(tensor: torch.Tensor) -> torch.Tensor:
    """
    将任意形状的激活统一变成 [1, H]
    这是 process_activation_for_storage 的简化版本
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        return tensor.mean(dim=0, keepdim=True)
    elif tensor.dim() == 3:
        return tensor.mean(dim=(0, 1), keepdim=True)
    else:
        # 对于更高维度，使用更通用的方法
        flattened = tensor.view(tensor.size(0), -1)
        return flattened.mean(dim=0, keepdim=True)