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
        text_activations = {}
        vision_activations = {}

        def hook_fn(module, input, output, name, branch_type):
            if output is None or not torch.is_tensor(output):
                return
            target_dict = text_activations if branch_type == "text" else vision_activations
            target_dict.setdefault(name, []).append(output.detach().cpu())

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

                except Exception as e:
                    logger.error(f"Error processing sample {processed_samples}: {e}")
                    continue

        # 移除钩子
        for h in hooks:
            h.remove()

        # 计算平均激活值
        def compute_avg_activations(activation_dict, branch_name):
            logger.info(f"{branch_name} 激活层数量：{len(activation_dict)}")
            for k, v in activation_dict.items():
                logger.info(f"{k}: {len(v)} batches, shape {v[0].shape}")
            avg_activations = {}
            for name, acts in activation_dict.items():
                try:
                    cat_acts = torch.cat(acts, dim=0)
                    avg_activations[name] = cat_acts.mean(dim=0)
                except Exception as e:
                    logger.warning(f"Failed to average {branch_name} activations for {name}: {e}")
            logger.info(f"{branch_name} 平均激活层数量：{len(avg_activations)}")
            return avg_activations

        text_avg = compute_avg_activations(text_activations, "文本分支")
        vision_avg = compute_avg_activations(vision_activations, "视觉分支")

        logger.info(f"校准完成，共处理 {processed_samples} 个样本")
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

        fallback_batch = self.config.data.calib_batch_size

        # 文本分支：使用 OBR 压缩
        logger.info("=== 开始压缩文本分支（OBR）===")
        lm = getattr(model, self.config.model.language_model_name)
        for name, module in lm.named_modules():
            if isinstance(module, torch.nn.Linear):
                act = text_activations.get(name, None)
                layer_device = next(module.parameters()).device

                if act is None or act.numel() == 0:
                    act = torch.randn(fallback_batch, module.in_features, device=layer_device)
                    logger.warning(f"No text activations for layer {name}; using synthetic activations {act.shape}")
                else:
                    act = act.unsqueeze(0).to(layer_device) if act.dim() == 1 else act.to(layer_device)

                # 应用 OBR 压缩
                apply_obr_to_linear(
                    module,
                    act,
                    bits=self.config.compression.lm_bits,
                    sparsity=self.config.compression.lm_sparsity,
                    alpha=self.config.compression.alpha,
                    device=str(layer_device)
                )
                logger.info(f"文本层 {name} - OBR压缩完成")

        # 视觉分支：使用 Adaptive Pooling 压缩
        logger.info("=== 开始压缩视觉分支（Adaptive Pooling）===")
        vision_encoder = getattr(model, self.config.model.vision_encoder_name)
        for name, module in vision_encoder.named_modules():
            if isinstance(module, torch.nn.Linear):
                act = vision_activations.get(name, None)
                layer_device = next(module.parameters()).device

                if act is None or act.numel() == 0:
                    act = torch.randn(fallback_batch, module.in_features, device=layer_device)
                    logger.warning(f"No vision activations for layer {name}; using synthetic activations {act.shape}")
                else:
                    act = act.unsqueeze(0).to(layer_device) if act.dim() == 1 else act.to(layer_device)

                # 应用 Adaptive Pooling 压缩
                apply_adaptive_pooling_to_linear(
                    module,
                    act,
                    target_ratio=self.config.compression.vision_pooling_ratio,
                    method=self.config.compression.pooling_method,
                    device=str(layer_device)
                )
                logger.info(f"视觉层 {name} - Adaptive Pooling压缩完成")

        logger.info("=== 模型压缩完成 ===")
        return model, processor


# 保留原有的 find_vision_encoder 方法
def find_vision_encoder(self, model):
    """自动检测视觉编码器名称"""
    possible_names = [
        'vision_tower', 'vision_model', 'visual',
        'vision_encoder', 'vit', 'model.vision_tower'
    ]

    for name in possible_names:
        if '.' in name:
            parts = name.split('.')
            obj = model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return name
            except AttributeError:
                continue
        elif hasattr(model, name):
            return name

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and any(
                keyword in name.lower() for keyword in ['vision', 'visual', 'vit', 'encoder']):
            if 'text' not in name.lower() and 'language' not in name.lower():
                logger.warning(f"自动选择视觉编码器: {name}")
                return name.split('.')[-1]

    raise AttributeError("无法找到视觉编码器，请检查模型结构")