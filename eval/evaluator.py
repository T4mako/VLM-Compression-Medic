# eval/evaluator.py

import torch
from PIL import Image
from transformers import AutoTokenizer
from logger import logger
from config import Config

class PerplexityEvaluator:
    def __init__(self, model, processor, config: Config):
        self.model = model
        self.processor = processor
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            trust_remote_code=True,
            local_files_only=True
        )

    def evaluate(self, eval_data):
        total_loss = 0.0
        total_tokens = 0
        sample_count = 0

        for sample in eval_data:
            try:
                # 加载图像
                images = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
                
                # 构造完整文本（输入 + 目标）
                full_text = sample["input_text"] + " " + sample["target_text"]
                
                # 使用 processor 编码
                inputs = self.processor(
                    images=images,
                    text=full_text,
                    return_tensors="pt"
                ).to(self.model.device)

                # === 关键：如果模型有 vit_compressor，则替换 vision_model 的 forward ===
                if hasattr(self.model, 'vit_compressor'):
                    # 临时 hook：替换 vision_model 的输出
                    def hooked_vision_forward(pixel_values):
                        with torch.no_grad():
                            # 原始 ViT 输出: [B, N, C]
                            raw_features = self.model.vision_model(pixel_values)
                            # 压缩: [B, M, C]
                            compressed = self.model.vit_compressor(raw_features)
                            return compressed

                    # 替换 forward
                    original_vision_forward = self.model.vision_model.forward
                    self.model.vision_model.forward = hooked_vision_forward

                    # 推理
                    with torch.no_grad():
                        outputs = self.model(**inputs, labels=inputs["input_ids"])

                    # 恢复原始 forward
                    self.model.vision_model.forward = original_vision_forward

                else:
                    # 无压缩，正常推理
                    with torch.no_grad():
                        outputs = self.model(**inputs, labels=inputs["input_ids"])

                # 累计 loss
                loss = outputs.loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    n_tokens = inputs["input_ids"].numel()
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens
                    sample_count += 1

            except Exception as e:
                logger.warning(f"Skip sample due to error: {e}")
                continue

        if total_tokens == 0:
            logger.error("No valid samples for evaluation!")
            return {"perplexity": float('inf')}

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logger.info(f"Evaluated {sample_count} samples. Perplexity: {perplexity:.2f}")
        return {"perplexity": perplexity}