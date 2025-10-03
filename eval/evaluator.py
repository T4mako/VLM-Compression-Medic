import torch
from transformers import AutoTokenizer
from ..logger import logger
from ..config import Config

class PerplexityEvaluator:
    def __init__(self, model, processor, config: Config):
        self.model = model
        self.processor = processor
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, trust_remote_code=True)

    def evaluate(self, eval_data):
        total_loss = 0.0
        total_tokens = 0

        for sample in eval_data:
            images = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
            full_text = sample["input_text"] + " " + sample["target_text"]
            inputs = self.processor(images, full_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        logger.info(f"Perplexity: {perplexity:.2f}")
        return {"perplexity": perplexity}