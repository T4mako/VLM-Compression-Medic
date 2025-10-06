from config import Config
from train.trainer import OBRTrainer
from eval.evaluator import PerplexityEvaluator
from data.dataset import load_pubmed_vision
from model.load_model import load_huatuo_vision_model
import argparse
import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["compress", "eval"], default="compress")
    args = parser.parse_args()

    config = Config.from_args()

    if args.mode == "compress":
        trainer = OBRTrainer(config)

        # 修改：现在返回 model 和 processor
        compressed_model, processor = trainer.compress_model()
        # Save model
        compressed_model.save_pretrained(f"{config.training.output_dir}/compressed")
        processor.save_pretrained(f"{config.training.output_dir}/compressed")

    elif args.mode == "eval":
        model, processor = load_huatuo_vision_model(f"{config.training.output_dir}/compressed")
        eval_data = load_pubmed_vision(config)
        evaluator = PerplexityEvaluator(model, processor, config)
        results = evaluator.evaluate(eval_data)
        print("Evaluation Results:", results)


if __name__ == "__main__":
    main()