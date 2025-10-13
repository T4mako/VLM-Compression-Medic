from config import Config
from train.trainer import OBRTrainer
from eval.evaluator import PerplexityEvaluator
from data.dataset import load_pubmed_vision
from model.load_model import load_huatuo_vision_model
import logger


def main():
    # 统一从命令行解析所有参数
    config = Config.from_args()

    if config.mode == "compress":
        trainer = OBRTrainer(config)

        # 压缩模型
        compressed_model, processor = trainer.compress_model()

        # 保存压缩后的模型
        output_dir = f"{config.training.output_dir}/compressed"
        compressed_model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        logger.logger.info(f"模型已保存到 {output_dir}")

    elif config.mode == "eval":
        # 加载压缩模型
        model, processor = load_huatuo_vision_model(f"{config.training.output_dir}/compressed")
        eval_data = load_pubmed_vision(config)

        evaluator = PerplexityEvaluator(model, processor, config)
        results = evaluator.evaluate(eval_data)
        print("Evaluation Results:", results)


if __name__ == "__main__":
    main()
