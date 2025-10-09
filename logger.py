import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(
    name="OBR-Med",
    level=logging.DEBUG,
    log_dir="./logs",
    when="midnight",  # 每天0点分割
    backup_count=30   # 保留最近30天日志
):
    """
    创建支持控制台输出 + 每日滚动文件日志的 logger。
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 防止重复添加 handler
        return logger

    logger.setLevel(level)

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === 控制台输出 ===
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # === 按日期分割的文件输出 ===
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when=when,          # 时间间隔类型
        interval=1,         # 每1个周期分割
        backupCount=backup_count,  # 保留的旧日志数量
        encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"  # 自动生成文件名后缀，例如 OBR-Med.log.2025-10-09
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
