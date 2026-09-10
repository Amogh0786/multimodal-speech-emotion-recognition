import sys
from loguru import logger

def setup_logger(log_file="logs/training.log"):
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}", level="INFO")
    logger.add(log_file, rotation="10 MB", level="DEBUG")
    return logger
