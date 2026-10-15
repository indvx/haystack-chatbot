import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from decouple import config

LOG_DIR = Path(config("LOG_DIR", default="./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Keep logging setup in one place so reloads don't add duplicate handlers.
logger = logging.getLogger("haystack-chatbot")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logging.getLogger("watchfiles").setLevel(logging.WARNING)
