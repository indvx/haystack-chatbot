import logging
import os

# ------------------------------------------------------------
# Centralized Logger for FastAPI app
# Prevents reinitialization during auto-reload
# ------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Avoid multiple handler setup on reload
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

# Silence watchfiles reload spam
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

# Application logger
logger = logging.getLogger("app_logger")
logger.info(" Logger initialized successfully.")
