import importlib.metadata

try:
    __version__ = importlib.metadata.version("assignment1-basics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # 或其他默认版本号


import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
