"""Centralized logger for SmartBI Pro."""
import os
from loguru import logger
import yaml
from pathlib import Path

def _load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}

_cfg = _load_config()
_log_cfg = _cfg.get("logging", {})
_level = _log_cfg.get("level", "INFO")
_log_file = _log_cfg.get("log_file", "outputs/smartbi.log")

# Ensure output dir exists
os.makedirs(os.path.dirname(_log_file), exist_ok=True)

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
)
logger.add(
    sink=_log_file,
    level=_level,
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
)

def get_logger(name: str):
    return logger.bind(module=name)

__all__ = ["get_logger", "logger"]