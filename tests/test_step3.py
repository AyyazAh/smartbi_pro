"""Verify config + logger work."""
from utils.logger import get_logger
import yaml
from pathlib import Path

log = get_logger("test_step3")

# Test 1: config loads
cfg_path = Path("config.yaml")
assert cfg_path.exists(), "config.yaml missing!"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

log.info(f"Config loaded: {list(cfg.keys())}")
assert "forecasting" in cfg, "forecasting section missing"
assert cfg["forecasting"]["periods"] == 6

# Test 2: logger works
log.info("Logger test — INFO level ✅")
log.warning("Logger test — WARNING level ⚠️")
log.error("Logger test — ERROR level ❌ (ye expected hai, ignore karo)")

print("\n🎉 STEP 3 PASSED — Config + Logger working!")