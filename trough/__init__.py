import logging
from pathlib import Path
from ._config import Config

logger = logging.getLogger(__name__)

config = Config()
_config_path_file = Path(__file__).parent / "config_path.txt"
if _config_path_file.exists():
    _config_path = _config_path_file.read_text()
    config.load_json(_config_path)
    logger.info(f"Loading config: {_config_path}")
else:
    logger.info("no previous config found, using default config")

from ._tec import get_tec_data  # noqa: E402
from ._trough import get_trough_labels, get_data  # noqa: E402
from . import utils  # noqa: E402

__all__ = ['config', 'get_tec_data', 'get_trough_labels', 'get_data', 'utils']
