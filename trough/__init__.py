from ._config import Config, InvalidConfiguration

config = Config()

from . import scripts
from ._aux_data import get_omni_data, get_arb_data
