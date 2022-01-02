"""
use cases and features:
    - maintain last config in env trough dir which is loaded by default
    - directories are set to defaults using appdirs
    - downloading data to specific directory sets env config
    - can temporarily use alternative configs with context manager
    - all directories can be set with base dir or each can be set individually
"""
import dataclasses
from pathlib import Path
import appdirs
import json
import contextlib


class InvalidConfiguration(Exception):
    ...


@dataclasses.dataclass
class TroughIdParams:
    bg_est_shape: tuple[int] = (1, 19, 17)
    model_weight_max: float = 15
    rbf_bw: int = 1
    tv_hw: float = 2
    tv_vw: float = 1
    l2_weight: float = .06
    tv_weight: float = .15
    perimeter_th: float = 40
    area_th: float = 40
    threshold: float = 0.7
    closing_rad: float = 0

    def dict(self):
        return dataclasses.asdict(self)


def _get_default_directory_structure(base_dir):
    base = Path(base_dir)
    download_base = base / 'download'
    processed_base = base / 'processed'
    download_tec_dir = download_base / 'tec'
    download_arb_dir = download_base / 'arb'
    download_omni_dir = download_base / 'omni'
    processed_tec_dir = processed_base / 'tec'
    processed_arb_dir = processed_base / 'arb'
    processed_omni_dir = processed_base / 'omni'
    processed_labels_dir = processed_base / 'labels'
    return {
        'download_tec_dir': str(download_tec_dir),
        'download_arb_dir': str(download_arb_dir),
        'download_omni_dir': str(download_omni_dir),
        'processed_tec_dir': str(processed_tec_dir),
        'processed_arb_dir': str(processed_arb_dir),
        'processed_omni_dir': str(processed_omni_dir),
        'processed_labels_dir': str(processed_labels_dir),
    }


class Config:

    def __init__(self):
        trough_dirs = appdirs.AppDirs(appname='trough')
        default_dirs = _get_default_directory_structure(trough_dirs.user_data_dir)
        self.download_tec_dir = default_dirs['download_tec_dir']
        self.download_arb_dir = default_dirs['download_arb_dir']
        self.download_omni_dir = default_dirs['download_omni_dir']
        self.processed_tec_dir = default_dirs['processed_tec_dir']
        self.processed_arb_dir = default_dirs['processed_arb_dir']
        self.processed_omni_dir = default_dirs['processed_omni_dir']
        self.processed_labels_dir = default_dirs['processed_labels_dir']
        self.trough_id_params = TroughIdParams()
        self.madrigal_user_name = None
        self.madrigal_user_email = None
        self.madrigal_user_affil = None

        self.config_path = Path(__file__).parent / 'trough.json'

    def load_json(self, config_path):
        with open(config_path) as f:
            params = json.load(f)
        self.load_dict(params)

    def load_dict(self, config_dict):
        self.download_tec_dir = config_dict.get('download_tec_dir', self.download_tec_dir)
        self.download_arb_dir = config_dict.get('download_arb_dir', self.download_arb_dir)
        self.download_omni_dir = config_dict.get('download_omni_dir', self.download_omni_dir)
        self.processed_tec_dir = config_dict.get('processed_tec_dir', self.processed_tec_dir)
        self.processed_arb_dir = config_dict.get('processed_arb_dir', self.processed_arb_dir)
        self.processed_omni_dir = config_dict.get('processed_omni_dir', self.processed_omni_dir)
        self.processed_labels_dir = config_dict.get('processed_labels_dir', self.processed_labels_dir)
        self.trough_id_params = config_dict.get('trough_id_params', self.trough_id_params)
        if not isinstance(self.trough_id_params, TroughIdParams):
            self.trough_id_params = TroughIdParams(**self.trough_id_params)
        self.madrigal_user_name = config_dict.get('madrigal_user_name', self.madrigal_user_name)
        self.madrigal_user_email = config_dict.get('madrigal_user_email', self.madrigal_user_email)
        self.madrigal_user_affil = config_dict.get('madrigal_user_affil', self.madrigal_user_affil)

    def save(self):
        save_dict = self.__dict__.copy()
        del save_dict['config_path']
        save_dict['trough_id_params'] = self.trough_id_params.dict()
        with open(self.config_path, 'w') as f:
            json.dump(save_dict, f)

    def set_base_dir(self, base_dir):
        data_dirs = _get_default_directory_structure(base_dir)
        self.load_dict(data_dirs)

    def dict(self):
        param_dict = self.__dict__.copy()
        param_dict['trough_id_params'] = self.trough_id_params.dict()
        return param_dict

    @contextlib.contextmanager
    def temp_config(self, **kwargs):
        original_params = self.dict().copy()
        new_params = original_params.copy()
        new_params.update(**kwargs)
        if 'base_dir' in kwargs:
            new_params.update(**_get_default_directory_structure(kwargs['base_dir']))
        try:
            self.load_dict(new_params)
            yield self
        finally:
            self.load_dict(original_params)
