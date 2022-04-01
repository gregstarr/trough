import dataclasses
from pathlib import Path
import appdirs
import json
import contextlib
import typing
import numpy as np
from datetime import datetime


@dataclasses.dataclass
class TroughIdParams:
    bg_est_shape: typing.Tuple[int, int, int] = (1, 19, 17)
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
    processed_omni_file = processed_base / 'omni.nc'
    processed_labels_dir = processed_base / 'labels'
    return {
        'download_tec_dir': str(download_tec_dir),
        'download_arb_dir': str(download_arb_dir),
        'download_omni_dir': str(download_omni_dir),
        'processed_tec_dir': str(processed_tec_dir),
        'processed_arb_dir': str(processed_arb_dir),
        'processed_omni_file': str(processed_omni_file),
        'processed_labels_dir': str(processed_labels_dir),
    }


trough_dirs = appdirs.AppDirs(appname='trough')


def parse_date(date_str):
    if len(date_str) == 15:
        return datetime.strptime(date_str, '%Y%m%d_%H%M%S')
    if len(date_str) == 13:
        return datetime.strptime(date_str, '%Y%m%d_%H%M')
    if len(date_str) == 11:
        return datetime.strptime(date_str, '%Y%m%d_%H')
    if len(date_str) == 8:
        return datetime.strptime(date_str, '%Y%m%d')
    raise ValueError(f"Invalid date string: {date_str}")


class Config:

    def __init__(self, config_path=None):
        default_dirs = _get_default_directory_structure(trough_dirs.user_data_dir)
        self.download_tec_dir = default_dirs['download_tec_dir']
        self.download_arb_dir = default_dirs['download_arb_dir']
        self.download_omni_dir = default_dirs['download_omni_dir']
        self.processed_tec_dir = default_dirs['processed_tec_dir']
        self.processed_arb_dir = default_dirs['processed_arb_dir']
        self.processed_omni_file = default_dirs['processed_omni_file']
        self.processed_labels_dir = default_dirs['processed_labels_dir']
        self.trough_id_params = TroughIdParams()
        self.madrigal_user_name = None
        self.madrigal_user_email = None
        self.madrigal_user_affil = None
        self.nasa_spdf_download_method = 'ftp'

        self.lat_res = 1
        self.lon_res = 2
        self.time_res_unit = 'h'
        self.time_res_n = 1
        self.mlat_min = 30

        self.script_name = 'full_run'
        self.start_date = None
        self.end_date = None

        self.keep_download = False

        if config_path is not None:
            self.load_json(config_path)

    def get_config_name(self):
        cfg = self.dict()
        return f"{cfg['script_name']}_{cfg['start_date']}_{cfg['end_date']}_config.json"

    def get_mlat_bins(self):
        return np.arange(self.mlat_min - self.lat_res / 2, 90, self.lat_res)

    def get_mlat_vals(self):
        return np.arange(self.mlat_min, 90, self.lat_res)

    def get_mlt_bins(self):
        return np.arange(-12, 12 + 24 / 360, self.lon_res * 24 / 360)

    def get_mlt_vals(self):
        return np.arange(-12 + .5 * self.lon_res * 24 / 360, 12 + 24 / 360, self.lon_res * 24 / 360)

    def get_sample_dt(self):
        return np.timedelta64(self.time_res_n, self.time_res_unit)

    def load_json(self, config_path):
        with open(config_path) as f:
            params = json.load(f)
        if 'base_dir' in params:
            params.update(**_get_default_directory_structure(params['base_dir']))
        self.load_dict(params)

    def load_dict(self, config_dict):
        self.download_tec_dir = config_dict.get('download_tec_dir', self.download_tec_dir)
        self.download_arb_dir = config_dict.get('download_arb_dir', self.download_arb_dir)
        self.download_omni_dir = config_dict.get('download_omni_dir', self.download_omni_dir)
        self.processed_tec_dir = config_dict.get('processed_tec_dir', self.processed_tec_dir)
        self.processed_arb_dir = config_dict.get('processed_arb_dir', self.processed_arb_dir)
        self.processed_omni_file = config_dict.get('processed_omni_file', self.processed_omni_file)
        self.processed_labels_dir = config_dict.get('processed_labels_dir', self.processed_labels_dir)
        self.trough_id_params = config_dict.get('trough_id_params', self.trough_id_params)
        if not isinstance(self.trough_id_params, TroughIdParams):
            self.trough_id_params = TroughIdParams(**self.trough_id_params)
        self.madrigal_user_name = config_dict.get('madrigal_user_name', self.madrigal_user_name)
        self.madrigal_user_email = config_dict.get('madrigal_user_email', self.madrigal_user_email)
        self.madrigal_user_affil = config_dict.get('madrigal_user_affil', self.madrigal_user_affil)
        self.nasa_spdf_download_method = config_dict.get('nasa_spdf_download_method', self.nasa_spdf_download_method)
        self.lat_res = config_dict.get('lat_res', self.lat_res)
        self.lon_res = config_dict.get('lon_res', self.lon_res)
        self.time_res_unit = config_dict.get('time_res_unit', self.time_res_unit)
        self.time_res_n = config_dict.get('time_res_n', self.time_res_n)
        self.start_date = config_dict.get('start_date', self.start_date)
        if isinstance(self.start_date, str):
            self.start_date = parse_date(self.start_date)
        self.end_date = config_dict.get('end_date', self.end_date)
        if isinstance(self.end_date, str):
            self.end_date = parse_date(self.end_date)
        self.script_name = config_dict.get('script_name', self.script_name)
        self.keep_download = config_dict.get('keep_download', self.keep_download)

    def save(self, config_path=None):
        if config_path is None:
            config_path = Path(trough_dirs.user_config_dir) / self.get_config_name()
        save_dict = self.dict().copy()
        Path(config_path).parent.mkdir(exist_ok=True, parents=True)
        with open(config_path, 'w') as f:
            json.dump(save_dict, f)
        cfg_pointer = Path(__file__).parent / "config_path.txt"
        cfg_pointer.write_text(str(config_path))
        print(f"Saved config and setting default: {config_path}")

    def set_base_dir(self, base_dir):
        data_dirs = _get_default_directory_structure(base_dir)
        self.load_dict(data_dirs)

    def dict(self):
        param_dict = self.__dict__.copy()
        param_dict['trough_id_params'] = self.trough_id_params.dict()
        if param_dict['start_date'] is not None:
            param_dict['start_date'] = param_dict['start_date'].strftime('%Y%m%d')
        if param_dict['end_date'] is not None:
            param_dict['end_date'] = param_dict['end_date'].strftime('%Y%m%d')
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
