import subprocess
import json
from tempfile import TemporaryDirectory
from pathlib import Path


def test_default_main():
    with TemporaryDirectory() as base_dir:
        config_options = {
            'base_dir': base_dir,
            'start_date': '20200508_090000',
            'end_date': '20200509_120000',
            'madrigal_user_name': 'gstarr',
            'madrigal_user_email': 'gstarr@bu.edu',
            'madrigal_user_affil': 'bu',
            'nasa_spdf_download_method': 'http',
        }
        input_config_path = Path(base_dir) / "input_config.json"
        output_config_path = Path(base_dir) / "output_config.json"
        with open(input_config_path, 'w') as f:
            json.dump(config_options, f)
        subprocess.run(['python', '-m', 'trough', str(input_config_path.absolute()), '--config-save',
                        str(output_config_path.absolute())])
        print()
