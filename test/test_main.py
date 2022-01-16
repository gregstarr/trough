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
        }
        input_config_path = Path(base_dir) / "input_config.json"
        output_config_path = Path(base_dir) / "output_config.json"
        with open(input_config_path, 'w') as f:
            json.dump(config_options, f)
        try:
            proc1 = subprocess.run(['python', '-m', 'trough', str(input_config_path.absolute()), '--config-save',
                                    str(output_config_path.absolute())], capture_output=True)
            subtest_fn = Path(__file__).parent / "subtest.py"
            proc2 = subprocess.run(['python', str(subtest_fn.absolute())], capture_output=True)
            outputs = [s.decode() for s in proc2.stdout.splitlines()]
            assert outputs[0] == '(28, 60, 180)'
            assert outputs[1] == '(28,)'
            assert outputs[2] == '(28, 60, 180)'
            assert float(outputs[3]) < .5
            assert float(outputs[4]) > 0
        finally:
            subtest_fn = Path(__file__).parent.parent / "trough" / "config_path.txt"
            subtest_fn.unlink(missing_ok=True)
            assert not subtest_fn.exists()
