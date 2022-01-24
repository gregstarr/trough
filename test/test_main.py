import subprocess
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import pytest


@pytest.fixture(scope='module')
def preserve_config_pointer():
    cfg_ptr = Path(__file__).parent.parent / "trough" / "config_path.txt"
    config_path = None
    if cfg_ptr.exists():
        config_path = cfg_ptr.read_text()
    yield
    if config_path is not None:
        with open(cfg_ptr, 'w') as f:
            f.write(config_path)


def test_default_main(preserve_config_pointer):
    with TemporaryDirectory() as base_dir:
        config_options = {
            'base_dir': base_dir,
            'start_date': '20200908_090000',
            'end_date': '20200909_120000',
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
            print(proc1)
            subtest_fn = Path(__file__).parent / "subtest.py"
            proc2 = subprocess.run(['python', str(subtest_fn.absolute())], capture_output=True)
            print(proc2)
            outputs = [s.decode() for s in proc2.stdout.splitlines()]
            assert outputs[0] == '(28, 60, 180)'
            assert outputs[1] == '(28,)'
            assert outputs[2] == '(28, 60, 180)'
            assert float(outputs[3]) < .5
            assert float(outputs[4]) > 0
        finally:
            cfg_ptr = Path(__file__).parent.parent / "trough" / "config_path.txt"
            cfg_ptr.unlink(missing_ok=True)
            assert not cfg_ptr.exists()
