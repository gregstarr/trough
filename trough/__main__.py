import argparse
from datetime import datetime
import logging

import trough


DESCRIPTION = """Main Ionospheric Trough Python Library main script
By Greg Starr

Use to download data, process data and identify trough"""


logger = logging.getLogger(__name__)


def setup(args):
    logging.basicConfig(format='%(asctime)-15s %(name)-20s %(levelname)-8s %(message)s', datefmt='%Y%m%d_%H%M%S',
                        level=logging.INFO)
    trough.config.load_json(args.config)
    trough.config.save(args.config_save)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("config", type=str, help="config file input")
    parser.add_argument("--config-save", type=str, help="config file save file")
    args = parser.parse_args()
    script_fun = getattr(trough.scripts, trough.config.script_name)
    setup(args)
    script_fun(trough.config.start_date, trough.config.end_date)


if __name__ == "__main__":
    main()
