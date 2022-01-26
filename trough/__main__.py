import argparse
import logging
from sys import stdout

from trough import config, scripts


DESCRIPTION = """Main Ionospheric Trough Python Library main script
By Greg Starr

Use to download data, process data and identify trough"""


logger = logging.getLogger(__name__)


def setup(args):
    stream_handler = logging.StreamHandler(stdout)
    file_handler = logging.FileHandler('log_output.log', mode='w')
    logging.basicConfig(format='%(asctime)-19s %(name)-20s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, handlers=[stream_handler, file_handler])
    config.load_json(args.config)


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("config", type=str, help="config file input")
    parser.add_argument("--config-save", type=str, help="config file save file")
    args = parser.parse_args()
    setup(args)
    script_fun = getattr(scripts, config.script_name)
    script_fun(config.start_date, config.end_date)
    config.save(args.config_save)


if __name__ == "__main__":
    main()
