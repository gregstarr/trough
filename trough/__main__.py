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
    if args.config is not None:
        trough.config.load_json(args.config)
    trough.config.madrigal_user_name = args.user_name
    trough.config.madrigal_user_email = args.user_email
    trough.config.madrigal_user_affil = args.user_affil
    trough.config.save()
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d')
    return start_date, end_date


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("script", type=str, help="which script to run")
    parser.add_argument("start_date", type=str, help="start date YYYYMMDD")
    parser.add_argument("end_date", type=str, help="end date YYYYMMDD")
    parser.add_argument("-c", "--config", type=str, help="config file")
    parser.add_argument("--user-name", type=str, help="madrigal user name")
    parser.add_argument("--user-email", type=str, help="madrigal user email")
    parser.add_argument("--user-affil", type=str, help="madrigal user affiliation")
    args = parser.parse_args()
    script_fun = getattr(trough.scripts, args.script)
    start_date, end_date = setup(args)
    script_fun(start_date, end_date)


if __name__ == "__main__":
    main()
