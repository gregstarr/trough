import logging
from pathlib import Path

from trough import config
from trough import _download, _tec, _arb, _omni, _trough


logger = logging.getLogger(__name__)


def download_tec(start_date, end_date):
    user_data = [config.madrigal_user_name, config.madrigal_user_email, config.madrigal_user_affil]
    logger.info(f"running 'download_tec', start date: {start_date}, end date: {end_date}, user data: {user_data}")
    downloader = _download.MadrigalTecDownloader(config.download_tec_dir, *user_data)
    downloader.download(start_date, end_date)
    logger.info("'download_tec' completed")


def download_arb(start_date, end_date):
    logger.info(f"running 'download_arb', start date: {start_date}, end date: {end_date}")
    downloader = _download.ArbDownloader(config.download_arb_dir)
    downloader.download(start_date, end_date)
    logger.info("'download_arb' completed")


def download_omni(start_date, end_date):
    logger.info(f"running 'download_omni', start date: {start_date}, end date: {end_date}")
    downloader = _download.OmniDownloader(config.download_omni_dir, config.nasa_spdf_download_method)
    downloader.download(start_date, end_date)
    logger.info("'download_omni' completed")


def download_all(start_date, end_date):
    download_tec(start_date, end_date)
    download_arb(start_date, end_date)
    download_omni(start_date, end_date)


def process_tec(start_date, end_date):
    logger.info(f"running 'process_tec'")
    _tec.process_tec_dataset(start_date, end_date)


def process_arb(start_date, end_date):
    logger.info(f"running 'process_arb'")
    _arb.process_auroral_boundary_dataset(start_date, end_date)


def process_omni():
    logger.info(f"running 'process_omni'")
    _omni.process_omni_dataset(config.download_omni_dir, Path(config.processed_omni_file))


def process_all(start_date, end_date):
    logger.info(f"running 'process_all'")
    process_tec(start_date, end_date)
    process_arb(start_date, end_date)
    process_omni()


def label_trough(start_date, end_date):
    logger.info(f"running 'label_trough'")
    _trough.label_trough_dataset(start_date, end_date)


def full_run(start_date, end_date):
    logger.info(f"running 'full_run'")
    download_all(start_date, end_date)
    process_all(start_date, end_date)
    label_trough(start_date, end_date)
