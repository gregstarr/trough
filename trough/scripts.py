import logging
from pathlib import Path
from datetime import datetime, timedelta

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
    if not config.keep_download:
        for path in Path(config.download_tec_dir).glob("*.hdf5"):
            date = _tec._parse_madrigal_fn(path)
            if start_date <= date <= end_date:
                path.unlink()


def process_arb(start_date, end_date):
    logger.info(f"running 'process_arb'")
    _arb.process_auroral_boundary_dataset(start_date, end_date)
    if not config.keep_download:
        for path in Path(config.download_arb_dir).glob("*.NC"):
            sat_name, date = _arb._parse_arb_fn(path)
            if start_date <= date <= end_date:
                path.unlink()


def process_omni(start_date, end_date):
    logger.info(f"running 'process_omni'")
    _omni.process_omni_dataset(config.download_omni_dir, Path(config.processed_omni_file))


def process_all(start_date, end_date):
    logger.info(f"running 'process_all'")
    process_tec(start_date, end_date)
    process_arb(start_date, end_date)
    process_omni(start_date, end_date)


def label_trough(start_date, end_date):
    logger.info(f"running 'label_trough'")
    _trough.label_trough_dataset(start_date, end_date)


def _tec_interval_check(start, end):
    tec_file_names = _tec.get_tec_paths(start, end, config.processed_tec_dir)
    for file in tec_file_names:
        needs_processing = _tec.check_processed_data_interval(start, end, file)
        if needs_processing:
            return True
    return False


def _arb_interval_check(start, end):
    arb_file_names = _arb.get_arb_paths(start, end, config.processed_arb_dir)
    for file in arb_file_names:
        needs_processing = _arb.check_processed_data_interval(start, end, file)
        if needs_processing:
            return True
    return False


def full_run(start_date, end_date):
    logger.info(f"running 'full_run'")
    for year in range(start_date.year, end_date.year + 1):
        start = max(start_date, datetime(year, 1, 1))
        end = min(end_date, datetime(year + 1, 1, 1))
        if end - start <= timedelta(hours=1):
            continue

        if _tec_interval_check(start, end):
            download_tec(start, end)
            process_tec(start, end)
        if _arb_interval_check(start, end):
            download_arb(start, end)
            process_arb(start, end)

    download_omni(start_date, end_date)
    process_omni(start_date, end_date)

    label_trough(start_date, end_date)
