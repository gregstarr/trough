import logging

from trough import config, InvalidConfiguration
import trough._download as trough_download
import trough._aux_data as trough_prepare
import trough._tec as trough_tec
import trough._trough as trough_label


logger = logging.getLogger(__name__)


def download_tec(start_date, end_date):
    user_data = [config.madrigal_user_name, config.madrigal_user_email, config.madrigal_user_affil]
    logger.info(f"running 'download_tec', start date: {start_date}, end date: {end_date}, user data: {user_data}")
    if None in user_data:
        raise InvalidConfiguration("To download from Madrigal, user name, email, and affiliation must be specified")
    downloader = trough_download.MadrigalTecDownloader(config.download_tec_dir, *user_data)
    downloader.download(start_date, end_date)
    logger.info("'download_tec' completed")


def download_arb(start_date, end_date):
    logger.info(f"running 'download_arb', start date: {start_date}, end date: {end_date}")
    if config.nasa_spdf_download_method == 'ftp':
        downloader = trough_download.AuroralBoundaryFtpDownloader(config.download_arb_dir)
    elif config.nasa_spdf_download_method == 'http':
        downloader = trough_download.AuroralBoundaryHttpDownloader(config.download_arb_dir)
    else:
        raise InvalidConfiguration(f"nasa spdf download method (given: {config.nasa_spdf_download_method}) "
                                   f"must be 'ftp' or 'http'")
    downloader.download(start_date, end_date)
    logger.info("'download_arb' completed")


def download_omni(start_date, end_date):
    logger.info(f"running 'download_omni', start date: {start_date}, end date: {end_date}")
    if config.nasa_spdf_download_method == 'ftp':
        downloader = trough_download.OmniFtpDownloader(config.download_omni_dir)
    elif config.nasa_spdf_download_method == 'http':
        downloader = trough_download.OmniHttpDownloader(config.download_omni_dir)
    else:
        raise InvalidConfiguration(f"nasa spdf download method (given: {config.nasa_spdf_download_method}) "
                                   f"must be 'ftp' or 'http'")
    downloader.download(start_date, end_date)
    logger.info("'download_omni' completed")


def download_all(start_date, end_date):
    download_tec(start_date, end_date)
    download_arb(start_date, end_date)
    download_omni(start_date, end_date)


def process_tec():
    logger.info(f"running 'process_tec'")
    trough_tec.process_tec_dataset()


def process_arb():
    logger.info(f"running 'process_arb'")
    trough_prepare.process_auroral_boundary_dataset()


def process_omni():
    logger.info(f"running 'process_omni'")
    trough_prepare.process_omni_dataset()


def process_all():
    process_tec()
    process_arb()
    process_omni()


def label_trough():
    trough_label.label_trough()


def full_run(start_date, end_date):
    download_all(start_date, end_date)
    process_all()
    label_trough()
