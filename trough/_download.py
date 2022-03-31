import numpy as np
from datetime import datetime, timedelta
import math
import pathlib
import socket
import abc
import ftplib
from urllib import request
import re
import json
import functools
import logging
import warnings
try:
    import h5py
    from madrigalWeb import madrigalWeb
    import bs4
except ImportError as imp_err:
    warnings.warn(f"Packages required for recreating dataset not installed: {imp_err}")


from trough.exceptions import InvalidConfiguration
from trough._arb import parse_arb_fn

logger = logging.getLogger(__name__)


def _doy(date):
    return math.floor((date - datetime(date.year, 1, 1)) / timedelta(days=1)) + 1


class Downloader(abc.ABC):

    def __init__(self, download_dir: pathlib.Path, *args, **kwargs):
        self.download_dir = pathlib.Path(download_dir)
        self.cache_fn = self.download_dir / "file_list.json"
        self.cache = {}
        if self.cache_fn.exists():
            with open(self.cache_fn) as f:
                self.cache = json.load(f)

    @abc.abstractmethod
    def _get_file_list(self, start_date, end_date):
        """get list of files on server

        Parameters
        ----------
        start_date: datetime
        end_date: datetime

        Returns
        -------
        dict[str, list[str]]
            dictionary mapping date id to file list for that date
        """
        ...

    @abc.abstractmethod
    def _download_files(self, files):
        """download list of server files

        Parameters
        ----------
        files: list[str]
            server file paths

        Returns
        -------
        list[str]
            local file paths
        """
        ...

    @abc.abstractmethod
    def _verify_files(self, local_files, files):
        """verify a list of local files, return server file paths corresponding to corrupted local files

        Parameters
        ----------
        local_files: list[str]
        files: list[str]

        Returns
        -------
        list[str]
            bad files (server)
        """
        ...

    def download(self, start_date: datetime, end_date: datetime):
        """Runs download routine

        Parameters
        ----------
        start_date: datetime
        end_date: datetime
        """
        # make sure download dict exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("collecting file information...")
        # get dictionary mapping some id (e.g. date) to file lists
        download_dict = self._get_file_list(start_date, end_date)
        # update and save file list
        self.cache.update(**download_dict)
        with open(self.cache_fn, 'w') as f:
            json.dump(self.cache, f)
        # collect files
        server_files = functools.reduce(lambda x, y: x + y, download_dict.values())
        logger.info(f"downloading {len(server_files)} files")
        # download files
        local_files = self._download_files(server_files)
        # make sure all files open and have data
        logger.info("verifying...")
        bad_server_files = self._verify_files(local_files, server_files)
        logger.info(f"{len(bad_server_files)} bad files")
        if bad_server_files:
            fixed_local_files = self._download_files(bad_server_files)
            still_bad_server_files = self._verify_files(fixed_local_files, bad_server_files)
            for file in still_bad_server_files:
                logger.error(f"unable to properly download {file}")


class MadrigalTecDownloader(Downloader):

    def __init__(self, download_dir, user_name, user_email, user_affil):
        super().__init__(download_dir)
        if None in [user_name, user_email, user_affil]:
            raise InvalidConfiguration("To download from Madrigal, user name, email, and affiliation must be specified")
        self.user_name = user_name
        self.user_email = user_email
        self.user_affil = user_affil
        logger.info("connecting to server")
        self.server = madrigalWeb.MadrigalData("http://cedar.openmadrigal.org")

    def _get_tec_experiments(self, start_date: datetime, end_date: datetime):
        logger.info(f"getting TEC experiments between {start_date} and {end_date}")
        experiments = self.server.getExperiments(
            8000,
            start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute, start_date.second,
            end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second,
        )
        return experiments

    def _download_file(self, tec_file, local_path):
        logger.info(f"downloading TEC file {tec_file} to {local_path}")
        try:
            if pathlib.Path(local_path).exists():
                logger.info(f"already exists: {local_path}")
            else:
                return self.server.downloadFile(
                    tec_file, local_path, self.user_name, self.user_email, self.user_affil, 'hdf5'
                )
        except socket.timeout:
            logger.error(f'Failure downloading {tec_file} because it took more than allowed number of seconds')

    def _download_files(self, files):
        local_files = []
        for i, file in enumerate(files):
            if len(files) > 100 and not (i % (len(files) // 100)):
                logger.info(f"{round(100 * i / len(files))}% finished")
            server_path = pathlib.PurePosixPath(file)
            local_path = str(self.download_dir / f"{server_path.stem}.hdf5")
            local_files.append(local_path)
            self._download_file(file, local_path)
        return local_files

    def _get_file_list(self, start_date, end_date):
        logger.info("Getting file list...")
        experiments = sorted(self._get_tec_experiments(start_date - timedelta(hours=3), end_date + timedelta(hours=3)))
        logger.info(f"found {len(experiments)} experiments")
        tec_files = {}
        for i, experiment in enumerate(experiments):
            if len(experiments) > 100 and not (i % (len(experiments) // 100)):
                logger.info(f"{round(100 * i / len(experiments))}% finished")
            cache_key = str(experiment.id)
            if cache_key in self.cache:
                files = self.cache[cache_key]
            else:
                experiment_files = self.server.getExperimentFiles(experiment.id)
                files = [exp.name for exp in experiment_files if exp.kindat == 3500]
            tec_files[cache_key] = files
        return tec_files

    @staticmethod
    def _verify_local_file(local_file):
        try:
            with h5py.File(local_file, 'r') as f:
                tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
                timestamps = f['Data']['Array Layout']['timestamps'][()]
        except Exception as e:
            logger.warning(f"bad local file: {local_file}, error: {e}")
            return False
        return (timestamps.shape[0] > 10) and (np.sum(np.isfinite(tec)) > 100)

    def _verify_files(self, local_files, server_files):
        bad_server_files = [
            server_file for (server_file, local_file) in zip(server_files, local_files)
            if not self._verify_local_file(local_file)
        ]
        return bad_server_files


class OmniDownloader(Downloader):

    def __init__(self, download_dir, method='ftp', *args, **kwargs):
        super().__init__(download_dir, *args, **kwargs)
        self.method = method
        if method == 'ftp':
            logger.info("connecting to server")
            self.server = ftplib.FTP_TLS("spdf.gsfc.nasa.gov")
            self.server.login()
            self._download_file = self._download_ftp_file
        elif method == 'http':
            self._download_file = self._download_http_file

    def _download_files(self, files):
        logger.info(f"downloading {len(files)} files")
        local_files = []
        for file in files:
            file_name = file.split('/')[-1]
            local_path = str(self.download_dir / file_name)
            local_files.append(local_path)
            self._download_file(file, local_path)
        return local_files

    @staticmethod
    def _download_http_file(file, local_path):
        url = "https://spdf.gsfc.nasa.gov" + file
        _download_http_file(url, local_path)

    def _download_ftp_file(self, file, local_path):
        _download_ftp_file(self.server, file, local_path)

    def _get_file_list(self, start_date, end_date):
        new_start_date = start_date - timedelta(hours=3)
        new_end_date = end_date + timedelta(hours=3)
        files = {
            str(year): [f'/pub/data/omni/low_res_omni/omni2_{year:4d}.dat']
            for year in range(new_start_date.year, new_end_date.year + 1)
        }
        return files

    @staticmethod
    def _verify_local_file(local_file):
        return (pathlib.Path(local_file).stat().st_size / (2 ** 20)) > 1  # file size > 1Mb

    def _verify_files(self, local_files, server_files):
        bad_server_files = [
            server_file for (server_file, local_file) in zip(server_files, local_files)
            if not self._verify_local_file(local_file)
        ]
        return bad_server_files


class ArbDownloader(Downloader):

    def __init__(self, download_dir, *args, **kwargs):
        super().__init__(download_dir, *args, **kwargs)
        self.satellites = ['f16', 'f17', 'f18', 'f19']

    def _download_files(self, files):
        logger.info(f"downloading {len(files)} files")
        local_files = []
        for i, file in enumerate(files):
            if len(files) > 100 and not (i % (len(files) // 100)):
                logger.info(f"{round(100 * i / len(files))}% finished")
            file_name = file.split('/')[-1]
            local_path = str(self.download_dir / file_name)
            local_files.append(local_path)
            if pathlib.Path(local_path).exists():
                logger.info(f"already exists: {local_path}")
            else:
                _download_http_file(file, local_path)
        return local_files

    def _get_file_list(self, start_date, end_date):
        date1 = start_date - timedelta(days=1)
        date2 = end_date + timedelta(days=1)
        n_days = math.ceil((date2 - date1) / timedelta(days=1))
        logger.info(f"getting files for {n_days} days")
        days = [date1 + timedelta(days=t) for t in range(n_days)]
        arb_files = {}
        for i, day in enumerate(days):
            if len(days) > 100 and not (i % (len(days) // 100)):
                logger.info(f"{round(100 * i / len(days))}% finished")
            arb_files.update(self._get_files_for_day(day))
        return arb_files

    def _get_files_for_day(self, day):
        files = {}
        for satellite in self.satellites:
            doy = _doy(day)
            year = day.year
            cache_key = f"{satellite}_{year}_{doy}"
            if cache_key in self.cache:
                files[cache_key] = self.cache[cache_key]
            else:
                files[cache_key] = []
                url = f'https://ssusi.jhuapl.edu/data_retriver?spc={satellite}&type=edr-aur&' \
                      f'year={year:04d}&Doy={doy:03d}'
                with request.urlopen(url) as r:
                    if r.status == 200:
                        soup = bs4.BeautifulSoup(r.read(), 'html.parser')
                        links = soup.find_all('a')
                        for link in links:
                            if 'href' in link.attrs and re.match(r'PS\.APL_.+EDR-AURORA.+\.NC', str(link.string)):
                                sat_name, date = parse_arb_fn(pathlib.Path(link['href']))
                                if date.date() == day.date() and sat_name.lower() == satellite:
                                    files[cache_key].append(f"https://ssusi.jhuapl.edu/{link['href']}")
        return files

    @staticmethod
    def _verify_local_file(local_file):
        try:
            with h5py.File(local_file, 'r') as f:
                lon = f['MODEL_NORTH_GEOGRAPHIC_LONGITUDE'][()]
        except Exception as e:
            logger.warning(f"bad local file: {local_file}, error: {e}")
            return False
        return lon.shape[0] > 10

    def _verify_files(self, local_files, server_files):
        bad_server_files = [
            server_file for (server_file, local_file) in zip(server_files, local_files)
            if not self._verify_local_file(local_file)
        ]
        return bad_server_files


def _download_ftp_file(server, server_file: str, local_path: str):
    logger.info(f"downloading file {server_file} to {local_path}")
    with open(local_path, 'wb') as f:
        server.retrbinary(f'RETR {str(server_file)}', f.write)


def _download_http_file(http_file: str, local_path: str):
    logger.info(f"downloading file {http_file} to {local_path}")
    try:
        with request.urlopen(http_file, timeout=60) as r:
            with open(local_path, 'wb') as f:
                f.write(r.read())
    except socket.timeout:
        logger.error(f'Failure downloading {http_file} because it took more than allowed number of seconds')
