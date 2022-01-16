from datetime import datetime, timedelta
import math
import pathlib
import socket
import logging
import abc
import ftplib
from urllib import request
import bs4
import re
from madrigalWeb import madrigalWeb

from trough.exceptions import InvalidConfiguration
from trough._arb import _parse_arb_fn

logger = logging.getLogger(__name__)


def _doy(date):
    return math.floor((date - datetime(date.year, 1, 1)) / timedelta(days=1)) + 1


class Downloader(abc.ABC):

    def __init__(self, download_dir: pathlib.Path, *args, **kwargs):
        self.download_dir = pathlib.Path(download_dir)

    @abc.abstractmethod
    def _get_file_list(self, start_date, end_date):
        ...

    @abc.abstractmethod
    def _download_files(self, files):
        ...

    def download(self, start_date: datetime, end_date: datetime):
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("collecting file information...")
        files = self._get_file_list(start_date, end_date)
        logger.info(f"downloading {len(files)} files")
        self._download_files(files)


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
        return self.server.getExperiments(
            8000,
            start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute, start_date.second,
            end_date.year, end_date.month, end_date.day, end_date.hour, end_date.minute, end_date.second,
        )

    def _download_file(self, tec_file, local_path):
        logger.info(f"downloading TEC file {tec_file.name} to {local_path}")
        try:
            return self.server.downloadFile(
                tec_file.name, str(local_path), self.user_name, self.user_email, self.user_affil, 'hdf5'
            )
        except socket.timeout:
            logger.error(f'Failure downloading {tec_file.name} because it took more than allowed number of seconds')

    def _download_files(self, files):
        for file in files:
            server_path = pathlib.PurePosixPath(file.name)
            local_path = self.download_dir / f"{server_path.stem}.hdf5"
            self._download_file(file, local_path)

    def _get_file_list(self, start_date, end_date):
        tec_files = []
        experiments = sorted(self._get_tec_experiments(start_date - timedelta(hours=3), end_date + timedelta(hours=3)))
        for experiment in experiments:
            experiment_files = self.server.getExperimentFiles(experiment.id)
            tec_files += [exp for exp in experiment_files if exp.kindat == 3500]
        return tec_files


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
        for file in files:
            file_name = file.split('/')[-1]
            local_path = self.download_dir / file_name
            self._download_file(file, local_path)

    def _download_http_file(self, file, local_path):
        url = "https://spdf.gsfc.nasa.gov" + file
        _download_http_file(url, local_path)

    def _download_ftp_file(self, file, local_path):
        _download_ftp_file(self.server, file, local_path)

    def _get_file_list(self, start_date, end_date):
        new_start_date = start_date - timedelta(hours=3)
        new_end_date = end_date + timedelta(hours=3)
        files = [f'/pub/data/omni/low_res_omni/omni2_{year:4d}.dat'
                 for year in range(new_start_date.year, new_end_date.year + 1)]
        return files


class ArbDownloader(Downloader):

    def __init__(self, download_dir, *args, **kwargs):
        super().__init__(download_dir, *args, **kwargs)
        self.satellites = ['f16', 'f17', 'f18', 'f19']

    def _download_files(self, files):
        logger.info(f"downloading {len(files)} files")
        for file in files:
            file_name = file.split('/')[-1]
            local_path = self.download_dir / file_name
            _download_http_file(file, local_path)

    def _get_file_list(self, start_date, end_date):
        start_date -= timedelta(hours=3)
        end_date += timedelta(hours=3)
        n_days = math.ceil((end_date - start_date) / timedelta(days=1))
        logger.info(f"getting files for {n_days} days")
        days = [start_date + timedelta(days=t) for t in range(n_days)]
        dates = [d.date() for d in days]
        years = set([date.year for date in days])
        date_struct = {year: [_doy(date) for date in days if date.year == year] for year in years}
        files = []

        for satellite in self.satellites:
            for year, doys in date_struct.items():
                for doy in doys:
                    url = f'https://ssusi.jhuapl.edu/data_retriver?spc={satellite}&type=edr-aur&year={year:04d}&Doy={doy:03d}'
                    with request.urlopen(url) as r:
                        if r.status == 200:
                            soup = bs4.BeautifulSoup(r.read(), 'html.parser')
                            links = soup.find_all('a')
                            for link in links:
                                if 'href' in link.attrs and re.match('PS\.APL_.+EDR-AURORA.+\.NC', str(link.string)):
                                    sat_name, date = _parse_arb_fn(pathlib.Path(link['href']))
                                    if date.date() in dates and sat_name.lower() == satellite:
                                        files.append(f"https://ssusi.jhuapl.edu/{link['href']}")
        return files


def _download_ftp_file(server, server_file, local_path):
    logger.info(f"downloading file {server_file} to {local_path}")
    with open(local_path, 'wb') as f:
        server.retrbinary(f'RETR {str(server_file)}', f.write)


def _download_http_file(http_file, local_path):
    logger.info(f"downloading file {http_file} to {local_path}")
    with request.urlopen(http_file) as r:
        with open(local_path, 'wb') as f:
            f.write(r.read())
