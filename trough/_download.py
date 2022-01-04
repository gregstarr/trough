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
        experiments = sorted(self._get_tec_experiments(start_date, end_date))
        for experiment in experiments:
            experiment_files = self.server.getExperimentFiles(experiment.id)
            tec_files += [exp for exp in experiment_files if exp.kindat == 3500]
        return tec_files


class NasaSpdfFtpDownloader(Downloader, abc.ABC):

    def __init__(self, download_dir):
        super().__init__(download_dir)
        logger.info("connecting to server")
        self.server = ftplib.FTP_TLS("spdf.gsfc.nasa.gov")
        self.server.login()

    def _download_file(self, server_file, local_path):
        logger.info(f"downloading file {server_file.name} to {local_path}")
        with open(local_path, 'wb') as f:
            self.server.retrbinary(f'RETR {str(server_file)}', f.write)

    def _download_files(self, files):
        logger.info(f"downloading {len(files)} files")
        for file in files:
            server_file = pathlib.PurePosixPath(file)
            local_path = self.download_dir / f"{server_file.name}"
            self._download_file(server_file, local_path)


class AuroralBoundaryFtpDownloader(NasaSpdfFtpDownloader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.satellites = ['dmspf16', 'dmspf17', 'dmspf18', 'dmspf19']

    def _get_file_list(self, start_date, end_date):
        n_days = math.ceil((end_date - start_date) / timedelta(days=1))
        days = [start_date + timedelta(days=t) for t in range(n_days)]
        years = set([date.year for date in days])
        date_struct = {year: [_doy(date) for date in days if date.year == year] for year in years}
        files = []

        for satellite in self.satellites:
            sat_years = self.server.nlst(f'/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora')
            for year, doys in date_struct.items():
                year_dir = f'/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora/{year}'
                if year_dir in sat_years:
                    sat_doys = self.server.nlst(year_dir)
                    for doy in doys:
                        doy_dir = f'/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora/{year}/{doy:03d}'
                        if doy_dir in sat_doys:
                            files += self.server.nlst(doy_dir)
        return files


class OmniFtpDownloader(NasaSpdfFtpDownloader):

    def _get_file_list(self, start_date, end_date):
        files = [f'/pub/data/omni/low_res_omni/omni2_{year:4d}.dat'
                 for year in range(start_date.year, end_date.year + 1)]
        return files


class NasaSpdfHttpDownloader(Downloader, abc.ABC):

    def _download_file(self, server_file, local_path):
        logger.info(f"downloading file {server_file} to {local_path}")
        with request.urlopen(server_file) as r:
            with open(local_path, 'wb') as f:
                f.write(r.read())

    def _download_files(self, files):
        for file in files:
            file_name = file.split('/')[-1]
            local_path = self.download_dir / file_name
            self._download_file(file, local_path)


class OmniHttpDownloader(NasaSpdfHttpDownloader):

    def _get_file_list(self, start_date, end_date):
        files = [f'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year:4d}.dat'
                 for year in range(start_date.year, end_date.year + 1)]
        return files


class AuroralBoundaryHttpDownloader(NasaSpdfHttpDownloader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.satellites = ['dmspf16', 'dmspf17', 'dmspf18', 'dmspf19']

    def get_links(self, url, pattern):
        with request.urlopen(url) as r:
            soup = bs4.BeautifulSoup(r.read())
            links = soup.find_all('a')
        return [url + link.attrs['href'] for link in links if re.match(pattern, link.string)]

    def _get_file_list(self, start_date, end_date):
        n_days = math.ceil((end_date - start_date) / timedelta(days=1)) + 1
        days = [start_date + timedelta(days=t) for t in range(n_days)]
        years = set([date.year for date in days])
        date_struct = {year: [_doy(date) for date in days if date.year == year] for year in years}
        files = []
        for satellite in self.satellites:
            sat_years = self.get_links(f'https://spdf.gsfc.nasa.gov/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora/', '\d{4}')
            for year, doys in date_struct.items():
                year_dir = f'https://spdf.gsfc.nasa.gov/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora/{year}/'
                if year_dir in sat_years:
                    sat_doys = self.get_links(year_dir, '\d{3}')
                    for doy in doys:
                        doy_dir = f'https://spdf.gsfc.nasa.gov/pub/data/dmsp/{satellite}/ssusi/data/edr-aurora/{year}/{doy:03d}/'
                        if doy_dir in sat_doys:
                            files += self.get_links(doy_dir, '.*\.nc')
        return files
