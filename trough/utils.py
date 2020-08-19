import h5py
import numpy as np
import datetime
import pandas as pd
import xarray as xr
import os
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as u
import glob
import re
from scipy.interpolate import UnivariateSpline


def read_converted_tec_h5(fn):
    """Opens one of Sebastijan's H5 GPS TEC files

    Parameters
    ----------
    fn: string
            filename to open, must be an H5 file

    Returns
    -------
    dictionary
            contains the following keys:
            - time (T, )
            - longitude (M, )
            - latitude (N, )
            - TEC (T, M, N)
    """
    with h5py.File(fn, 'r') as f:
        t = pd.to_datetime(f['GPSTEC/time'][()], unit='s')
        lon = f['GPSTEC/lon'][()]
        lat = f['GPSTEC/lat'][()]
        images = f['GPSTEC/im'][()]
    return {'time': t, 'longitude': lon, 'latitude': lat, 'tec': images}


def convert_sebs_h5_to_nc(fn, data_folder="E:\\tec_data\\data"):
    """converts one of Sebastijan's H5 GPS TEC files to NetCDF using xarray

    Parameters
    ----------
    fn: string
            input filename
    data_folder: string
            output folder

    Returns
    -------
    string
            converted filename
    """
    data_dict = read_converted_tec_h5(fn)
    coords = [data_dict['time'], data_dict['longitude'], data_dict['latitude']]
    dims = ['time', 'longitude', 'latitude']
    data_array = xr.DataArray(data=data_dict['tec'], coords=coords, dims=dims, name='tec')
    dt_accessor = data_array.time[0].dt
    nc_file_name = os.path.join(data_folder, f"{dt_accessor.year.item():04d}{dt_accessor.month.item():02d}{dt_accessor.day.item():02d}_tec.nc")
    data_array.to_netcdf(nc_file_name)
    return nc_file_name


def convert_all_sebs_data():
    """converts all of Sebastijan's h5 data to NetCDF using xarray
    """
    base_dir = "E:\\tec_data\\data"
    for root, dirnames, files in os.walk(base_dir):
        for file in files:
            if 'conv' in file:
                full_file_path = os.path.join(root, file)
                try:
                    print(convert_sebs_h5_to_nc(full_file_path))
                except Exception as e:
                    print(e)


def datetime64_to_datetime(dt64):
    """convert a numpy.datetime64 to a datetime.datetime

    Parameters
    ----------
    dt64: numpy.datetime64
            the datetime64 to convert

    Returns
    -------
    datetime.datetime
            the converted object
    """
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)


def timestamp_to_datetime(ts):
    return datetime.datetime.utcfromtimestamp(ts)


def get_sun_elevation(time, glon, glat):
    """Returns the sun elevation angle in degrees given a single time and arrays of geographic longitudes and latitudes

    Parameters
    ----------
    time: datetime-like, anything that can be passed to astropy.time.Time.
            Time to get elevation for. Only pass in one.
    glon: array-like (M x N)
            geographic longitudes
    glat: array-like (M x N)
            geographic latitudes

    Returns
    -------
    array-like (M x N)
            sun elevation angles
    """
    locations = ac.EarthLocation(lat=glat.ravel() * u.deg, lon=glon.ravel() * u.deg)
    t = at.Time(time)
    sun = ac.get_sun(t)
    alt = sun.transform_to(ac.AltAz(obstime=t, location=locations)).alt.value
    return alt.reshape(glon.shape)


def get_mahali_files(data_dir="E:\\tec_data\\data\\rinex"):
    """Get all the rinex files for Mahali

    Parameters
    ----------
    data_dir: string
            base directory which holds all the rinex files

    Returns
    -------
    obs_file_dict: dictionary
            {receiver name: observation files for the receiver, ...}
    nav_file_dict: dictionary
            {folder name (should be date): navigation file for that day, ...}
    """
    obs_files = glob.glob(os.path.join(data_dir, '*', '*.15o'))
    nav_files = glob.glob(os.path.join(data_dir, '*', '*.15n'))
    names = list(set(os.path.basename(fn)[:4] for fn in obs_files))
    obs_file_dict = {"MAH"+re.search("(\\d+)", name).group(1): [fn for fn in obs_files if name in fn] for name in names}
    nav_file_dict = {}
    for fn in nav_files:
        day = os.path.basename(os.path.dirname(fn))
        if day in nav_file_dict:
            continue
        nav_file_dict[day] = fn
    return obs_file_dict, nav_file_dict
