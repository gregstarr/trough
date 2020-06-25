import h5py
import numpy as np
import datetime
import pandas as pd
import xarray as xr
import os
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as u
import astropy.constants as aconst
from skimage import measure
import apexpy
import glob
import re


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


def convert_h5_to_nc(fn, data_folder="E:\\tec_data\\data"):
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
                    print(convert_h5_to_nc(full_file_path))
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


def get_terminator(time, alt_km=0, resolution=1):
    """This is a plotting convenience function which will return the terminator (sun boundary) lines for a given time.

    Parameters
    ----------
    time: datetime-like, anything that can be passed to astropy.time.Time.
            Time to get terminator lines for. Only pass in one.
    alt_km: float
            altitude to get terminator lines for
    resolution: float
            resolution of resulting terminator lines

    Returns
    -------
    list of shape (N, ) ndarrays
            longitudes of each line
    list of shape (N, ) ndarrays
            latitudes of each line
    """
    glon, glat = np.meshgrid(np.arange(-180, 180, resolution), np.arange(-90, 90.1, resolution))

    sun_el = get_sun_elevation(time, glon, glat)
    r_e = aconst.R_earth.to('km').value
    horizon = np.rad2deg(np.arccos(r_e / (r_e + alt_km)))
    terminators = measure.find_contours(sun_el, -1 * horizon)
    x = []
    y = []
    for terminator in terminators:
        x.append(terminator[:, 1] - 180)
        y.append(terminator[:, 0] - 90)

    return x, y


def get_magnetic_coordinate_lines(date, coord_sys='mlt', height=0, mlat_levels=None, mlon_levels=None, resolution=1):
    """This is a plotting convenience function which will return the magnetic coordinate lines for a given time.

    Parameters
    ----------
    date: datetime.datetime or numpy.datetime64
            the time at which to find the magnetic coordinate lines
    coord_sys: str, optional
            the coordinate system to use for magnetic longitude, must be 'mlon' or 'mlt', default to 'mlt'
    height: float, optional
            the height to calculate the magnetic coordinate lines at, defaults to 0 (at surface)
    mlat_levels: iterable, optional
            the magnetic latitude lines to find, defaults to every 10 degrees from -80 to 80
    mlon_levels: iterable, optional
            the magnetic longitude lines to find, defaults to every 3 hours from 0 to 24
    resolution: float, optional
            resolution of the magnetic coordinate lines, defaults to 1 degree latitude and longitude

    Returns
    -------
    dictionary
            keys: 'mlat' and 'mlon'
            each key points to a dictionary whose keys are the requested levels for that coordinate and whose
            values are a list of (N x 2) arrays each describing a single line to draw

    """
    # input checking
    if isinstance(date, np.datetime64):
        date = datetime64_to_datetime(date)
    if mlat_levels is None:
        mlat_levels = np.arange(-80, 90, 10)
    if mlon_levels is None:
        if coord_sys == 'mlt':
            mlon_levels = np.arange(0, 24, 3)
        elif coord_sys == 'apex':
            mlon_levels = np.arange(0, 360, 45)
    # create grid, convert to magnetic
    glon, glat = np.meshgrid(np.arange(-180, 180, resolution), np.arange(-90, 90.1, resolution))
    apex_converter = apexpy.Apex(date=date)
    mag_lat, mag_lon = apex_converter.convert(glat, glon, 'geo', coord_sys, datetime=date, height=height)

    magnetic_coordinate_lines = {'mlat': {}, 'mlon': {}}
    # identify each magnetic latitude line
    for level in mlat_levels:
        magnetic_coordinate_lines['mlat'][level] = []
        lines = measure.find_contours(mag_lat, level)
        for line in lines:
            lon = line[:, 1] - 180
            lat = line[:, 0] - 90
            magnetic_coordinate_lines['mlat'][level].append(np.column_stack((lon, lat)))

    # deal with discontinuity
    y_diff = np.diff(mag_lon, axis=0, prepend=mag_lon[-1, None])
    x_diff = np.diff(mag_lon, axis=1, prepend=mag_lon[:, -1, None])
    diff_mag = x_diff**2 + y_diff**2
    mask = diff_mag < 10
    # identify each magnetic longitude line except for boundary lines
    for level in mlon_levels:
        if level in [0, 24, 360]:
            continue
        magnetic_coordinate_lines['mlon'][level] = []
        lines = measure.find_contours(mag_lon, level, mask=mask)
        for line in lines:
            lon = line[:, 1] - 180
            lat = line[:, 0] - 90
            magnetic_coordinate_lines['mlon'][level].append(np.column_stack((lon, lat)))

    for level in [0, 24, 360]:
        if level in mlon_levels:
            magnetic_coordinate_lines['mlon'][level] = []
            param = np.linspace(-90, 90, 100)
            lat, lon = apex_converter.convert(param, level, coord_sys, 'geo', height=height, datetime=date)
            magnetic_coordinate_lines['mlon'][level].append(np.column_stack((lon, lat)))

    return magnetic_coordinate_lines


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
    sat_bias_files = glob.glob(os.path.join(data_dir, "DCB", "*.BSX"))
    return obs_file_dict, nav_file_dict
