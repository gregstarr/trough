import apexpy
import numpy as np

import trough


def mlt_to_geo(mlat, mlt, times, converter=None, height=0, ssheight=50*6371):
    """Convert MLT coordinates to geographic

    Parameters
    ----------
    mlat, mlt: np.ndarray float
            magnetic latitude, magnetic local time and UT arrays, all 1d and the same size
    times: xarray time index
            times gotten from an xarray as DataArray.time
    converter: apexpy.Apex
            pre-initialized apex converter
    height: float
    ssheight: float

    Returns
    -------
    lat, lon: np.ndarray
            geographic latitude and longitude
    """
    broadcast = True
    n_locs = mlat.shape[0]
    n_times = times.shape[0]
    if n_locs == n_times:
        broadcast = False
    if converter is None:
        converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(times[0]))
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)
    if broadcast:
        mlon = (15 * mlt[None, :] - 180 + ssalon[:, None] + 360) % 360
        mlat = mlat[None, :] * np.ones_like(times, dtype=float)[:, None]
        lat, lon, _ = converter.apex2geo(mlat.ravel(), mlon.ravel(), height, precision=1e-3)
        return lat.reshape((n_times, n_locs)), lon.reshape((n_times, n_locs))
    else:
        mlon = (15 * mlt - 180 + ssalon + 360) % 360
        lat, lon, _ = converter.apex2geo(mlat, mlon, height, precision=1e-3)
        return lat, lon


def geo_to_mlt(lat, lon, times, converter=None, height=0, ssheight=50*6371):
    if converter is None:
        converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(times[0]))
    mlat, mlon = converter.geo2apex(lat, lon, height)
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)
    if lon.shape == times.shape:
        mlt = (180 + mlon - ssalon) / 15 % 24
        return mlat, mlt
    mlt = (180 + mlon[None, :] - ssalon[:, None]) / 15 % 24
    return mlat[None, :] * np.ones_like(times, dtype=float), mlt


def mlon_to_mlt(mlon, times, converter=None, ssheight=50*6371):
    if converter is None:
        converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(times[0]))
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)
    # np.float64 will ensure lists are converted to arrays
    if mlon.shape == times.shape:
        return (180 + np.float64(mlon) - ssalon) / 15 % 24
    return (180 + mlon[None, :] - ssalon[:, None]) / 15 % 24


def mlt_to_mlon(mlt, times, converter=None, ssheight=50*6371):
    if converter is None:
        converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(times[0]))
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)

    if mlt.shape == times.shape:
        return (15 * np.float64(mlt) - 180 + ssalon + 360) % 360
    return (15 * mlt[None, :] - 180 + ssalon[:, None] + 360) % 360


def subsol_array(times):
    # convert to year, day of year and seconds since midnight
    year = times.dt.year.values
    doy = times.dt.dayofyear.values
    ut = times.dt.hour.values * 3600 + times.dt.minute.values * 60 + times.dt.second.values

    if not np.all(1601 <= year) and np.all(year <= 2100):
        raise ValueError('Year must be in [1601, 2100]')

    yr = year - 2000

    nleap = np.floor((year - 1601.0) / 4.0).astype(int)
    nleap -= 99
    mask_1900 = year <= 1900
    if np.any(mask_1900):
        ncent = np.floor((year[mask_1900] - 1601.0) / 100.0).astype(int)
        ncent = 3 - ncent[mask_1900]
        nleap[mask_1900] = nleap[mask_1900] + ncent

    l0 = -79.549 + (-0.238699 * (yr - 4.0 * nleap) + 3.08514e-2 * nleap)
    g0 = -2.472 + (-0.2558905 * (yr - 4.0 * nleap) - 3.79617e-2 * nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut / 86400.0 - 1.5) + doy

    # Mean longitude of Sun:
    lmean = l0 + 0.9856474 * df

    # Mean anomaly in radians:
    grad = np.radians(g0 + 0.9856003 * df)

    # Ecliptic longitude:
    lmrad = np.radians(lmean + 1.915 * np.sin(grad)
                       + 0.020 * np.sin(2.0 * grad))
    sinlm = np.sin(lmrad)

    # Obliquity of ecliptic in radians:
    epsrad = np.radians(23.439 - 4e-7 * (df + 365 * yr + nleap))

    # Right ascension:
    alpha = np.degrees(np.arctan2(np.cos(epsrad) * sinlm, np.cos(lmrad)))

    # Declination, which is also the subsolar latitude:
    sslat = np.degrees(np.arcsin(np.sin(epsrad) * sinlm))

    # Equation of time (degrees):
    etdeg = lmean - alpha
    nrot = np.round(etdeg / 360.0)
    etdeg = etdeg - 360.0 * nrot

    # Subsolar longitude:
    sslon = 180.0 - (ut / 240.0 + etdeg)  # Earth rotates one degree every 240 s.
    nrot = np.round(sslon / 360.0)
    sslon = sslon - 360.0 * nrot

    return sslat, sslon
