import apexpy
import numpy as np

import trough


def mlt_to_geo(mlat, mlt, times, height=0, ssheight=50*6371):
    """

    Parameters
    ----------
    mlat
    mlon
    time
    height
    ssheight

    Returns
    -------

    """
    converter = apexpy.Apex(date=trough.utils.datetime64_to_datetime(times[0]))
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)

    # np.float64 will ensure lists are converted to arrays
    mlon = np.ravel((15 * mlt[None, :] - 180 + ssalon[:, None] + 360) % 360)
    mlat = np.ravel(mlat[None, :] * np.ones((times.size, 1)))
    lat, lon, _ = converter.apex2geo(mlat, mlon, height)
    return lat.reshape((times.size, -1)), lon.reshape((times.size, -1))


def subsol_array(times):
    """Finds subsolar geocentric latitude and longitude.

    Parameters
    ==========
    datetime : :class:`datetime.datetime`

    Returns
    =======
    sbsllat : float
        Latitude of subsolar point
    sbsllon : float
        Longitude of subsolar point

    Notes
    =====
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994). Usable for years 1601-2100,
    inclusive. According to the Almanac, results are good to at least 0.01
    degree latitude and 0.025 degrees longitude between years 1950 and 2050.
    Accuracy for other years has not been tested. Every day is assumed to have
    exactly 86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored (their effect is below the accuracy threshold of the
    algorithm).

    After Fortran code by A. D. Richmond, NCAR. Translated from IDL
    by K. Laundal.

    """
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
