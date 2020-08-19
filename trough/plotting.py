import numpy as np
import xarray as xr
import pymap3d as pm
import apexpy
from astropy import constants as aconst
import datetime
from skimage import measure
from skimage import filters
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib import transforms
from cartopy.feature import COASTLINE
from cartopy.mpl.geoaxes import GeoAxes
from cartopy import crs

from trough import utils, convert, gps


_cache = {}


########################################################################################################################
# UTILITIES ############################################################################################################
########################################################################################################################
def unlink_wrap(dat, lims=(-np.pi, np.pi), thresh=0.95):
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))


########################################################################################################################
# AXES #################################################################################################################
########################################################################################################################
def format_polar_mag_ax(ax):
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.set_ylim(0, 60)
    ax.set_xticks(np.arange(8) * np.pi/4)
    ax.set_xticklabels((np.arange(8) * 3 + 6) % 24)
    ax.set_yticks([10, 20, 30, 40, 50])
    ax.set_yticklabels([80, 70, 60, 50, 40])
    ax.grid()
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, labelleft=True, width=0, length=0)
    ax.set_rlabel_position(80)


########################################################################################################################
# BORDERS ##############################################################################################################
########################################################################################################################
def plot_mlt_lines_mag(ax, date, mlt_vals=np.arange(-6, 7, 3)):
    date = utils.datetime64_to_datetime(date.values)
    converter = apexpy.Apex(date=date)
    mlon = converter.mlt2mlon(mlt_vals, date)
    mlon[mlon > 180] -= 360
    ax.vlines(mlon, 20, 80, colors='w', linestyles='--')
    for i in range(mlt_vals.shape[0]):
        ax.text(mlon[i], 20, str(mlt_vals[i]), color='w')


def plot_solar_terminator(ax, date, altitude=300, **plot_kwargs):
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    converter = apexpy.Apex(date=date)
    glon, glat = np.meshgrid(np.arange(-180, 180, 1), np.arange(-90, 90.1, 1))
    sun_el = utils.get_sun_elevation(date, glon, glat)
    r_e = aconst.R_earth.to('km').value
    horizon = np.rad2deg(np.arccos(r_e / (r_e + altitude)))
    terminators = measure.find_contours(sun_el, -1 * horizon)
    x = []
    y = []
    for terminator in terminators:
        x.append(terminator[:, 1] - 180)
        y.append(terminator[:, 0] - 90)
    for lon, lat in zip(x, y):
        mlat, mlt = converter.convert(lat, lon, 'geo', 'mlt', datetime=date)
        mlt[mlt > 12] -= 24
        theta = np.pi * (mlt - 6) / 12
        r = 90 - mlat
        ax.plot(theta, r, **plot_kwargs)


########################################################################################################################
# BORDERS ##############################################################################################################
########################################################################################################################
def plot_coastline_geo(ax, **plot_kwargs):
    for geo in COASTLINE.geometries():
        for line in geo:
            x, y = line.xy
            for slc in unlink_wrap(x, [-12, 12]):
                ax.plot(x[slc], y[slc], 'k', **plot_kwargs)


def plot_coastline_mag(ax, date, coord_sys='apex', **plot_kwargs):
    if isinstance(date, np.datetime64):
        datetime = utils.datetime64_to_datetime(date)
    else:
        datetime = date
    converter = apexpy.Apex(date=datetime)
    for geo in COASTLINE.geometries():
        for line in geo:
            x, y = line.xy
            y, x = converter.convert(y, x, 'geo', coord_sys, datetime=datetime)
            if coord_sys == 'mlt':
                x[x > 12] -= 24
            for slc in unlink_wrap(x, [-12, 12]):
                if isinstance(ax, PolarAxes):
                    if coord_sys == 'mlt':
                        theta = np.pi * (x[slc] - 6) / 12
                    else:
                        theta = np.pi * x[slc] / 180
                    r = 90 - y[slc]
                    ax.plot(theta, r, 'k', **plot_kwargs)
                else:
                    ax.plot(x[slc], y[slc], 'k', **plot_kwargs)


########################################################################################################################
# TEC MAP ##############################################################################################################
########################################################################################################################
def plot_tec_map_mag(ax, date, data, coord_sys='mlt', **plot_kwargs):
    """Plot a magnetic coordinates TEC map no matter what: cartesian, polar, mlon, mlt
    """
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    elif isinstance(date, datetime.datetime):
        pass
    else:
        raise Exception("bad date")

    if coord_sys == 'mlt':
        converter = apexpy.Apex(date=date)
        x = converter.mlon2mlt(data.mlon.values, date)
        x[x > 12] -= 24
    else:
        x = data.mlon.values

    x, y = np.meshgrid(x, data.mlat.values)

    if isinstance(ax, PolarAxes):
        if coord_sys == 'mlt':
            theta = np.pi * (x - 6) / 12
        else:
            theta = np.pi * (x - 90) / 180
        r = 90 - y
        ax.pcolormesh(theta, r, data.values, **plot_kwargs)
    else:
        sorting_index = np.argsort(x[0])
        ax.pcolormesh(x[:, sorting_index], y[:, sorting_index], data.values[:, sorting_index], **plot_kwargs)


def plot_lr_debug(ax, date, debug):
    """plot:
        - section
        - trough area
    """
    data = debug.sel(time=date)
    used_low = []
    used_high = []
    if isinstance(ax, PolarAxes):
        for pwall, ewall, trough, low_mlt, high_mlt in zip(data['pwall'].item(), data['ewall'].item(),
                                                           data['trough'].item(), data['low_mlt'].item(), data['high_mlt'].item()):
            theta_l = np.pi * (low_mlt - 6) / 12
            theta_h = np.pi * (high_mlt - 6) / 12
            if trough:
                ax.plot([theta_l, theta_h], [90 - pwall, 90 - pwall], f'r-')
                ax.plot([theta_l, theta_h], [90 - ewall, 90 - ewall], f'r-')
            if np.any(abs(np.array(used_high) - high_mlt) < .5):
                continue
            if np.any(abs(np.array(used_low) - low_mlt) < .5):
                continue
            used_high.append(high_mlt)
            used_low.append(low_mlt)
            ax.plot([theta_l, theta_l], [10, 50], f'k-')
            ax.plot([theta_h, theta_h], [10, 50], f'k-')

    else:
        return


def plot_tec_trough(ax, date, troughs, coord_sys='mlt', **plot_kwargs):
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    elif isinstance(date, datetime.datetime):
        pass
    else:
        raise Exception("bad date")

    if coord_sys == 'mlt':
        converter = apexpy.Apex(date=date)
        x = converter.mlon2mlt(troughs.mlon.values, date)
        x[x > 12] -= 24
    else:
        x = troughs.mlon.values

    x, y = np.meshgrid(x, troughs.mlat.values)

    if isinstance(ax, PolarAxes):
        if coord_sys == 'mlt':
            theta = np.pi * (x - 6) / 12
        else:
            theta = np.pi * (x - 90) / 180
        r = 90 - y
        ax.contourf(theta, r, troughs.values, levels=1, colors='none', hatches=[None, '////'], **plot_kwargs)
    else:
        return


def plot_lr_predictions(ax, date, features, lr, coord_sys='mlt', **plot_kwargs):
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    elif isinstance(date, datetime.datetime):
        pass
    else:
        raise Exception("bad date")

    if coord_sys == 'mlt':
        converter = apexpy.Apex(date=date)
        x = converter.mlon2mlt(features.mlon.values, date)
        x[x > 12] -= 24
    else:
        x = features.mlon.values

    x, y = np.meshgrid(x, features.mlat.values)
    d = lr.predict_proba(features.values.reshape((-1, 4)))[:, 1].reshape(features.shape[:-1])

    if isinstance(ax, PolarAxes):
        if coord_sys == 'mlt':
            theta = np.pi * (x - 6) / 12
        else:
            theta = np.pi * (x - 90) / 180
        r = 90 - y
        ax.pcolormesh(theta, r, d, **plot_kwargs)
    else:
        sorting_index = np.argsort(x[0])
        # ax.pcolormesh(x[:, sorting_index], y[:, sorting_index], d, **plot_kwargs)


def plot_feature_maps(ax, date, data, coeffs=None, coord_sys='mlt', **plot_kwargs):
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    elif isinstance(data, datetime.datetime):
        pass
    else:
        raise Exception("bad date")

    if coord_sys == 'mlt':
        converter = apexpy.Apex(date=date)
        x = converter.mlon2mlt(data.mlon.values, date)
        x[x > 12] -= 24
    else:
        x = data.mlon.values

    x, y = np.meshgrid(x, data.mlat.values)

    for i, a in enumerate(ax):
        if isinstance(a, PolarAxes):
            if coord_sys == 'mlt':
                theta = np.pi * (x - 6) / 12
            else:
                theta = np.pi * (x - 90) / 180
            r = 90 - y
            if coeffs is not None:
                a.pcolormesh(theta, r, coeffs[:, i] * data.values[:, :, i], **plot_kwargs)
            else:
                a.pcolormesh(theta, r, data.values[:, :, i], **plot_kwargs)
        else:
            sorting_index = np.argsort(x[0])
            # ax.pcolormesh(x[:, sorting_index], y[:, sorting_index], d, **plot_kwargs)


def plot_tec_lat_profiles(polar_ax, line_axs, date, data, mlts=np.array((-3, -2, -1, 0, 1, 2, 3))):
    if isinstance(date, np.datetime64):
        date = utils.datetime64_to_datetime(date)
    elif isinstance(data, datetime.datetime):
        pass
    else:
        raise Exception("bad date")

    converter = apexpy.Apex(date=date)
    mlons = converter.mlt2mlon(mlts, date)
    mlons[mlons > 180] -= 360
    for i in range(mlts.shape[0]):
        theta = np.pi * (mlts[i] - 6) / 12
        polar_ax.plot([theta, theta], [10, 70], 'w--', alpha=.5)
        profile = data.interp(mlon=mlons[i], method='nearest')
        line_axs[i].plot(profile.mlat.values, profile.values, '.')
        mask = profile.notnull().values
        if mask.sum() < 5:
            continue
        spl = UnivariateSpline(profile.mlat.values[mask], profile.values[mask], k=4)
        spl.set_smoothing_factor(4)
        line_axs[i].plot(profile.mlat.values[mask], spl(profile.mlat.values[mask]), '--')


########################################################################################################################
# SATELLITES ###########################################################################################################
########################################################################################################################
dmsp_colors = {15: 'r', 16: 'b', 17: 'g', 18: 'y'}
swarm_colors = {'A': 'k', 'B': 'm', 'C': 'c'}


def plot_trough_locations(ax, date, data_dict, dt=np.timedelta64(60, 'm'), satellites=None):
    if satellites is None:
        satellites = list(data_dict.keys())
    for sat in satellites:
        if sat in dmsp_colors:
            c = dmsp_colors[sat]
        elif sat in swarm_colors:
            c = swarm_colors[sat]
        else:
            c = ''
        date_mask = (data_dict[sat].time.values >= date - dt / 2) * (data_dict[sat].time.values <= date + dt / 2)
        if not date_mask.any().item():
            return
        sat_data = data_dict[sat].isel(time=date_mask)
        if isinstance(ax, PolarAxes):
            x = sat_data['mlt'].values
            y = sat_data['min'].values
            x[x > 12] -= 24
            theta = np.pi * (x - 6) / 12
            r = 90 - y
            ax.plot(theta, r, f'{c}x', ms=10)
        else:
            ax.plot(sat_data['mlon'], sat_data['min'], f'{c[sat]}x', ms=10)


def plot_ion_drift(ax, date, dataset, dt=np.timedelta64(60, 'm'), satellites=None, min_mlat=50, hemisphere='north'):
    """
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018EA000546
    """
    data = dataset.sel(time=slice(date - dt / 2, date + dt / 2))
    if satellites is None:
        satellites = data.satellite.values
    if 'Viy' in data:
        drift = xr.where(data['Quality_flags'] == 4, data['Viy'], 0)
        colors = swarm_colors
    else:
        drift = data['hor_ion_v']
        colors = dmsp_colors
    plotted = False
    if drift.size == 0:
        return
    for sat in satellites:
        sat_data = data.sel(satellite=sat).coarsen(time=10, boundary='trim').mean()
        sat_drift = drift.sel(satellite=sat).coarsen(time=10, boundary='trim').mean()
        if hemisphere == 'north':
            mask = sat_data['mlat'] >= min_mlat
            if mask.sum() == 0:
                continue
            r = 90 - sat_data['mlat'].values[mask]
            z = np.column_stack((np.zeros((r.shape[0] - 1, 2)), -1 * np.ones(r.shape[0] - 1)))
        elif hemisphere == 'south':
            mask = -1 * sat_data['mlat'] >= min_mlat
            if mask.sum() == 0:
                continue
            r = 90 + sat_data['mlat'].values[mask]
            z = np.column_stack((np.zeros((r.shape[0] - 1, 2)), np.ones(r.shape[0] - 1)))
        else:
            raise Exception("hemisphere must be south or north")
        theta = (np.pi * sat_data['mlt'].values[mask] / 12) - np.pi / 2  # zero at bottom
        xy = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)))
        v = np.cross(z, np.diff(xy, axis=0))
        v = sat_drift.values[mask][:-1, None] * v / np.linalg.norm(v, axis=1)[:, None]
        q = ax.quiver(theta[:-1], r[:-1], v[:, 0], v[:, 1], units='xy', width=.05, color=colors[sat], scale=300)
        plotted = True
    if plotted:
        ax.quiverkey(q, .9, .9, 500, "500 m/s")


def plot_sat_location(ax, date, dataset, dt=np.timedelta64(60, 'm'), xbounds=(-120, 20), ybounds=(20, 80),
                       satellites=None):
    data = dataset.sel(time=slice(date-dt/2, date+dt/2))
    if satellites is None:
        satellites = data.satellite.values
    for sat in satellites:
        if sat in dmsp_colors:
            c = dmsp_colors[sat]
        elif sat in swarm_colors:
            c = swarm_colors[sat]
        else:
            c = ''
        sat_data = data.sel(satellite=sat).coarsen(time=10, boundary='trim').mean()
        if isinstance(ax, PolarAxes):
            x = sat_data['mlt'].values
            y = sat_data['mlat'].values
            mask = y > 30
            x[x > 12] -= 24
            theta = np.pi * (x - 6) / 12
            r = 90 - y
            ax.plot(theta[mask], r[mask], f'{c}.', ms=1)
        else:
            mask = ((sat_data['mlon'] > xbounds[0]) * (sat_data['mlon'] < xbounds[1]) *
                    (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
            ax.plot(sat_data['mlon'][mask], sat_data['mlat'][mask], f'{dmsp_colors[sat]}.', ms=2)


def plot_ne(ax, date, dataset, dt=np.timedelta64(60, 'm'), satellites=None, min_mlat=30):
    data = dataset.sel(time=slice(date - dt / 2, date + dt / 2))
    if satellites is None:
        satellites = data.satellite.values
    if 'n' in data:
        ne = data['n']
        colors = swarm_colors
    else:
        ne = data['ne']
        colors = dmsp_colors
    for sat in satellites:
        sat_data = data.sel(satellite=sat).coarsen(time=4, boundary='trim').mean()
        sat_ne = ne.sel(satellite=sat).coarsen(time=4, boundary='trim').mean()
        x = sat_data['mlat'].values
        y = sat_ne.values
        mask = x > min_mlat
        if isinstance(ax, plt.Axes):
            ax.plot(x[mask], y[mask], f'{colors[sat]}.')
        elif isinstance(ax, list):
            mlt = sat_data['mlt'].values[mask]
            mlt[mlt > 12] -= 24
            m = (-12 <= mlt) * (mlt <= -6)
            ax[0][0].plot(x[mask][m], y[mask][m], f'{colors[sat]}.')
            m = (6 <= mlt) * (mlt <= 12)
            ax[0][1].plot(x[mask][m], y[mask][m], f'{colors[sat]}.')
            m = (-6 <= mlt) * (mlt <= 0)
            ax[1][0].plot(x[mask][m], y[mask][m], f'{colors[sat]}.')
            m = (0 <= mlt) * (mlt <= 6)
            ax[1][1].plot(x[mask][m], y[mask][m], f'{colors[sat]}.')


def plot_dmsp_hor_ion_v_mag(ax, date, dmsp, dt=np.timedelta64(60, 'm'), xbounds=(-120, 20), ybounds=(20, 80),
                            vmin=-1000, vmax=1000):
    data = dmsp.sel(time=slice(date-dt/2, date+dt/2))
    for sat in data.sat_id.values:
        sat_data = data.sel(sat_id=sat)
        mask = ((sat_data['mlong'] > xbounds[0]) * (sat_data['mlong'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        ax.scatter(sat_data['mlong'][mask], sat_data['mlat'][mask], s=2, c=sat_data['hor_ion_v'][mask], cmap='jet', vmin=vmin, vmax=vmax)


def plot_dmsp_hor_ion_v_timeseries(ax, date, dmsp, dt=np.timedelta64(60, 'm'), xbounds=(-125, 25), ybounds=(15, 85),
                                   mlt_ax=None):
    data = dmsp.sel(time=slice(date-dt/2, date+dt/2))
    for sat in data.satellite.values:
        sat_data = data.sel(satellite=sat)
        mask = ((sat_data['mlon'] > xbounds[0]) * (sat_data['mlon'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        ax.plot(sat_data['mlat'][mask], sat_data['hor_ion_v'][mask], f'{dmsp_colors[sat]}.', label=f'DMSP-F{sat}', ms=1)


def plot_swarm_location_mag(ax, date, swarm, dt=np.timedelta64(60, 'm'), xbounds=(-180, 180), ybounds=(20, 80),
                            satellites=None):
    data = swarm.sel(time=slice(date-dt/2, date+dt/2))
    if satellites is None:
        satellites = data.satellite.values
    for sat in satellites:
        sat_data = data.sel(satellite=sat)
        mask = ((sat_data['mlon'] > xbounds[0]) * (sat_data['mlon'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        mask *= sat_data['Quality_flags'] == 4
        ax.plot(sat_data['mlon'][mask], sat_data['mlat'][mask], f'{swarm_colors[sat]}.', ms=1)


def plot_swarm_timeseries(ax, date, swarm, dt=np.timedelta64(60, 'm'), xbounds=(-125, 25), ybounds=(20, 80),
                          satellites=None):
    data = swarm.sel(time=slice(date - dt / 2, date + dt / 2))
    if satellites is None:
        satellites = data.satellite.values
    for sat in satellites:
        sat_data = data.sel(satellite=sat)
        mask = ((sat_data['mlon'] > xbounds[0]) * (sat_data['mlon'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        mask *= sat_data['Quality_flags'] == 4
        ax.plot(sat_data['mlat'][mask], sat_data['Viy'][mask], f'{swarm_colors[sat]}.', label=f'SWARM-{sat}', ms=1)


def plot_rx_locations_mag(ax, sites, coord_sys='apex'):
    converter = apexpy.Apex(date=utils.timestamp_to_datetime(sites.timestamps.values[0]))
    mlat, mlon = converter.convert(sites['gdlatr'].values[0], sites['gdlonr'].values[0], 'geo', coord_sys)
    ax.plot(mlon, mlat, 'k.', ms=.5)


def plot_sv_ipp_geo(ax, date, rx_loc_ecef, sat_pos, ipp_height=350., dt=None, min_el=0, **plot_kwargs):
    svs = np.unique(sat_pos.sv.values)
    if len(svs) != 1:
        raise Exception("Only input 1 satellite")
    sv = svs[0]
    cache_key = f'plot_sv_ipp__pierce_points_{sv}'
    if _cache[cache_key] is None:
        ipp = gps.get_pierce_points(sat_pos, rx_loc_ecef, ipp_height)
        _cache[cache_key] = ipp
    else:
        ipp = _cache[cache_key]

    if dt is None:
        ipp = ipp.interp(time=date, method='nearest')
        pos = sat_pos.interp(time=date, method='nearest')
    else:
        time_range = slice(date - dt/2, date + dt/2)
        ipp = ipp.sel(time=time_range)
        pos = sat_pos.sel(time=time_range)

    # convert ipp to lla
    ipp_lat, ipp_lon, ipp_alt = pm.ecef2geodetic(ipp.sel(component='x').values,
                                                 ipp.sel(component='y').values,
                                                 ipp.sel(component='z').values)

    # elevation mask
    rxlat, rxlon, rxalt = pm.ecef2geodetic(rx_loc_ecef[0], rx_loc_ecef[1], rx_loc_ecef[2])
    az, el, r = pm.ecef2aer(pos.sel(component='x').values, pos.sel(component='y').values, pos.sel(component='z').values,
                            rxlat[0], rxlon[0], rxalt[0])
    el_mask = el > min_el
    x = ipp_lon[el_mask]
    y = ipp_lat[el_mask]

    if isinstance(ax, GeoAxes):
        plot_kwargs.update(transform=crs.PlateCarree())
    ax.plot(x, y, **plot_kwargs)


def plot_sv_ipp_mag(ax, date, rx_loc_lla, sat_pos, ipp_height=350., dt=None, min_el=0, coord_sys='apex', **plot_kwargs):
    svs = np.unique(sat_pos.sv.values)
    if len(svs) != 1:
        raise Exception("Only input 1 satellite")
    sv = svs[0]

    rx_loc_ecef = pm.geodetic2ecef(rx_loc_lla[0], rx_loc_lla[1], rx_loc_lla[2])

    cache_key = f'plot_sv_ipp__pierce_points_{sv}'
    if cache_key not in _cache:
        ipp = gps.get_pierce_points(sat_pos, np.array(rx_loc_ecef), ipp_height)
        _cache[cache_key] = ipp.copy()
    else:
        ipp = _cache[cache_key].copy()

    if dt is None:
        ipp = ipp.interp(time=date, method='nearest')
        pos = sat_pos.interp(time=date, method='nearest')
    else:
        time_range = slice(date - dt/2, date + dt/2)
        ipp = ipp.sel(time=time_range)
        pos = sat_pos.sel(time=time_range)

    # convert ipp to lla
    ipp_lat, ipp_lon, ipp_alt = pm.ecef2geodetic(ipp.sel(component='x').values,
                                                 ipp.sel(component='y').values,
                                                 ipp.sel(component='z').values)

    # elevation mask
    az, el, r = pm.ecef2aer(pos.sel(component='x').values, pos.sel(component='y').values, pos.sel(component='z').values,
                            rx_loc_lla[0], rx_loc_lla[1], rx_loc_lla[2])
    el_mask = el > min_el

    x = ipp_lon[el_mask]
    y = ipp_lat[el_mask]
    converter = apexpy.Apex(date=utils.datetime64_to_datetime(ipp.time.values[0]))
    if coord_sys == 'mlt':
        y, x = convert.geo_to_mlt(y, x, ipp.time, converter=converter)
        x[x > 12] -= 24
    elif coord_sys == 'apex':
        y, x = converter.convert(y, x, 'geo', coord_sys, precision=-1)
    else:
        raise Exception("bad coordinate system")
    rxmlat, rxmlon = converter.convert(rx_loc_lla[0], rx_loc_lla[1], 'geo', coord_sys,
                                       datetime=utils.datetime64_to_datetime(ipp.time.values[0]))

    ax.plot(x, y, **plot_kwargs)
    ax.plot(rxmlon, rxmlat, 'wx')


def plot_tec_timeseries(self, ax, date, dt, svs=None, rxs=None, min_el=0, **plot_kwargs):
    if self.vtec is not None:
        tec = self.vtec
    elif self.stec is not None:
        tec = self.stec
    else:
        raise Exception("RxArray needs TEC first")
    if svs is None:
        svs = tec.sv.values
    if rxs is None:
        rxs = tec.rx.values
    time_range = slice(date - dt / 2, date + dt / 2)
    for rx in rxs:
        for sv in svs:
            el = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='el')
            t = tec.sel(time=time_range, sv=sv, rx=rx)
            good_data_mask = el.notnull() * (el > min_el) * t.notnull()
            t = t[good_data_mask]
            if t.size == 0:
                continue
            ax.plot(t.time.values, t.values, **plot_kwargs)


def plot_ipp(self, ax, date, dt, mag_coords=None, svs=None, rxs=None, min_el=0, **plot_kwargs):
    """plots ionospheric pierce points over a time range
    TODO:
        - sv / rx specific colors

    Parameters
    ----------
    ax: Matplotlib Axes or cartopy GeoAxes
            axes on which to plot
    date: np.datetime64
            (center) date which the plot represents
    dt: np.timedelta64
            width of time window to plot
    mag_coords: str {'apex', 'mlt'}
            magnetic coordinate system to convert to (can't specify this with cartopy GeoAxes)
    svs, rxs: list
            list of svs and rxs to plot
    min_el: float
            minimum elevation to limit points to
    plot_kwargs:
            kwargs to pass to ax.plot
    """
    if isinstance(ax, GeoAxes) and mag_coords is not None:
        raise Exception("Can't plot magnetic coordinates on a GeoAxes")
    if svs is None:
        svs = self.pierce_points.sv.values
    if rxs is None:
        rxs = self.pierce_points.rx.values
    if mag_coords is not None:
        apex_converter = apexpy.Apex(date=utils.datetime64_to_datetime(date))
    time_range = slice(date - dt/2, date+dt/2)
    for rx in rxs:
        for sv in svs:
            x = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='lon')
            y = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='lat')
            el = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='el')
            good_data_mask = x.notnull() * y.notnull() * el.notnull() * (el > min_el)
            x = x[good_data_mask]
            y = y[good_data_mask]
            if x.size == 0:
                continue
            if isinstance(ax, GeoAxes):
                plot_kwargs.update(transform=crs.PlateCarree())
            if mag_coords == 'apex':
                y, x = apex_converter.convert(y, x, 'geo', mag_coords)
            elif mag_coords == 'mlt':
                y, x = convert.geo_to_mlt(y, x, x.time, converter=apex_converter)
                x[x > 12] = x[x > 12] - 24
            ax.plot(x, y, '.', **plot_kwargs)


def plot_tec_ipp(self, ax, date, dt, mag_coords=None, svs=None, rxs=None, min_el=0, **plot_kwargs):
    """plots ionospheric pierce points colored with TEC over a time range

    Parameters
    ----------
    ax: Matplotlib Axes or cartopy GeoAxes
            axes on which to plot
    date: np.datetime64
            (center) date which the plot represents
    dt: np.timedelta64
            width of time window to plot
    mag_coords: str {'apex', 'mlt'}
            magnetic coordinate system to convert to (can't specify this with cartopy GeoAxes)
    svs, rxs: list
            list of svs and rxs to plot
    min_el: float
            minimum elevation to limit points to
    plot_kwargs:
            kwargs to pass to ax.plot
    """
    if self.vtec is not None:
        tec = self.vtec
    elif self.stec is not None:
        tec = self.stec
    else:
        raise Exception("RxArray needs TEC first")
    if isinstance(ax, GeoAxes) and mag_coords is not None:
        raise Exception("Can't plot magnetic coordinates on a GeoAxes")
    if svs is None:
        svs = tec.sv.values
    if rxs is None:
        rxs = tec.rx.values
    if mag_coords is not None:
        apex_converter = apexpy.Apex(date=utils.datetime64_to_datetime(date))
    time_range = slice(date - dt / 2, date + dt / 2)
    for rx in rxs:
        for sv in svs:
            x = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='lon')
            y = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='lat')
            el = self.pierce_points.sel(time=time_range, sv=sv, rx=rx, component='el')
            t = tec.sel(time=time_range, sv=sv, rx=rx)
            good_data_mask = x.notnull() * y.notnull() * el.notnull() * (el > min_el) * t.notnull()
            x = x[good_data_mask]
            y = y[good_data_mask]
            t = t[good_data_mask]
            if x.size == 0:
                continue
            if isinstance(ax, GeoAxes):
                plot_kwargs.update(transform=crs.PlateCarree())
            if mag_coords == 'apex':
                y, x = apex_converter.convert(y, x, 'geo', mag_coords)
            elif mag_coords == 'mlt':
                y, x = convert.geo_to_mlt(y, x, x.time, converter=apex_converter)
                x[x > 12] = x[x > 12] - 24
            ax.scatter(x, y, c=t.values, **plot_kwargs)
