import numpy as np
import xarray as xr
import pymap3d as pm
import apexpy
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
                ax.plot(x[slc], y[slc], 'k', **plot_kwargs)


########################################################################################################################
# TEC MAP ##############################################################################################################
########################################################################################################################
def plot_tec_map_geo(ax, date, tec, y_arange_args=None, x_arange_args=None, dt=None, **plot_kwargs):
    # averaging and filling
    if dt is None:
        tec = tec.interp(time=date, method='nearest')
    else:
        time_range = slice(date - dt / 2, date + dt / 2)
        tec = tec.sel(time=time_range).mean(dim='time')
        binned = tec.coarsen(longitude=2, latitude=2).mean()
        binned.load()
        interpolated = binned.interp(longitude=tec.longitude, latitude=tec.latitude)
        tec = tec.where(tec.notnull(), interpolated)

    # setup bounds
    if x_arange_args is None:
        x_vals = np.arange(-180, 180)
    else:
        x_vals = np.arange(*x_arange_args)
    if y_arange_args is None:
        y_vals = np.arange(-90, 90)
    else:
        y_vals = np.arange(*y_arange_args)
    x, y = np.meshgrid(x_vals, y_vals)
    tec = tec.sel(longitude=x_vals, latitude=y_vals).T

    # plotting
    if isinstance(ax, GeoAxes):
        plot_kwargs.update(transform=crs.PlateCarree())
    ax.pcolormesh(x, y, tec, **plot_kwargs)


def plot_tec_map_mag(ax, date, tec, coord_sys='apex', y_arange_args=None, x_arange_args=None, dt=None, **plot_kwargs):
    # averaging and filling
    if dt is None:
        tec = tec.interp(time=date, method='nearest')
    else:
        time_range = slice(date - dt / 2, date + dt / 2)
        tec = tec.sel(time=time_range).mean(dim='time')
        binned = tec.coarsen(longitude=2, latitude=2).mean()
        binned.load()
        interpolated = binned.interp(longitude=tec.longitude, latitude=tec.latitude)
        tec = tec.where(tec.notnull(), interpolated)

    # bounds
    datetime = utils.datetime64_to_datetime(date)
    converter = apexpy.Apex(date=datetime)
    if x_arange_args is None:
        if coord_sys == 'apex':
            x_vals = np.arange(-180, 180)
        elif coord_sys == 'mlt':
            x_vals = np.arange(-12, 12, 24/360)
        else:
            raise Exception("Bad coordinate system must be apex or mlt")
    else:
        x_vals = np.arange(*x_arange_args)
    if y_arange_args is None:
        y_vals = np.arange(-90, 90)
    else:
        y_vals = np.arange(*y_arange_args)
    x, y = np.meshgrid(x_vals, y_vals)

    # coordinate conversion
    glat, glon = converter.convert(y.ravel(), x.ravel(), coord_sys, 'geo', datetime=datetime, precision=-1)
    glat_grid = glat.reshape(y.shape)
    glon_grid = glon.reshape(y.shape)
    lat = xr.DataArray(glat_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})
    lon = xr.DataArray(glon_grid, dims=["y", "x"], coords={"x": x_vals, "y": y_vals})
    tec = tec.interp(longitude=lon, latitude=lat)

    # plotting
    ax.pcolormesh(x, y, tec, **plot_kwargs)


########################################################################################################################
# SATELLITES ###########################################################################################################
########################################################################################################################
dmsp_colors = {15: 'r', 16: 'b', 17: 'g', 18: 'y'}


def plot_dmsp_hor_ion_v_mag(ax, date, dmsp, dt=np.timedelta64(60, 'm'), xbounds=(-120, 20), ybounds=(20, 80),
                            vmin=-1000, vmax=1000):
    data = dmsp.sel(time=slice(date-dt/2, date+dt/2))
    for sat in data.sat_id.values:
        sat_data = data.sel(sat_id=sat)
        mask = ((sat_data['mlong'] > xbounds[0]) * (sat_data['mlong'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        ax.scatter(sat_data['mlong'][mask], sat_data['mlat'][mask], s=2, c=sat_data['hor_ion_v'][mask], cmap='jet', vmin=vmin, vmax=vmax)


def plot_dmsp_location(ax, date, dmsp, dt=np.timedelta64(60, 'm'), xbounds=(-120, 20), ybounds=(20, 80)):
    data = dmsp.sel(time=slice(date-dt/2, date+dt/2))
    for sat in data.sat_id.values:
        sat_data = data.sel(sat_id=sat)
        mask = ((sat_data['mlong'] > xbounds[0]) * (sat_data['mlong'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        ax.plot(sat_data['mlong'][mask], sat_data['mlat'][mask], f'{dmsp_colors[sat]}.', ms=1)


def plot_dmsp_hor_ion_v_timeseries(ax, date, dmsp, dt=np.timedelta64(60, 'm'), xbounds=(-125, 25), ybounds=(15, 85),
                                   mlt_ax=None):
    data = dmsp.sel(time=slice(date-dt/2, date+dt/2))
    for sat in data.sat_id.values:
        sat_data = data.sel(sat_id=sat)
        mask = ((sat_data['mlong'] > xbounds[0]) * (sat_data['mlong'] < xbounds[1]) *
                (sat_data['mlat'] > ybounds[0]) * (sat_data['mlat'] < ybounds[1]))
        ax.plot(sat_data['mlat'][mask], sat_data['hor_ion_v'][mask], f'{dmsp_colors[sat]}-', label=f'DMSP-F{sat}')
        # ax.plot(sat_data['mlat'][mask], sat_data['vert_ion_v'][mask], f'{dmsp_colors[sat]}-.')
        if mlt_ax is not None:
            mlt_ax.plot(sat_data['mlat'][mask], sat_data['mlt'][mask], f'{dmsp_colors[sat]}--')


swarm_colors = {'A': 'k', 'B': 'm', 'C': 'c'}


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
        ax.plot(sat_data['mlat'][mask], sat_data['Viy'][mask], f'{swarm_colors[sat]}-', label=f'SWARM-{sat}')


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


class MagneticCoordinates:

    def get_magnetic_coordinate_lines(date, coord_sys='mlt', height=0, mlat_levels=None, mlon_levels=None,
                                      resolution=1):
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
        diff_mag = x_diff ** 2 + y_diff ** 2
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

    def _add(self, ax, date_range):
        # get parameters, set defaults
        coord_sys = self.params['coordinate_system']
        mlon_levels = self.params['xlocs']
        mlat_levels = self.params['ylocs']
        height = self.params['height']
        line_color = self.params['line_color']
        line_width = self.params['line_width']
        line_style = self.params['line_style']
        mag_coord_lines = utils.get_magnetic_coordinate_lines(date_range.start, coord_sys, height, mlat_levels, mlon_levels)
        for level, lines in mag_coord_lines['mlon'].items():
            for line in lines:
                self.plotted_objects += ax.plot(line[:, 0], line[:, 1], c=line_color, lw=line_width, ls=line_style,
                                                zorder=90, transform=ccrs.PlateCarree())
        for level, lines in mag_coord_lines['mlat'].items():
            for line in lines:
                self.plotted_objects += ax.plot(line[:, 0], line[:, 1], c=line_color, lw=line_width, ls=line_style,
                                                zorder=90, transform=ccrs.PlateCarree())


class Terminator:

    def get_terminator(self, time, alt_km=0, resolution=1):
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

        sun_el = utils.get_sun_elevation(time, glon, glat)
        r_e = aconst.R_earth.to('km').value
        horizon = np.rad2deg(np.arccos(r_e / (r_e + alt_km)))
        terminators = measure.find_contours(sun_el, -1 * horizon)
        x = []
        y = []
        for terminator in terminators:
            x.append(terminator[:, 1] - 180)
            y.append(terminator[:, 0] - 90)

        return x, y

    def _add(self, ax, date_range):
        date = date_range.start
        if isinstance(date, np.datetime64):
            date = trough.utils.datetime64_to_datetime(date_range)
        altitude = self.params['altitude']
        line_color = self.params['line_color']
        line_width = self.params['line_width']
        line_style = self.params['line_style']
        term_lons, term_lats = trough.utils.get_terminator(date, alt_km=altitude)
        for lon, lat in zip(term_lons, term_lats):
            self.plotted_objects += ax.plot(np.unwrap(lon, 180), np.unwrap(lat, 90), c=line_color,
                                            lw=line_width, ls=line_style, zorder=90, transform=ccrs.PlateCarree())


