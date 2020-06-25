import georinex as gr
import numpy as np
from astropy import constants as ac
import xarray as xr
import os
import pymap3d as pm
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
import glob
import apexpy

from trough import utils
from trough.datasource import DataSource
from trough import convert


# GPS L1 and L2 frequencies
F1 = 1575420000
F2 = 1227600000


class ReceiverArray(DataSource):
    """
    I want this to be able to give me TEC and pierce points for any satellite and all the receivers. This should be
    an xarray.DataSet who's DataArrays correspond to the various receivers. Within each DataArray, there will be the
    variables: TEC, IPP and any other data aligned with the GPS samples.

    The receiver array shares several data and parameter files including satellite orbital parameters and satellite
    bias.
    """
    def __init__(self):
        super().__init__()
        self.obs = None
        self.rx_position = None
        self.stec = None
        self.pierce_points = None
        self.ipp_h = None
        self.vtec = None

        save_name = "E:\\tec_data\\data\\mahali\\satellite_positions.nc"
        obs_files, nav_files = utils.get_mahali_files()

        self.update_obs(obs_files)
        self.update_tec()
        # get satellite positions
        if os.path.isfile(save_name):
            self.satellite_positions = xr.load_dataarray(save_name)
        else:
            nav_params = open_and_merge_nav_files(nav_files.values())
            self.satellite_positions = get_satellite_positions(nav_params, times=self.obs.time.values)
            print(f"Saving: {save_name}")
            self.satellite_positions.to_netcdf(save_name)

        # pierce points
        self.update_pierce_points()

    def update_obs(self, obs_files):
        save_name = "E:\\tec_data\\data\\mahali\\{}_obs.nc"
        obs_list = []
        for rx_name, files in obs_files.items():
            rx_save_name = save_name.format(rx_name)
            if os.path.isfile(rx_save_name):
                obs_list.append(xr.open_dataset(rx_save_name))
            else:
                obs = open_and_merge_obs_files(files, rx_name=rx_name)
                print(f"Saving: {rx_save_name}")
                obs.to_netcdf(rx_save_name)
                obs_list.append(obs)
        self.rx_position = xr.DataArray(np.array([(o.position, o.position_geodetic) for o in obs_list]).reshape((-1, 6)),
                                        dims=['rx', 'component'],
                                        coords={
                                            'rx': [rx for rx in obs_files],
                                            'component': ['x', 'y', 'z', 'lat', 'lon', 'alt']
                                        })

        self.obs = xr.concat(obs_list, dim='rx')

    def update_tec(self):
        tec = []
        for rx in self.obs.rx.values:
            tec.append(_tec_rx(self.obs.sel(rx=rx)))
        self.stec = xr.concat(tec, dim='rx')

    def update_pierce_points(self, pp_alt=350.):
        self.ipp_h = pp_alt
        ipp = []
        for rx in self.obs.rx.values:
            ipp.append(self._get_pierce_points_rx(rx, pp_alt))
        self.pierce_points = xr.concat(ipp, dim='rx')

    def _get_pierce_points_rx(self, rx, pp_alt):
        save_name = f"E:\\tec_data\\data\\mahali\\{rx}_{int(round(pp_alt)):04d}km_ipp.nc"
        if os.path.isfile(save_name):
            return xr.open_dataarray(save_name)
        print(rx)
        ds = self.stec.sel(rx=rx)
        times = ds.time.where(ds.notnull().any(dim='sv'), drop=True).values
        print("ECEF")
        ecef = get_pierce_points(self.satellite_positions.sel(time=times),
                                 self.rx_position.sel(rx=rx, component=['x', 'y', 'z']), pp_alt)
        ecef2geo = conversion_wrapper(pm.ecef2geodetic, ['x', 'y', 'z'], ['lat', 'lon', 'alt'])
        ecef2aer = conversion_wrapper(pm.ecef2aer, ['x', 'y', 'z'], ['az', 'el', 'r'])
        print("GEO")
        geo = ecef.groupby('sv').map(ecef2geo)
        print("AER")
        aer = ecef.groupby('sv').map(ecef2aer, args=tuple(self.rx_position.sel(rx=rx, component=['lat', 'lon', 'alt']).values))
        pierce_points = xr.concat([ecef, geo, aer], dim='component')
        print(f"Saving: {save_name}")
        pierce_points.to_netcdf(save_name)
        return pierce_points

    def mapping_func(self):
        rc1 = 6371.0 / (6371.0 + self.ipp_h)
        return np.sqrt(1.0 - (np.cos(np.radians(self.pierce_points.sel(component='el'))) ** 2 * rc1 ** 2))

    @property
    def obs_time_range(self):
        return slice(self.obs.time.values[0], self.obs.time.values[-1])

    def align_with_madrigal_tec_map(self, tec_map):
        save_name = f"E:\\tec_data\\data\\mahali\\madrigal_aligned_vtec.nc"
        if os.path.isfile(save_name):
            self.vtec = xr.open_dataarray(save_name)
            return
        vtec = []
        for sv in self.stec.sv.values:
            pierce_points = self.pierce_points.sel(sv=sv, component=['lat', 'lon'])
            mapping = self.mapping_func().sel(sv=sv)
            mahali = self.stec.sel(sv=sv)
            sv_vtec = []
            for i, rx in enumerate(pierce_points.rx.values):
                print(rx)
                ipp = pierce_points.sel(rx=rx)
                mask = ipp.notnull().all(dim='component')
                ipp = ipp[mask]
                mah = mahali.sel(rx=rx)[mask]
                M = mapping.sel(rx=rx)[mask]
                interp_mad = tec_map.interp(time=ipp.time, latitude=ipp.sel(component='lat'),
                                            longitude=ipp.sel(component='lon'), method='nearest')
                interp_mad /= M
                mah += (interp_mad - mah).mean(dim='time')
                mah -= np.minimum(mah.min(dim='time'), 0)
                mah *= M
                sv_vtec.append(mah)
            vtec.append(xr.concat(sv_vtec, dim='rx'))
        self.vtec = xr.concat(vtec, dim='sv')
        self.vtec.to_netcdf(save_name)

    def plot_tec_timeseries(self, ax, date):
        pass

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
                if isinstance(ax, GeoAxes):
                    plot_kwargs.update(transform=ccrs.PlateCarree())
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
                if isinstance(ax, GeoAxes):
                    plot_kwargs.update(transform=ccrs.PlateCarree())
                if mag_coords == 'apex':
                    y, x = apex_converter.convert(y, x, 'geo', mag_coords)
                elif mag_coords == 'mlt':
                    y, x = convert.geo_to_mlt(y, x, x.time, converter=apex_converter)
                    x[x > 12] = x[x > 12] - 24
                ax.scatter(x, y, c=t.values, **plot_kwargs)


def _tec_rx(obs):
    """This function gets biased slant TEC for all satellites for a single receiver. This aligns the phase TEC to
    the pseudorange TEC.

    Parameters
    ----------
    obs: xarray.Dataset
            observation data found in a rinex OBS file

    Returns
    -------
    biased_tec: xarray.DataArray
            slant tec
    """
    code_tec = 2.85e9 * (obs['C2'] - obs['C1']) / ac.c.to('m/s').value
    phase_tec = 2.85e9 * (obs['L1']/F1 - obs['L2']/F2)
    good_svs = (code_tec.notnull() * phase_tec.notnull()).sum(dim='time') > 60*60
    separate_tec = xr.Dataset({'code': code_tec.where(good_svs, drop=True),
                               'phase': phase_tec.where(good_svs, drop=True)})
    biased_tec = separate_tec.groupby('sv').map(_single_sat_tec)
    return biased_tec


def _single_sat_tec(da):
    """This function gets unprocessed (biased) slant TEC for a single satellite for a receiver

    Parameters
    ----------
    da: xarray.Dataset
            observation data found in a rinex OBS file for a single satellite e.g. obs.sel(sv='G31')

    Returns
    -------
    xarray.DataArray
            slant tec for a satellite
    """
    da = da.interpolate_na(dim='time', method='nearest', limit=5, use_coordinate=False)
    dt = da['phase'].where(da['phase'].notnull(), drop=True).time.dt
    sat_pass = dt.dayofyear + (dt.hour + dt.minute / 60 + dt.second / (60 * 60)) / 24
    sat_pass -= sat_pass.min()
    sat_pass //= 1
    da = da.assign_coords({'sat_pass': sat_pass})
    return da.groupby('sat_pass').map(_single_pass_tec)


def _single_pass_tec(ds):
    """This function gets unprocessed (biased) slant TEC for a single satellite pass

    Parameters
    ----------
    ds: xarray.Dataset
            observation data found in a rinex OBS file for a single satellite pass e.g. obs.sel(sv='G31', sat_pass=1)

    Returns
    -------
    xarray.DataArray
            slant tec for a satellite pass
    """
    diffs = ds['phase'].diff('time')
    for i in np.where(abs(diffs) > 1)[0]:
        ds['phase'][:i+1] += diffs[i]
    return ds['phase'] + (ds['code'] - ds['phase']).median()


########################################################################################################################
# COMPUTATION ##########################################################################################################
########################################################################################################################
def conversion_wrapper(func, input_components, output_components):
    """This wraps pymap3d's conversion functions to input and output xarray.DataArrays

    Parameters
    ----------
    func: callable
            the pymap3d conversion function to apply
    input_components, output_components: list
            list of 'component' names. 'component' must be a dimension of the DataArray

    Returns
    -------
    conversion function: callable
            the pymap3d conversion function suitable for DataArrays
    """
    def xarray_func(darr, *args):
        converted = func(darr.sel(component=input_components[0]).values,
                         darr.sel(component=input_components[1]).values,
                         darr.sel(component=input_components[2]).values,
                         *args)
        return xr.DataArray(np.column_stack(converted), dims=['time', 'component'],
                            coords={'time': darr.time.values, 'component': output_components})
    return xarray_func


def get_single_sat_pos(orbital_params, times=None, interp_res=60):
    """get a satellite's position in ECEF coordinates at specified times

    Parameters
    ----------
    orbital_params: xarray.DataArray
            Orbital parameters for the satellite. Should have 1 dimension, "time", and no missing values
    times: np.ndarray, optional
            times to get satellite position at
    interp_res: float, optional
            if `times` is None, then create time series from beginning to end of `orbital_params` at a resolution
            of `interp_res` seconds

    Returns
    -------
    X, Y, Z: np.ndarray
            satellite positions (ECEF)
    """
    if times is None:
        dt = np.timedelta64(interp_res, 's')
        times = np.arange(orbital_params.time.values[0], orbital_params.time.values[-1], dt)
    interpolated_params = orbital_params.interp(time=times, method='nearest')
    return xr.DataArray(np.column_stack(gr.keplerian2ecef(interpolated_params)), dims=['time', 'component'],
                        coords={'time': times, 'component': ['x', 'y', 'z']})


def get_satellite_positions(orbital_params, times=None, interp_res=60):
    """Get satellite positions for all satellites in a nav file

    Parameters
    ----------
    orbital_params
    times
    interp_res

    Returns
    -------

    """
    if times is None:
        dt = np.timedelta64(interp_res, 's')
        times = np.arange(orbital_params.time.values[0], orbital_params.time.values[-1], dt)
    positions = []
    for sv in orbital_params.sv.values:
        print(sv)
        p = get_single_sat_pos(orbital_params.sel(sv=sv), times=times)
        positions.append(p.expand_dims({'sv': [sv]}))
    return xr.concat(positions, dim='sv')


def get_pierce_points(arg, rx_location, pierce_point_altitude=350, times=None, interp_res=60):
    """get the ionospheric pierce points for a satellite using spherical approximation

    Parameters
    ----------
    arg: xarray.DataArray, one of the following
        orbital_params: xarray.DataArray
                Orbital parameters for the satellite. Should have 1 dimension, "time", and no missing values
        satellite_positions: xarray.DataArray
                ECEF coordinates of the satellite
    rx_location: iterable of floats
                X, Y, Z in ECEF (m)
    pierce_point_altitude: float, optional
            altitude of pierce points
    times: np.ndarray, optional
            times to get satellite position at
    interp_res: float, optional
            if `times` is None, then create time series from beginning to end of `orbital_params` at a resolution
            of `interp_res` seconds

    Returns
    -------
    Pierce Points: xarray.DataArray
    """
    nparray = isinstance(arg, np.ndarray)
    # satellite position in ECEF (km)
    if not nparray and "Eccentricity" in arg:
        sat_pos = get_single_sat_pos(arg, times, interp_res) / 1000.
    else:
        sat_pos = arg / 1000.
    r_e = ac.R_earth.to('km').value  # earth radius
    rx_loc = rx_location.copy() / 1000.  # receiver location
    r2_rx = np.sum(rx_loc**2)  # receiver
    if nparray:
        rx_sat_dot = sat_pos @ rx_loc
        a = r2_rx - 2 * rx_sat_dot + (sat_pos ** 2).sum(axis=1)
    else:
        rx_sat_dot = (sat_pos * rx_loc).sum(dim='component')
        a = r2_rx - 2 * rx_sat_dot + (sat_pos**2).sum(dim='component')
    b = 2 * (rx_sat_dot - r2_rx)
    c = r2_rx - (r_e + pierce_point_altitude)**2
    t = (-1 * b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    if nparray:
        ipp = (1 - t)[:, None] * rx_loc[None, :] + t[:, None] * sat_pos
    else:
        ipp = (1 - t) * rx_loc + t * sat_pos
    return ipp * 1000.


########################################################################################################################
# FILE IO ##############################################################################################################
########################################################################################################################
def open_obs(fn, rx_name=None):
    """open observation file and fill in missing values

    Parameters
    ----------
    fn: str
            obs file to open (.15o or .nc)

    Returns
    -------
    xarray.DataSet
            obs data
    """
    print(f"Opening: {fn}")
    dataset = gr.load(fn)
    if rx_name is not None:
        dataset = dataset.assign_coords({'rx': [rx_name]})
    return dataset


def open_nav(fn):
    """Open rinex nav file and fill in missing values

    Parameters
    ----------
    fn: str
            nav file to open (.15n or .nc)

    Returns
    -------
    xarray.DataSet
            nav data parameters
    """
    print(f"Opening: {fn}")
    dataset = gr.load(fn)
    for variable in dataset:
        if dataset[variable].isnull().any().item():
            dataset = dataset.interpolate_na(dim='time', fill_value='extrapolate', method='nearest')
            break
    return dataset


def open_and_merge_nav_files(files):
    """Opens a list of nav files, combining them along the 'time' axis

    Parameters
    ----------
    files: list
            file names (.15n)

    Returns
    -------
    navigation parameters: xarray.Dataset
    """
    nav_data = [open_nav(fn) for fn in files]
    return xr.concat(nav_data, dim='time')


def open_and_merge_obs_files(files, rx_name=None):
    obs_data = []
    for fn in files:
        try:
            obs_data.append(open_obs(fn, rx_name=rx_name))
        except Exception as e:
            print(f"Skipping: {fn}, error: {e}")
    return xr.concat(obs_data, dim='time')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    array = ReceiverArray()
