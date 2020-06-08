import abc
import apexpy as ap
import numpy as np
import cartopy.crs as ccrs

import trough


MAGNETIC_COORDINATE_SYSTEMS = {
    'mlt': {
        'range': (0, 24)
    },
    'apex': {
        'range': (0, 360)
    }
}


class DateFeature(abc.ABC):
    """
    This will keep track of map features that need to be updated with the date. This class will have a function
    to add the lines or data or whatever to the plot, it will then keep references to the objects returned by
    ax.plot or ax.whatever so that they can be removed by a cleanup function.
    """

    def __init__(self, params):
        self.params = params
        self.plotted_objects = []

    def update(self, ax, date):
        if len(self.plotted_objects) > 0:
            self._clear(ax)
        self._add(ax, date)

    @abc.abstractmethod
    def _add(self, ax, date):
        pass

    @abc.abstractmethod
    def _clear(self, ax):
        pass


class LineDateFeature(DateFeature, abc.ABC):

    def _clear(self, ax):
        plotted_objects_copy = self.plotted_objects.copy()
        for obj in plotted_objects_copy:
            ax.lines.remove(obj)
            self.plotted_objects.remove(obj)
        del plotted_objects_copy


class MagneticCoordinates(LineDateFeature):

    def _add(self, ax, date):
        # get parameters, set defaults
        coord_sys = self.params['coordinate_system']
        mlon_levels = self.params['xlocs']
        mlat_levels = self.params['ylocs']
        height = self.params['height']
        line_color = self.params['line_color']
        line_width = self.params['line_width']
        line_style = self.params['line_style']
        mag_coord_lines = trough.utils.get_magnetic_coordinate_lines(date, coord_sys, height, mlat_levels, mlon_levels)
        for level, lines in mag_coord_lines['mlon'].items():
            for line in lines:
                self.plotted_objects += ax.plot(line[:, 0], line[:, 1], c=line_color, lw=line_width, ls=line_style,
                                                zorder=90, transform=ccrs.PlateCarree())
        for level, lines in mag_coord_lines['mlat'].items():
            for line in lines:
                self.plotted_objects += ax.plot(line[:, 0], line[:, 1], c=line_color, lw=line_width, ls=line_style,
                                                zorder=90, transform=ccrs.PlateCarree())


class Terminator(LineDateFeature):

    def _add(self, ax, date):
        if isinstance(date, np.datetime64):
            date = trough.utils.datetime64_to_datetime(date)
        altitude = self.params['altitude']
        line_color = self.params['line_color']
        line_width = self.params['line_width']
        line_style = self.params['line_style']
        term_lons, term_lats = trough.utils.get_terminator(date, alt_km=altitude)
        for lon, lat in zip(term_lons, term_lats):
            self.plotted_objects += ax.plot(np.unwrap(lon, 180), np.unwrap(lat, 90), c=line_color,
                                            lw=line_width, ls=line_style, zorder=90, transform=ccrs.PlateCarree())


class TimeAverageTecMap(DateFeature):

    def __init__(self, params, data):
        super().__init__(params)
        self.data = data

    def _add(self, ax, date):
        vmin = self.params.get('vmin', 0)
        vmax = self.params.get('vmax', 30)
        cmap = self.params.get('cmap', 'jet')
        average = self.params.get('average')
        if average:
            date_select = slice(date - average/2, date + average/2)
            current_date_data = self.data.sel(time=date_select).mean(dim='time')
        else:
            current_date_data = self.data.sel(time=date)
        self.plotted_objects.append(current_date_data.plot.pcolormesh(ax=ax, x='longitude', y='latitude',
                                                                      transform=ccrs.PlateCarree(), vmin=vmin,
                                                                      vmax=vmax, cmap=cmap, add_colorbar=False))

    def _clear(self, ax):
        ax.collections.remove(self.plotted_objects[0])
        self.plotted_objects.remove(self.plotted_objects[0])
