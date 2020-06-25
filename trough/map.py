import yaml
import os
import numpy as np
from . import date_features
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt


DEFAULT_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), 'defaults.yaml'))


projection_dict = {
    'stereographic': ccrs.Stereographic,
    'mercator': ccrs.Mercator,
    'plate_carree': ccrs.PlateCarree,
    'lambert_conformal': ccrs.LambertConformal,
    'mollweide': ccrs.Mollweide,
    'north_polar_stereo': ccrs.NorthPolarStereo,
    'south_polar_stereo': ccrs.SouthPolarStereo,
    'orthographic': ccrs.Orthographic,
    'nearside_perspective': ccrs.NearsidePerspective
}

NATURAL_EARTH_FEATURES = {
    'borders': cfeat.BORDERS,
    'coastline': cfeat.COASTLINE,
    'lakes': cfeat.LAKES,
    'land': cfeat.LAND,
    'ocean': cfeat.OCEAN,
    'rivers': cfeat.RIVERS,
    'states': cfeat.STATES
}

DATE_FEATURES = {
    'magnetic_coordinates': date_features.MagneticCoordinates,
    'terminator': date_features.Terminator
}


class DatePlot:

    def __init__(self, config=None):
        with open(DEFAULT_CONFIG, 'r') as f:
            self.params = yaml.safe_load(f)

        if config is not None:
            if isinstance(config, dict):
                self.params.update(config)
            elif isinstance(config, str):
                with open(config, 'r') as f:
                    user_settings = yaml.safe_load(f)
                    self.params.update(user_settings)

        # create figure and axes
        figsize = self.params['general']['figsize']
        self.fig = plt.figure(figsize=figsize)
        self.ax = self._create_axes()
        background_color = self.params['general']['background_color']
        self.ax.background_patch.set_facecolor(background_color)
        if self.params['projection']['enable']:
            self._add_natural_earth_features()
        self._add_grid()

        self.updates = []
        # register update funcs
        for feature_name, feature_class in DATE_FEATURES.items():
            if self.params[feature_name]['enable']:
                feature = feature_class(self.params[feature_name])
                self.updates.append(feature)

    def plot_date_range(self, date_range):
        for feature in self.updates:
            feature.update(self.ax, date_range)
        self.ax.set_title(f"{str(date_range.start)[:19]} - {str(date_range.stop)[:19]}")

    def save_fig(self, directory, date, name=None):
        date_str = str(date).replace('-', '').replace(':', '')[:15]
        if name is not None:
            file_name = f"{date_str}_{name}.png"
        else:
            file_name = date_str + ".png"
        save_path = os.path.join(directory, file_name)
        plt.savefig(save_path, fig=self.fig)

    def _create_axes(self):
        if not self.params['projection']['enable']:
            ax = self.fig.add_subplot(1, 1, 1)
            return ax
        name = self.params['projection']['name']
        if name not in projection_dict:
            raise KeyError(f"Projection is invalid. Please choose from:\n{', '.join(projection_dict.keys())}")
        kwargs = self.params['projection']['kwargs']
        if kwargs is None:
            kwargs = {}
        proj = projection_dict[name](**kwargs)
        ax = self.fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_global()
        return ax

    def _add_natural_earth_features(self):
        for feature_name, feature_dict in self.params['natural_earth_features'].items():
            if feature_dict['enable']:
                feature = NATURAL_EARTH_FEATURES[feature_name]
                kwargs = feature_dict['kwargs']
                if kwargs is None:
                    kwargs = {}
                self.ax.add_feature(feature, **kwargs)

    def _add_grid(self):
        enable = self.params['grid'].get('enable')
        if enable:
            kwargs = self.params['grid'].get('kwargs')
            if kwargs is None:
                kwargs = {}
            self.ax.gridlines(**kwargs)
