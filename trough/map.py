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


class MapPlot:
    """
    The purpose of this is to make it easy to make a time series of plots by separating the general plot aesthetics
    from the data. The plot aesthetics will all be determined (if desired) in a separate yaml file. I'm thinking the
    workflow will be as follows:
        - configure yaml file
        - create MP object from yaml
        - MP.create_ax() or equivelant called in __init__
        - maybe something like MP.load_data ? this will load a time series of data to be plotted
        - then you could call MP.plot_date or MP.plot_index which will set the ax to the proper date (magnetic
        coordinate lines, title, etc.)
        - it would be cool to have some integration with the multifile array thing I was working on so that you can
        define some directory structure file pattern which will automatically load the various files as necessary
        during iteration
        - should keep track of what plot features are date-dependent and have a function which updates each one
    """

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
        self.ax = self._create_projection_axes()
        background_color = self.params['general']['background_color']
        self.ax.background_patch.set_facecolor(background_color)
        self._add_natural_earth_features()
        self._add_grid()

        self.updates = []
        # register update funcs
        for feature_name, feature_class in DATE_FEATURES.items():
            if self.params[feature_name]['enable']:
                feature = feature_class(self.params[feature_name])
                self.updates.append(feature)

    def plot_date(self, date):
        for feature in self.updates:
            feature.update(self.ax, date)
        self.ax.set_title(str(date))

    def save_fig(self, directory, date, name=None):
        date_str = str(date).replace('-', '').replace(':', '')
        if name is not None:
            file_name = f"{date_str}_{name}.png"
        else:
            file_name = date_str + ".png"
        save_path = os.path.join(directory, file_name)
        plt.savefig(save_path)

    def _create_projection_axes(self):
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
