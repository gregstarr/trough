
class DataSource:

    def __init__(self):
        self.plots = []  # ax: method

    def add_plot(self, ax, method, *args, **kwargs):
        self.plots.append((ax, method, args, kwargs))

    def plot_date(self, date):
        for ax, method, args, kwargs in self.plots:
            method(ax, date, *args, **kwargs)
