import numpy as np
import matplotlib.pyplot as plt
import pandas
import time
import os
import h5py
from sklearn.metrics import confusion_matrix
from skimage.measure import label

from ttools import plotting, swarm, rbf_inversion
from teclab import config, utils


class TroughComparer:

    def __init__(self, save_dir):
        self.model = rbf_inversion.RbfIversion(config.mlt_grid, config.mlat_grid, ds=(1, 4))
        self.save_dir = save_dir
        self.tec_troughs = []
        self.years = []
        self.months = []
        self.indexes = []

    def compare_single(self, year, month, index, plot=False):
        """Compare a single trough

        steps:
            - run both trough detectors
            - for pixels near the satellite paths
            - degrees latitude swarm pwall above tec pwall
            - degrees latitude swarm ewall above tec ewall
            - if no trough in swarm, assume swarm trough pwall = ewall = middle of tec pwall and ewall
            - if no trough in tec, assume tec trough pwall = ewall = middle of swarm pwall and ewall
            - keep track of cases where trough mismatch
        """
        x, ut, tec = self.model.load_and_preprocess(year, month, index)
        tec_trough_model = self.model.run(x, ut)
        tec_trough_model = self.model.postprocess(tec_trough_model)
        tec_trough = self.model.decision(tec_trough_model)
        tec_trough = self.model.postprocess_labels(tec_trough)
        tec_trough_labels = label(tec_trough, connectivity=2)

        results = []
        swarm_paths = np.zeros_like(tec_trough, dtype=int)
        swarm_segments = []
        for sat in swarm.SWARM_SATELLITES:
            ewall_diff = 0
            pwall_diff = 0
            interval = swarm.SwarmDataInterval.create_and_process(year, month, index, sat)
            front, back, total = interval.get_closest_segments()
            for side_name, side in zip(['front', 'back'], [front, back]):
                swarm_segments.append(side)
                m = -1
                for l in range(tec_trough_labels.max()):
                    its = side.get_tec_trough_intersection(tec_trough_labels == l + 1, radius=20)
                    mp = its.sum()
                    if mp > m:
                        intersection = its
                        m = mp
                swarm_paths += side.path_mask
                side_tec_trough = intersection.any()
                if side_tec_trough:
                    tec_pwall = config.mlat_grid[intersection].max()
                    tec_ewall = config.mlat_grid[intersection].min()
                if side.trough and side_tec_trough:
                    pwall_diff = side.pwall_lat - tec_pwall
                    ewall_diff = side.ewall_lat - tec_ewall
                elif side_tec_trough:
                    pwall_diff = (tec_ewall - tec_pwall) / 2
                    ewall_diff = (tec_pwall - tec_ewall) / 2
                elif side.trough:
                    pwall_diff = (side.pwall_lat - side.ewall_lat) / 2
                    ewall_diff = (side.ewall_lat - side.pwall_lat) / 2

                res = {
                    'pwall_diff': pwall_diff,
                    'ewall_diff': ewall_diff,
                    'tec_trough': side_tec_trough,
                    'swarm_trough': side.trough,
                    'sat': sat,
                    'side': side_name,
                    'year': year,
                    'month': month,
                    'index': index,
                }
                results.append(res)

        if plot:
            plot_dir = os.path.join(self.save_dir, f"{year}_{month}_{index}")
            os.makedirs(plot_dir, exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True, subplot_kw=dict(projection='polar'))
            plotting.polar_pcolormesh(ax[0], config.mlat_grid, config.mlt_grid, tec, vmin=0, vmax=20)
            plotting.polar_pcolormesh(ax[1], config.mlat_grid, config.mlt_grid, tec_trough_model, vmin=0, vmax=1, cmap='Blues')
            plotting.polar_pcolormesh(ax[2], config.mlat_grid, config.mlt_grid, swarm_paths + 4 * tec_trough.astype(int), vmin=0, vmax=7, cmap='Blues')
            plotting.plot_swarm_trough_detections_polar(ax[2], swarm_segments)
            plotting.format_polar_mag_ax(ax)
            fig.savefig(os.path.join(plot_dir, "polar.png"))
            plt.close(fig)

            fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True, figsize=(16, 8))
            for s in range(6):
                plotting.plot_swarm_trough_detection(ax.flatten()[s], swarm_segments[s])
                ax.flatten()[s].grid()
            fig.savefig(os.path.join(plot_dir, "swarm.png"))
            plt.close(fig)

        self.tec_troughs.append(tec_trough)
        self.years.append(year)
        self.months.append(month)
        self.indexes.append(index)

        return results

    def run(self, iterations):
        t0 = time.time()
        results = []
        done = set()
        plot = True
        for i in range(iterations):
            if i == 100:
                plot = False
            while True:
                tid = utils.get_random_map_id()
                # tid = (2018, 10, 619)
                if tid in done:
                    continue
                try:
                    results += self.compare_single(*tid, plot)
                    done.add(tid)
                    print(i, tid)
                    break
                except Exception as e:
                    print(e.__class__)
                    print(e)
                    print(i, tid)
        print(f"{iterations} trials took {(time.time() - t0)/60} minutes")

        with h5py.File(os.path.join(self.save_dir, 'labels.h5'), 'w') as f:
            f.create_dataset('labels', data=np.array(self.tec_troughs))
            f.create_dataset('year', data=np.array(self.years))
            f.create_dataset('month', data=np.array(self.months))
            f.create_dataset('index', data=np.array(self.indexes))

        return pandas.DataFrame(results)


if __name__ == "__main__":
    import seaborn as sns
    RECALCULATE = False
    save_dir = "C:\\Users\\Greg\\Documents\\trough meetings\\comparison_select_best"
    results_file = os.path.join(save_dir, "results.csv")
    if RECALCULATE:
        comparer = TroughComparer(save_dir)
        df = comparer.run(round(5 * 60 * 12))
        df.to_csv(results_file)
    else:
        df = pandas.read_csv(results_file)

    neither, no_tec_yes_s, yes_tec_no_sw, both = confusion_matrix(df['tec_trough'], df['swarm_trough']).ravel()
    print('neither', neither)
    print('no tec yes swarm', no_tec_yes_s)
    print('yes tec no swarm', yes_tec_no_sw)
    print('both', both)
    print('total', df.shape[0])
    both_mask = np.all(df[['tec_trough', 'swarm_trough']], axis=1)
    print('pwall diff both present', df['pwall_diff'][both_mask].mean(), df['pwall_diff'][both_mask].var())
    print('ewall diff both present', df['ewall_diff'][both_mask].mean(), df['ewall_diff'][both_mask].var())

    ac_trough = df.groupby(['year', 'month', 'index', 'side'])['swarm_trough'].apply(lambda x: x.iloc[0] and x.iloc[2])
    cols = ['pwall_diff', 'ewall_diff']
    grid = sns.pairplot(df[both_mask][cols], diag_kind='kde')
    for a in grid.axes.flatten():
        a.grid(True)

    with h5py.File(os.path.join(save_dir, 'labels.h5'), 'r') as f:
        labels = f['labels'][()]
        year = f['year'][()]
        month = f['month'][()]
        index = f['index'][()]
    idx = np.column_stack((year, month, index))

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, labels.mean(axis=0))
    plotting.format_polar_mag_ax(ax)
    plt.colorbar(pcm)
    plt.show()