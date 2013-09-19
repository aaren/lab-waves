import glob
import os
import argparse
import logging

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from labwaves import config
from labwaves.runbase import RawRun
from labwaves.runbase import ProcessedRun


# TODO: select from all runs on basis of something
# - hack read_parameters
single_layer_runs = {'r13_01_13g': 0.25,
                     'r13_01_13h': 0.20,
                     'r13_01_13i': 0.25,
                     'r13_01_13j': 0.25,
                     'r13_01_13k': 0.20,
                     'r13_01_13l': 0.25}

# visible regions
vis = ((0.71, 1.18),
       (1.70, 2.11),
       (2.70, 3.08))

# shifted visible regions
vis_ = ((-0.04, 0.43),
        (0.95, 1.36),
        (1.95, 2.33))


## command line interface
def command_line_parser():
    parser = argparse.ArgumentParser()
    # todo cli select runs on basis of anything in parameters
    parser.add_argument('run_pattern',
                        help="regex to select runs with",
                        nargs='?',
                        default='*')
    parser.add_argument('--process',
                        help='process run raw',
                        action='append_const',
                        const='raw_process',
                        dest='actions')
    parser.add_argument('--stitch',
                        help='create stitched images',
                        action='append_const',
                        const='stitch',
                        dest='actions')
    parser.add_argument('--cache',
                        help='extract interfaces and save to cache',
                        action='append_const',
                        const='cache',
                        dest='actions')
    parser.add_argument('--plots',
                        help='plots to create',
                        nargs='*',
                        default=[])
    parser.add_argument('--list',
                        help='list the runs that match run_pattern',
                        action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(filename='logfile', level=logging.DEBUG)
    action_table = {'raw_process': raw_process,
                    'stitch':      stitch,
                    'cache':       cache,
                    }

    runs = get_runs(args.run_pattern)
    actions = args.actions

    if args.list:
        print "matching pattern {}".format(args.run_pattern)
        print runs

    i = 1
    N = len(runs)
    for run in runs:
        print "\nthis is {run} {i}/{N}\n".format(run=run, i=i, N=N)
        print "performing these actions: {}".format(actions)
        for action in args.actions:
            try:
                action_table[action](run)
            except Exception as e:
                logmsg = "{run} {action} {error}".format
                logging.warning(logmsg(run=run, action=action, error=e))
        i += 1

        ra = RunPlotter(run)
        for plot in args.plots:
            print "plotting {}".format(plot)
            getattr(ra, plot)()


def get_runs(regex="r1[123]*"):
    runpaths = glob.glob(os.path.join(config.path, 'synced', regex))
    runs = [os.path.basename(p) for p in runpaths]
    return runs


def cache(run):
    ra = RunAnalysis(run, load_from_cache=False)
    ra.save_to_cache()


def stitch(run):
    r = ProcessedRun(run=run)
    r.write_out(si.with_visible_regions for si in r.stitched_images)


def raw_process(run):
    r = RawRun(run=run)
    r.process()


class RunAnalysis(object):
    """Container for basic analysis methods. Also implements caching
    of interface data."""
    def __init__(self, index, load_from_cache=True):
        self.pr = ProcessedRun(index)
        self.parameters = type('parameters', (object,), self.pr.parameters)

        self.cache_path = os.path.join(self.pr.output_dir, 'cache.npz')

        if load_from_cache:
            self.load_from_cache()

    @property
    def has_cache(self):
        return os.path.exists(self.cache_path)

    def load_from_cache(self):
        """Load interface data from cache file."""
        try:
            cache = np.load(self.cache_path)
        except IOError:
            print "nothing at {}".format(self.cache_path)
            return

        for k in cache.keys():
            setattr(self, k, cache[k])

    def save_to_cache(self):
        """Save interface data to cache."""
        arrays = {'combine_wave': self.combine_wave,
                  'combine_current': self.combine_current}
        np.savez(self.cache_path, **arrays)

    @property
    def combine_wave(self):
        """The wave interface."""
        if hasattr(self, '_combine_wave'):
            return self._combine_wave
        else:
            self.combine_wave = self.pr.combine_wave
            return self._combine_wave

    @combine_wave.setter
    def combine_wave(self, value):
        self._combine_wave = value

    @property
    def combine_current(self):
        """The current interface."""
        if hasattr(self, '_combine_current'):
            return self._combine_current
        else:
            self.combine_current = self.pr.combine_current
            return self._combine_current

    @combine_current.setter
    def combine_current(self, value):
        self._combine_current = value


class RunPlotter(RunAnalysis):
    """Container for plots."""
    def hovmoller_current(self, ax):
        Xg, Tg, Yg = self.combine_current
        D = self.parameters.D
        contour_current = dict(cmap=plt.cm.bone_r,
                                levels=np.linspace(0, D / 2, 100))

        contourf = ax.contourf(Xg, Tg, Yg, **contour_current)
        return contourf

    def hovmoller_wave(self, ax):
        Xw, Tw, Yw = self.combine_wave
        contour_wave = dict(cmap=plt.cm.bone_r,
                            levels=np.linspace(-1, 1, 100))

        Yw_ = Yw / Yw.mean() - 1
        contourf = ax.contourf(Xw, Tw, Yw_, **contour_wave)
        return contourf

    def hovmoller(self, savefig=False, visible_regions=False):
        # FIXME: catch single layer runs and do something else.

        fig, axes = plt.subplots(nrows=2, figsize=(8, 12), dpi=100)

        current_ax = axes[0]
        wave_ax = axes[1]

        c_wave = self.hovmoller_wave(wave_ax)
        c_current = self.hovmoller_current(current_ax)
        fig.colorbar(c_wave, ax=wave_ax)
        fig.colorbar(c_current, ax=current_ax)

        for ax in axes:
            if visible_regions:
            # vertical lines where the measurement regions are
                for i, v in enumerate(vis):
                    ax.axvspan(*v, color='r', alpha=0.2)

                # for i, v in enumerate(vis_):
                    # ax.axvspan(*v, color='y', alpha=0.2)

        # line at the transition
            ax.axvline(2.5, color='y', linewidth=2)
            ax.text(2.505, 6, 'phase transition', color='y', rotation=90)

        # label the current viewing region
            ax.text(2.8, 7, 'test run viewing window', color='w', rotation=90)

            ax.set_xlim(0.0, 3.5)
            ax.set_xlabel('distance from lock (m)')
            ax.set_ylabel('time from release (s)')

        title = ("Hovmoller, {run_index} \n"
                 r"$D={D}$, $H={H}$, $h_1={h_1}$, "
                 r"$\rho_0={rho_0}$, $\rho_1={rho_1}$, $\alpha={alpha}$")

        axes[0].set_title(title.format(**self.pr.parameters))

        fig.tight_layout()

        run = self.parameters.run_index

        if savefig:
            if visible_regions:
                fig.savefig('plots/hovmoller_visible_simple_' + run + '.png')
            else:
                fig.savefig('plots/hovmoller_combine_' + run + '.png')

            plt.close(fig)
        else:
            return fig

    def composite(self, visible_regions=False, h=0.13, tag='50mm'):
        X, T, Y = self.combine_current

        fig, ax = plt.subplots()

        for i in range(Y.shape[0]):
            ax.plot(X[i], Y[i], 'k', alpha=0.7)

        if visible_regions:
            for v in vis:
                w = v[1] - v[0]
                ax.add_patch(mpl.patches.Rectangle((v[0], 0),
                                                   width=w,
                                                   height=h,
                                                   color='r',
                                                   alpha=0.5))
            for v in vis_:
                w = v[1] - v[0]
                ax.add_patch(mpl.patches.Rectangle((v[0], 0),
                                                   width=w,
                                                   height=h,
                                                   color='y',
                                                   alpha=0.5))

        ax.set_xlim(0.0, 3.5)
        ax.set_ylim(0.0, 0.25)

        ax.set_xlabel('distance (m)')
        ax.set_ylabel('height (m)')

        run = self.parameters.run_index

        if visible_regions:
            fig.savefig('plots/composite_visible_' + tag + '_' + run + '.png')
        else:
            fig.savefig('plots/composite_' + tag + '_' + run + '.png')


if __name__ == '__main__':
    args = command_line_parser()
    main(args)
