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
    parser.add_argument('--hovmoller',
                        help='create hovmoller',
                        action='append_const',
                        const='hovmoller',
                        dest='actions')
    parser.add_argument('--wave_hovmoller',
                        help='create wave hovmoller',
                        action='append_const',
                        const='wave_hovmoller',
                        dest='actions')
    parser.add_argument('--list',
                        help='list the runs that match run_pattern',
                        action='store_true')

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(filename='logfile', level=logging.DEBUG)
    action_table = {'raw_process': raw_process,
                    'stitch':      stitch,
                    'hovmoller':   hovmoller,
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


def get_runs(regex="r1[123]*"):
    runpaths = glob.glob(os.path.join(config.path, 'synced', regex))
    runs = [os.path.basename(p) for p in runpaths]
    return runs


def stitch(run):
    r = ProcessedRun(run=run)
    r.write_out(si.with_visible_regions for si in r.stitched_images)


def raw_process(run):
    r = RawRun(run=run)
    r.process()


def hovmoller_current(pr, ax):
    Xg, Tg, Yg = pr.combine_current
    D = pr.parameters['D']
    contour_current = dict(cmap=plt.cm.bone_r,
                            levels=np.linspace(0, D / 2, 100))

    contourf = ax.contourf(Xg, Tg, Yg, **contour_current)
    return contourf


def hovmoller_wave(pr, ax):
    Xw, Tw, Yw = pr.combine_wave
    contour_wave = dict(cmap=plt.cm.bone_r,
                        levels=np.linspace(-1, 1, 100))

    Yw_ = Yw / Yw.mean() - 1
    contourf = ax.contourf(Xw, Tw, Yw_, **contour_wave)
    return contourf


def hovmoller(run, visible_regions=False):
    pr = ProcessedRun(run)
    # FIXME: catch single layer runs and do something else.

    fig, axes = plt.subplots(nrows=2, figsize=(8, 12), dpi=100)

    current_ax = axes[0]
    wave_ax = axes[1]

    c_wave = hovmoller_wave(pr, wave_ax)
    c_current = hovmoller_current(pr, current_ax)
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

    # FIXME: main figure title
    title = ("Hovmoller, {run_index} \n"
             r"$D={D}$, $H={H}$, $h_1={h_1}$, "
             r"$\rho_0={rho_0}$, $\rho_1={rho_1}$, $\alpha={alpha}$")

    axes[0].set_title(title.format(**pr.parameters))

    fig.tight_layout()

    if visible_regions:
        fig.savefig('plots/hovmoller_visible_simple_' + run + '.png')
    else:
        fig.savefig('plots/hovmoller_combine_' + run + '.png')

    plt.close(fig)


def composite(run, visible_regions=False, h=0.13, tag='50mm'):
    pr = ProcessedRun(run)
    X, T, Y = pr.combine_current

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

    if visible_regions:
        fig.savefig('plots/composite_visible_' + tag + '_' + run + '.png')
    else:
        fig.savefig('plots/composite_' + tag + '_' + run + '.png')


if __name__ == '__main__':
    args = command_line_parser()
    main(args)