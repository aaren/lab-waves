import glob
import os
import argparse

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
    action_table = {'raw_process': raw_process,
                    'stitch':      stitch,
                    'hovmoller':   hovmoller,
                    'wave_hovmoller': wave_hovmoller,
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
            action_table[action](run)
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


def hovmoller(run, visible_regions=False):
    pr = ProcessedRun(run)
    X, T, Y = pr.combine_current

    fig, ax = plt.subplots()

    contour_args = dict(cmap=plt.cm.bone,
                        levels=np.linspace(0, 0.25, 100))

    contourf = ax.contourf(X, T, Y, **contour_args)

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

    title = r"Hovmoller, {run}: $H={H}$, $\rho_0={rho_0}$, $\rho_1={rho_1}$"
    ax.set_title(title.format(run=run,
                              H=pr.parameters['H'],
                              rho_0=pr.parameters['rho_0'],
                              rho_1=pr.parameters['rho_1']))

    fig.colorbar(contourf)
    fig.tight_layout()

    if visible_regions:
        fig.savefig('plots/hovmoller_visible_simple_' + run + '.png')
    else:
        fig.savefig('plots/hovmoller_' + run + '.png')


def wave_hovmoller(run, visible_regions=False):
    pr = ProcessedRun(run)
    X, T, Y = pr.combine_wave

    fig, ax = plt.subplots()

    contour_args = dict(cmap=plt.cm.bone,
                        levels=np.linspace(-1, 1, 100))

    Y_ = Y / Y.mean() - 1
    contourf = ax.contourf(X, T, Y_, **contour_args)

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

    title = r"Wave Hovmoller, {run}: $H={H}$, $\rho_0={rho_0}$, $\rho_1={rho_1}$"
    ax.set_title(title.format(run=run,
                              H=pr.parameters['H'],
                              rho_0=pr.parameters['rho_0'],
                              rho_1=pr.parameters['rho_1']))

    cbar = fig.colorbar(contourf)
    cbar.set_label(r'$\eta(x, t) = \frac{z(x, t)}{\bar{z}} - 1$')
    fig.tight_layout()

    if visible_regions:
        fig.savefig('plots/hovmoller_visible_simple_' + run + '.png')
    else:
        fig.savefig('plots/hovmoller_wave' + run + '.png')


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
