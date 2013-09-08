# Aim: create a hovmoller for each of the single layer runs
import numpy as np
import matplotlib.pyplot as plt

from labwaves.runbase import ProcessedRun


single_layer_runs = ['r13_01_13g',
                     'r13_01_13h',
                     'r13_01_13i',
                     'r13_01_13j',
                     'r13_01_13k',
                     'r13_01_13l']


def create_hovmoller(run, visible_regions=False):
    pr = ProcessedRun(run)
    X, T, Y = pr.combine_current()

    fig, ax = plt.subplots()

    contour_args = dict(cmap=plt.cm.bone,
                        levels=np.linspace(0, 0.25, 100))

    contourf = ax.contourf(X, T, Y, **contour_args)

    # visible regions
    vis = ((0.71, 1.18),
           (1.70, 2.11),
           (2.70, 3.08))

    # shifted visible regions
    vis_ = ((-0.04, 0.43),
            (0.95, 1.36),
            (1.95, 2.33))

    if visible_regions:
    # vertical lines where the measurement regions are
        for i, v in enumerate(vis):
            ax.axvspan(*v, color='r', alpha=0.2)

        for i, v in enumerate(vis_):
            ax.axvspan(*v, color='y', alpha=0.2)

    ax.set_xlim(0, 3.5)
    ax.set_xlabel('distance from lock (m)')

    ax.set_ylabel('time from release (s)')

    ax.set_title("hovmoller {}".format(run))

    fig.colorbar(contourf)

    if visible_regions:
        fig.savefig('plots/hovmoller_visible_' + run + '.png')
    else:
        fig.savefig('plots/hovmoller_' + run + '.png')


for run in single_layer_runs:
    print run
    print 'hovmoller...'
    create_hovmoller(run)
    print 'hovmoller visible regions...'
    create_hovmoller(run, visible_regions=True)
    print 'done'
