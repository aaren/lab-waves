# Aim: create a hovmoller for each of the single layer runs
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from labwaves.runbase import ProcessedRun


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


def create_hovmoller(pr, visible_regions=False):
    run = pr.index
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
                              H=single_layer_runs[run],
                              rho_0=pr.parameters['rho_0'],
                              rho_1=pr.parameters['rho_1']))

    fig.colorbar(contourf)
    fig.tight_layout()

    if visible_regions:
        fig.savefig('plots/hovmoller_visible_simple_' + run + '.png')
    else:
        fig.savefig('plots/hovmoller_' + run + '.png')


def create_composite(pr, visible_regions=False, h=0.13, tag='50mm'):
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


for run in ['r13_01_13j']:
# for run in single_layer_runs:
    pr = ProcessedRun(run)
    print run
    print 'hovmoller...'
    # create_hovmoller(pr)
    print 'hovmoller visible regions...'
    create_hovmoller(pr, visible_regions=True)
    print 'composite...'
    # create_composite(pr, visible_regions=True, h=0.13, tag='50mm')
    # create_composite(pr, visible_regions=True, h=0.185, tag='35mm')
    print 'done'
