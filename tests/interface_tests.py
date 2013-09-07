import sys

import matplotlib.pyplot as plt

from labwaves.runbase import ProcessedRun, InterfaceImage


def test_interface(run, n):
    pr = ProcessedRun(run)
    iim = InterfaceImage(list(pr.images)[n])

    def test_plot(iim):
        fig = iim.plot_channels()

        lx, ly = iim.current_interface

        for ax in fig.axes[0: -1]:
            ax.plot(lx, ly, 'k.')

        if iim.has_waves:
            wx, wy = iim.wave_interface
            for ax in fig.axes[0:3]:
                ax.plot(wx, wy, 'r.')

            fig.axes[5].plot(wx, wy, 'r.')

        w, h = iim.im.size
        for ax in fig.axes:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

    test_plot(iim)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run = 'r13_01_13g'
        n = 45
    elif len(sys.argv) > 2:
        run = sys.argv[1]
        n = int(sys.argv[2])

    test_interface(run, n)
