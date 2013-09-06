from nose.tools import *

import matplotlib.pyplot as plt

from labwaves.runbase import ProcessedRun, InterfaceImage


def test_interface():
    pr = ProcessedRun('r13_01_13i')
    iima = InterfaceImage(list(pr.images)[30])
    pr = ProcessedRun('r11_07_06c')
    iimb = InterfaceImage(list(pr.images)[6])

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

    test_plot(iima)
    test_plot(iimb)

    plt.show()


if __name__ == '__main__':
    test_interface()
