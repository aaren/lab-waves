from nose.tools import *

import matplotlib.pyplot as plt

from labwaves.runbase import ProcessedRun, InterfaceImage


def test_interface():
    # pr = runbase.ProcessedRun('r13_01_13i')
    # iim = runbase.InterfaceImage(list(pr.images)[30])
    pr = ProcessedRun('r11_07_06c')
    iim = InterfaceImage(list(pr.images)[6])

    fig = iim.plot_channels()

    lx, ly = iim.lock_interface

    for ax in fig.axes:
        ax.plot(lx, ly, 'k.')

    plt.show()


if __name__ == '__main__':
    test_interface()
