# Code to plot a Hovmoller for the lab data, using the interface
# height and the baseline to make a 2d colour contoured plot of
# interface deformation in (x,t).
# AOL Feb 2012

# The data is effectively continuous in x, but discrete in t (each
# image constitutes a slice in time). This differs from e.g. field
# data, where one might expect continuous time but at discrete
# spatial locations. In model data, one might expect a reasonable
# density in x but only output every timestep.

# Steps:
# 1) Get the data object (interface and baseline from file).
# 2) Plot interface for each timeslice.
# 3) Interpolate.
# 4) Display with colour_bar, which must use the same scale over
# different runs.

from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
import numpy as np

from get_datav3 import read_data

# data = read_data('data/data_store_r11_7_06c')

# first try and do it with a single camera

# data[run][cam][frame]['interface']

def interp1(interface):
    int_x = zip(*interface)[0]
    int_h = zip(*interface)[1]
    int_x = np.asarray(int_x)[::-1]
    int_h = np.asarray(int_h)[::-1]
    # do a 1d interpolation first to reduce the overhead in 2d
    f1 = interp1d(int_x, int_h)
    # new values
    xn = np.arange(0.1, 5, 0.01)
    hn = f1(xn)
    return xn, hn

# ValueError: A value in x_new is below the interpolation range.

def interp(data, T):
    h = [data['%04d' % t]['interface'] for t in range(1,T)]
    hn = [interp1(ho) for ho in h]
    return hn

def interp2(hn):
    h = [hi[1] for hi in hn] 
    x = hn[0][0]
    T = range(len(hn))
    f = interp2d(x, T, h, kind='linear')
    return f

# i think a problem is that the x data is irregularly spaced, following
# the normalisation. -- > not true. x has been recalculated over a
# regular spacing.


class hovmoller(object):

    # make a map of wave speed in (x,t) by looking
    # at the detected waves
    def gen_c_map(run):
        # some stuff
        return c_map
    
    def h(x,t):
        # lookup what the value of h is at some given
        # (x,t). Should be able to work with any input
        # numbers, i.e. interpolates to grid.
        # initially, this can only look along discrete
        # time intervals.

    def interpolate(region):
        for (x, t) in region:
            c = c_map(x,t)
            t0 = int(t)
            t1 = t0 + 1
            x0 = x - c*(t - t0)
            x1 = x + c*(t1 - t)
            h0 = h(x0,t0)
            h1 = h(x1,t1)
            hxt = h0 + (h1-h0) * (t - t0) / (t1 - t0)
            # stick this in an array.

