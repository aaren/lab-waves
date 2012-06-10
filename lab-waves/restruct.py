# This aims to do what waves does but in a less confused way and
# such that the resulting data structure contains data that is
# indexed according to feature.

# 1) Plot data

# 2) Select coherent structures by eye

# 3) Save data in new format

# First we presume that the parallax has been properly corrected and
# the values coming from the camera data streams are correct. This
# is what waves is for anyway - plotting up the data so that it can
# be checked in this regard.

# The idea is to store things in the format rdata[cs] = [(x,z,t)...]
# where cs means something like 'front' or 'wave1' and contains a
# list of points, each of which is specified in x, z and t
from __future__ import division
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from aolcore import read_data
from config import data_storage

def points(arg, rdata):
    arg_list = []
    point = namedtuple(arg, 'x, z, t')
    for cam in ['cam1', 'cam2']:
        cdata = rdata[cam]
        for f in cdata.keys():
            arg_list += [point(e[0], e[1], int(f)) for e in cdata[f][arg]]
    return arg_list

def bounding_lines(x_maxes, m_err, c_err):
    """Given a list of (x,t), fit a straight line and return two
    lines [m, c] displaced by error in gradient and intercept.
    Arguments: x_maxes is a list of (x,t) points to put a line
            through.
            m_err, c_err are the error on the gradient and y (time)
            intercept to be applied to the line.
    Return:    line_up, line_down, two lines that are displaced from
            the fitted line by given error.
    """
    # fit a straight line, return [m,c] for line y=mx+c
    x, T = zip(*x_maxes)
    m, c = np.polyfit(x, T, 1)
    # calculate the bounding lines
    line = namedtuple('line', 'm, c')
    upper = line(m + m_err, c + c_err)
    lower = line(m - m_err, c - c_err)
    return upper, lower

def get_line(arg):
    """ Given a list of points [(x, z, t), ....], where each point is
    represented by a named tuple, plot all of the points and prompt
    for user selection of a line section by defining two points.

    Returns list of points that fall within an error of the defined
    line.
    """
    # plot the argument
    X, Z, T = zip(*arg)
    plt.plot(X, T, 'bo')
    # get user input points
    pts = plt.ginput(2,0)
    # calculate the limits
    hi, lo = bounding_lines(pts, 0, 0.5)
    # select points within limits
    def in_bounds(point, upper, lower):
        x_lo = (point.t - upper.c) / upper.m
        x = point.x
        x_hi = (point.t - lower.c) / lower.m
        return (x_lo < x < x_hi)

    line = [p for p in arg if in_bounds(p, hi, lo)]
    # plot again
    Xs, Zs, Ts = zip(*line)
    # xt = [(p.x, p.t) for p in line]
    plt.plot(Xs, Ts, 'ro')
    plt.show()
    raw_input('')
    plt.close()
    return line

def main(run):
    data_file = data_storage + run
    indata = read_data(data_file)
    rdata = indata[run]
    maxima = points('max', rdata)

    line = get_line(maxima)
    X = [p.x for p in line]
    T = [p.t for p in line]

    m, c = np.polyfit(X, T, 1)
    speed = 1 / m

    print "Speed is %s" % speed

# TODO: Decide whether to use a tuple (x,z,t) or a dict {'x':,
# 'z':,'t':} for each of the points.

# ANSWER: Use a named tuple. We then know which arg each point came
# from in the first place (can we actually access this though??) and
# the point can be accessed as an object.

# and then select structures from this. This is so much shorter and
# clearer than the other way!

# Surely this was easier than the far longer method in waves??

# For the human selection of points, we will use ginput. So we plot
# up a particular type of point, then have the user select points
# that define the locations of those points. The manner in which
# this is done varies a bit with the type of feature.

# FRONT: The front is typically a single line of points that might
# meander a bit. Thus, select a start and end point for each roughly
# straight line section of it.

# WAVES: These are a number of distinct straight lines with no
# overlap.

# The common feature is straight lines. If we can detect a straight
# line then the rest will follow.


# Getting subsets of all of the runs is important for dealing with
# results. How to?
