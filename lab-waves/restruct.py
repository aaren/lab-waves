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
    """From a given set of run data, i.e. data[run] if extracting
    from the data storage location, using arg as the key, pull out
    all of the data from both cameras and put it into a list.

    Each element of the list is a named tuple, with the x, z and t
    coords of the points found for arg.

    Returns the list of points.
    """
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

def in_bounds(point, upper, lower):
    x_lo = (point.t - upper.c) / upper.m
    x = point.x
    x_hi = (point.t - lower.c) / lower.m
    return (x_lo < x < x_hi)

def get_line(data):
    """ Given a list of points [(x, z, t), ....], where each point is
    represented by a named tuple, plot all of the points and prompt
    for user selection of a line section by defining two points.

    Returns list of points that fall within an error of the defined
    line.
    """
    # plot the points
    X, T = zip(*[(p.x, p.t) for p in data])
    plt.plot(X, T, 'bo')
    # get user input points
    pts = plt.ginput(2,0)
    # calculate the limits
    hi, lo = bounding_lines(pts, 0, 0.5)
    # select points within limits
    line = [p for p in data if in_bounds(p, hi, lo)]
    # plot again in red
    Xs, Ts = zip(*[(p.x, p.t) for p in line])
    plt.plot(Xs, Ts, 'ro')
    plt.draw()
    while True:
        print "Select bad points, middle click if none"
        bad = plt.ginput(0,0)
        if bad:
            for b in bad:
            # remove bad points from line
                line = [p for p in line if (b[0] - 0.1 < p[0] < b[0] + 0.1) \
                                       and (b[1] - 0.5 < p[1] < b[1] + 0.5)]
        elif not bad:
            break
        else:
            print "Indeterminate badness!"
        Xs, Ts = zip(*[(p.x, p.t) for p in line])
        plt.plot(Xs, Ts, 'go')
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

def get_front(run):
    data_file = data_storage + run
    indata = read_data(data_file)
    rdata = indata[run]
    front = points('front', rdata)

    # filter out anomalous points.
    filter_front = [f for f in front if f.x < 20]
    # sort by time
    time_front = sorted(filter_front, key=lambda p: p.t)

    # now get the points
    filtered = []
    init = time_front[:3]
    # really we want to iterate through time.
    t_max = time_front[-1].t
    for t in range(t_max):
        # find all points in this time slice
        pts = [p for p in time_front if p.t == t]
        # and the last three points in filtered
        # prepts = [p for p in filtered if (t - 3 < p.t < t)]
        prepts = filtered[-3:]
        # make the bounding lines from the previous points
        hi, lo = bounding_lines(prepts, 0, 0.5)
        for pt in pts:
            if in_bounds:
                filtered.append(pt)
            else:
                # append the nearest point in x to the previous
                closest = min(pts, key=lambda p: abs(prepts[-1].t - p.x))
                filtered.append(closest)
    return filtered

# Getting subsets of all of the runs is important for dealing with
# results. How to?
class Run(object):
    def __init__(self, run):
        self.index = run
        self.params = get_parameters(run, paramf)
        self.h1 = float(self.params['h_1/H'])
        self.h2 = 1 - self.h1
        self.D = float(self.params['D/H'])
        self.r0 = float(self.params['rho_0'])
        self.r1 = float(self.params['rho_1'])
        self.r2 = float(self.params['rho_2'])
        self.a = float(self.params['alpha'])

        self.c2l = f.two_layer_linear_longwave(self.h1, self.h2, \
                                                self.r1, self.r2)
        self.gce = f.gc_empirical(self.D / 2, self.r0, self.r1)
        self.gct = f.gc_theoretical(self.D / 2, self.r0, self.r1)

        self.data_file = data_storage + self.index

    def load(self):
        # Explicitly load the run data
        self.data = read_data(self.data_file)[run]

# make a list of the run indices
# indices = pull_col(0, paramf)
# make a list of run objects
# runs = [Run(index) for index in indices]
# select sub groupings, e.g.
# partial_runs = [r for r in runs if r.D == 0.4]
