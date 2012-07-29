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

from aolcore import read_data, get_parameters
from config import data_storage, paramf
import functions as f

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
        self.data = read_data(self.data_file)[self.index]

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
            arg_list += [point(e[0], e[1], int(f) - 1) for e in cdata[f][arg]]
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
        print "Select bad points, middle click if finished"
        bad = plt.ginput(0,0)
        if bad:
            # identify the bad points
            bad_points = []
            for b in bad:
                bad_points += [p for p in data if \
                                        (b[0] - 0.1 < p.x < b[0] + 0.1) \
                                    and (b[1] - 0.5 < p.t < b[1] + 0.5)]
            # overplot
            Xb, Tb = zip(*[(p.x, p.t) for p in bad_points])
            plt.plot(Xb, Tb, 'go')
            plt.draw()
            # remove bad points from line
            for b in bad_points:
                line.remove(b)
        elif not bad:
            break
        else:
            print "Indeterminate badness!"
    while True:
        print "Select missed points, middle click if finished"
        missed = plt.ginput(0,0)
        if missed:
            missed_points = []
            print missed
            print ""
            print line
            for m in missed:
                missed_points += [p for p in data if \
                                        (m[0] - 0.2 < p.x < m[0] + 0.2) \
                                    and (m[1] - 0.5 < p.t < m[1] + 0.5)]
            Xm = [p.x for p in missed_points]
            Tm = [p.t for p in missed_points]
            print Xm
            print Tm
            plt.plot(Xm, Tm, 'co')
            plt.draw()
            # add missed points to line
            for m in missed_points:
                line.append(m)
        elif not missed:
            break
        else:
            print "I don't know who you are or where you came from."
    Xs, Ts = zip(*[(p.x, p.t) for p in line])
    plt.plot(Xs, Ts, 'yo')
    plt.draw()
    return line

def get_front(run):
    """ From a particular run, take the front data and extract a
    single string of data. Needs to be monotonic in t, but allow
    for multivalued in x.
    """
    data_file = data_storage + run
    indata = read_data(data_file)
    print "read data"
    rdata = indata[run]
    front = points('front', rdata)

    # filter out anomalous points (produced from earlier
    # processing)
    filter_front = [f for f in front if f.x < 20]
    # sort by time
    time_front = sorted(filter_front, key=lambda p: p.t)
    # now get the points. set up a container with some initial
    # points.
    proc_front = time_front[:3]
    t_max = time_front[-1].t
    for t in range(t_max):
        pts = [p for p in time_front if p.t == t]
        prepts = [(p.x, p.t) for p in proc_front[-3:]]
        for pt in pts:
            # append the nearest point in x to the previous
            closest = min(pts, key=lambda p: abs(prepts[-1][0] - p.x))
            proc_front.append(closest)
    return proc_front

def get_lines(data, arg):
    """ From given data and argument, e.g. maxima, get a list
    of lines selected by the user.
    """
    lines = []
    y_x, y_y = [0,1,1,0],[35,35,40,40]
    n_x, n_y = [1,2,2,1],[35,35,40,40]
    plt.fill(y_x, y_y, 'g')
    plt.fill(n_x, n_y, 'r')
    while True:
        pts = points(arg, data)
        print "Choose a line by selecting a couple of points in it."
        line = get_line(pts)
        X = [p.x for p in line]
        T = [p.t for p in line]
        m, c = np.polyfit(X, T, 1)
        Xf = np.linspace(0,12)
        Tf = m * Xf + c
        plt.plot(Xf, Tf)
        lines.append(line)
        plt.draw()
        print "Finished?? green; yes; red/anywhere; no, need more waves"
        a = plt.ginput(1,0)
        if (y_x[0] < a[0][0] < y_x[1]) and (y_y[0] < a[0][1] < y_y[2]):
            break
        elif (n_x[0] < a[0][0] < n_x[1]) and (n_y[0] < a[0][1] < n_y[2]):
            pass
        else:
            # print "click in the red or green!"
            pass
    plt.close()
    return lines

def main(run):
    # intialise run as instance of Run class
    r = Run(run)
    # explicitly load the data
    r.load()
    # get the waves
    # lines = get_lines(r.data, 'max')
    # waves = {'w%s' % i: l for i,l in enumerate(lines)}
    # get the front
    front = get_front(run)
    # test plot this to check how separate things are
    # plot the front
    Xf = [p.x for p in front]
    Tf = [p.t for p in front]
    plt.plot(Xf, Tf, 'k*')
    # plot the waves
    # for i,w in enumerate(sorted(waves.keys())):
        # Xw = [p.x for p in waves[w]]
        # Tw = [p.t for p in waves[w]]
        # plt.plot(Xw, Tw, '*', \
            # color=plt.get_cmap('hsv')((i+1)/30))
    print "how u lik this applz??11"
    plt.show()

if __name__ == '__main__':
    main('r11_7_06c')








# Getting subsets of all of the runs is important for dealing with
# results. How to?  make a list of the run indices indices =
# pull_col(0, paramf) make a list of run objects runs = [Run(index)
# for index in indices] select sub groupings, e.g.  partial_runs =
# [r for r in runs if r.D == 0.4]
