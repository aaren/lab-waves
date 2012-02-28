# detection of constant speed coherent structures (i.e. waves)

# the point of this is twofold:
# 1) sanitise the detected data by only allowing features that
# are actually supposed to be there (i.e. remove noise).
# 2) move from time indexed data to feature indexed data.
# this requires an a apriori assumption of the form of the
# features observed, but we have a pretty good idea of that.

from get_data import read_data
import numpy as np
import matplotlib.pyplot as plt

# 1) for a given early time, locate furthest spatial coord 

# for cam in ['cam1', 'cam2']:
#     xtm[cam] = [[(p[0], int(frame) + off[cam])\
#             for p in cam_data[frame]['max']] \
#                        for frame in frames]
# 
# Xtm = map(list, zip(a,b))
# 
# # alternate...
# for cam in ['cam1', 'cam2']:
#     for frame in frames:
#         xm[cam][frame] = [[p[0] for p in cam_data[frame]['max']]
# 
# # what is the furthest spatial coord?
# Xtm[0] # means list of (x,t) for frame 0

# change it to
def wave(run='r11_7_06c', data_storage_file=None):
    off = {}
    off['cam1'] = -1
    off['cam2'] = -2
    
    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = 'data/data_store_' + run

    data = read_data(data_storage_file)

    xtm = {}
    for cam in ['cam1', 'cam2']:
        cam_data = data[run][cam]
        frames = sorted(cam_data.keys())
        Time = range(3, len(frames) - 1)
        xtm[cam] = [[round(p[0], 4)\
                for p in cam_data['img_%04d' % (T + off[cam])]['max']] \
                           for T in Time]

    Xtm = zip(xtm['cam1'], xtm['cam2'])
    Xtm = [Xtm[i][0] + Xtm[i][1] for i in range(len(Xtm))]

    # strip out non physical points (faster then 1:1)
    Xtm = [[x for x in Xtm[t] if x < t] for t in range(len(Xtm))]

    return Xtm

# then
# Xtm[0] # means list of x for t=0
# the maximum x for some t WHICH NEEDS TO BE DEFINED is

# 2) repeat three times (at least)
# get three of them
# x_maxes = [(max(Xtm[t]), t) for t in range(t, t + 3)]

def bounding_init(t_start, Xtm):
# """Initialise the list of coordinates with the first three
# that are furthest in x, but not physically unrealistic.
# """
    x_maxes = []
    # this gives us an initial bounding line
    for t in range(t_start, t_start+3):
        try:
            x_max = max(Xtm[t]) 
        except ValueError:
            break
        xt = (x_max, t)
        x_maxes.append(xt)

    return x_maxes

def bounding_lines(x_maxes, m_err, c_err):
# """Given a list of (x,t), fit a straight line and return two
# lines displaced by error in gradient and intercept.
# """
    x = [x_max[0] for x_max in x_maxes]
    T = [x_max[1] for x_max in x_maxes]

    # 3) fit a straight line
    # return [m,c] for line y=mx+c
    line = list(np.polyfit(x,T,1))

    # what is the allowed deviation in gradient and intercept?
    # m_err = 0.5
    # c_err = 1

    # calculate the bounding lines
    line_up = [line[0] + m_err, line[1] + c_err]
    line_down = [line[0] - m_err, line[1] - c_err]
    return line_up, line_down

def find_points(t, line_up, line_down, Xtm):
# """for a given time slice, look for (x,t) that fit 
# in the bounding lines. Returns a list of tuples, (x,t),
# of the points that match.
# """
    L = []
    for i in range(len(Xtm[t])):
        x = Xtm[t][i]
        x_down = (t - line_up[1]) / line_up[0]
        x_up = (t - line_down[1]) / line_down[0]
        if (x_down < x < x_up):
            coord = (x,t)
            L.append(coord)
        else:
            pass
    return L

# bounding_init # returns first three coords, given a starting time
# bounding_line # returns the lines that bound given list of coords
# find_points # returns list of points that fit inside the bounds for a given t

# 5) repeat, ignoring points in L from prior waves
# test the above first.
# for given input data, return a list of lists. each list corresponds
# to a wave and its elements are coordinates (x,t).

def remove_from(Xtm, points):
    # points is a list of (x,t) coordinates
    for point in points:
        x, t = point
        try:
            i = Xtm[t].index(x)
            Xtm[t].pop(i)
        except ValueError:
            pass
    return Xtm

def track(t_start, t_end, Xtm, m_err=0, c_err=1):
    x_maxes = bounding_init(t_start, Xtm)
    for t in xrange(t_start + 4, t_end):
        line_up, line_down = bounding_lines(x_maxes, m_err, c_err)
        x_max = find_points(t, line_up, line_down, Xtm)
        x_maxes = x_maxes + x_max

    return x_maxes

def get_waves(t_start, t_end, Xtm, no_waves):
    waves = []
    for i in range(no_waves):
        x_maxes = track(t_start, t_end, Xtm, m_err=0, c_err=1)
        remove_from(Xtm, x_maxes)
        waves.append(x_maxes)
    return waves


def plot_lines(line):
    x = np.linspace(0, 12, 30)
    t = line[0] * x + line[1]
    plt.plot(x,t)

def plot_xtm(Xtm):
    t = range(len(Xtm))
    x = [p[0] for p in Xtm]
    plt.plot(x, t, 'bo')

    # 4) for all time, look for (x,t) that fit in the bounding lines
    # and add them to a list
def find_points_all_t(upper, lower, Xtm):
    L = []
    for t in range(len(Xtm)):
        for i in range(len(Xtm[t])):
            x = Xtm[t][i]
            x_down = (t - line_up[1]) / line_up[0]
            x_up = (t - line_down[1]) / line_down[0]
            t_up = line_up[0] * x + line_up[1]
            t_down = line_down[0] * x + line_down[1]
            if ((x_down < x < x_up) and (t_down < t < t_up)):
                coord = (x,t)
                L.append(coord)
    return L

# L is now full of the points that fit between the lines. and it
# is sorted w.r.t time
# therefore, L describes the first wave exactly.
