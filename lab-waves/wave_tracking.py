# detection of constant speed coherent structures (i.e. waves)
from get_data import read_data
import numpy as np

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
    off['cam1'] = 0
    off['cam2'] = 0
    # this takes the data container and produces a single plot
    # with the wave and current trajectories plotted.
    # do this in a separate module at some point.
    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = 'data/data_store_' + run

    data = read_data(data_storage_file)

    xtm = {}
    for cam in ['cam1', 'cam2']:
        cam_data = data[run][cam]
        frames = sorted(cam_data.keys())
        Time = range(1, len(frames) - 1)
        xtm[cam] = [[round(p[0], 4)\
                for p in cam_data['img_%04d' % (T + off[cam])]['max']] \
                           for T in Time]

    Xtm = zip(xtm['cam1'], xtm['cam2'])
    Xtm = [Xtm[i][0] + Xtm[i][1] for i in range(len(Xtm))]

    return Xtm

    # then
    # Xtm[0] # means list of x for t=0
def track(t_start, Xtm):
    # the maximum x for some t WHICH NEEDS TO BE DEFINED is
    x_maxes = []
    for t in range(t_start, t_start+3):
        while True:
            x_max = max(Xtm[t]) 
            # don't accept outlying x_max (speed > 1 lock length/s)
            print 'loop'
            if x_max > t:
                try:
                    i = Xtm[t].index(x_max)
                    Xtm[t].pop(i)
                except ValueError:
                    pass
            else:
                break 
        print 'loop break'
        xt = (x_max, t)
        x_maxes.append(xt)

    # 2) repeat three times (at least)
    # get three of them
    # x_maxes = [(max(Xtm[t]), t) for t in range(t, t + 3)]

    x = [x_max[0] for x_max in x_maxes]
    T = [x_max[1] for x_max in x_maxes]

    # 3) fit a straight line
    # return [m,c] for line y=mx+c
    line = list(np.polyfit(x,T,1))

    # what is the allowed deviation in gradient and intercept?
    m_err = 0.5
    c_err = 1

    # calculate the bounding lines
    line_up = [line[0] + m_err, line[1] + c_err]
    line_down = [line[0] - m_err, line[1] - c_err]

    # 4) for all time, look for (x,t) that fit in the bounding lines
    # and add them to a list

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

# 5) repeat, ignoring points in L from prior waves
# test the above first.
# for given input data, return a list of lists. each list corresponds
# to a wave and its elements are coordinates (x,t).
