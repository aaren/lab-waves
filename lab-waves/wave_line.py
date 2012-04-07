# detection of constant speed coherent structures (i.e. waves)

# the point of this is twofold:
# 1) sanitise the detected data by only allowing features that
# are actually supposed to be there (i.e. remove noise).
# 2) move from time indexed data to feature indexed data.
# this requires an a apriori assumption of the form of the
# features observed, but we have a pretty good idea of that.
# (in this case straight lines in spacetime).

# This is a good candidate for wrapping in a class when moved back
# into the main get_data program. Should it actually be moved back??
# This is just a utility feature - could have all of these utilities
# in the same file and then call them from a separate file.

from get_data import read_data
import numpy as np
import matplotlib.pyplot as plt

# # what is the furthest spatial coord?
# Xtm[0] # means list of (x,t) for frame 0
def wave(run='r11_7_06c', data_storage_file=None):
    """Joins together the cam1 and cam2 data to give
    a single data set in (x,t) for a given run.
    Inputs: run_data dictionary
    Returns: Xtm, which is a list of x for given t, i.e. Xtm[0]
    is a list of all the measured x for t=0.
    """
    data_dir = '/home/eeaol/code/lab-waves/data/'
    
    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = data_dir + 'data/data_store_' + run

    data = read_data(data_storage_file)

    xtm = {}
    for cam in ['cam1', 'cam2']:
        cam_data = data[run][cam]
        frames = sorted(cam_data.keys())
        xtm[cam] = [[p for p in cam_data[frame]['max']] for frame in frames]

    Xtm = zip(xtm['cam1'], xtm['cam2'])
    Xtm = [Xtm[i][0] + Xtm[i][1] for i in range(len(Xtm))]

    # strip out non physical points (faster then 1:1)
    Xtm = [[x for x in Xtm[t] if x < t] for t in range(len(Xtm))]

    return Xtm

def conjoin_data(run, data_storage_file=None):
    data_dir = '/home/eeaol/code/lab-waves/data/'
    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = data_dir + 'data/data_store_' + run
    data = read_data(data_storage_file)

    Xt = {}
    for arg in ['max', 'min', 'front']:
        xt = {}
        for cam in ['cam1', 'cam2']:
            cam_data = data[run][cam]
            frames = sorted(cam_data.keys())
            xt[cam] = [[p for p in cam_data[frame][arg]] for frame in frames]

        Xt[arg] = zip(xt['cam1'], xt['cam2'])
        Xt[arg] = [e[0] + e[1] for e in Xt[arg]]
    
    # we can keep Xtm if it is used as shorthand for Xt[arg]

    Xtm = Xt['max']
    Xtmin = Xt['min']
    Xtf = Xt['front']

    # strip out non physical points (faster then 1:1)
    #for arg in ['max', 'min']:
    #    Xt[arg] = [[x for x in Xt[arg] if x[0] < t] for t in range(len(Xtm))]

    return Xt

def bounding_init(t_start, Xtm):
    """Initialise the list of coordinates with the first 
    three that are furthest in x. This gives an initial 
    bounding region in which to search for more
    points. 
    """
    x_maxes = [] 
    for t in range(t_start, t_start+3): 
        try: x_max = max(Xtm[t])
        except ValueError: 
            break 
        xt = (x_max, t) 
        x_maxes.append(xt) 
    return x_maxes

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
    x,T = zip(*x_maxes)
    m,c = np.polyfit(x,T,1)
    # calculate the bounding lines
    line_up = [m + m_err, c + c_err]
    line_down = [m - m_err, c - c_err]
    return line_up, line_down

def find_points(t, line_up, line_down, Xtm):
    """For a given time slice, look for (x,t) that fit 
    in the given bounding lines. 

    Return: a list of tuples (x,t) of the points that fit.
    """
    x_down = (t - line_up[1]) / line_up[0]
    x_up = (t - line_down[1]) / line_down[0]

    return [(x,t) for x in Xtm[t] if (x_down < x < x_up)]

# METHOD SUMMARY
# bounding_init # returns first three coords, given a starting time
# bounding_line # returns the lines that bound given list of coords
# find_points # returns list of points that fit inside the bounds 
#               for a given t
def remove_from(Xtm, points):
    # Take Xtm and remove supplied points.
    # Points is a list of (x,t) coordinates
    for point in points:
        x, t = point
        try:
            i = Xtm[t].index(x)
            Xtm[t].pop(i)
        except ValueError:
            pass
    return Xtm

def track(t_start, t_end, Xtm, m_err=0, c_err=1):
    # For a given time range and data array, find the coordinates
    # of the points that fit in a wave-like structure matching the
    # fastest wave (i.e. furthest out in x for given t), given some
    # allowed error in the coordinates.
    init_x_maxes = bounding_init(t_start, Xtm)
    line_up, line_down = bounding_lines(init_x_maxes, m_err, c_err)
    x_maxes = []
    for t in xrange(t_start, t_end):
        x_max = find_points(t, line_up, line_down, Xtm)
        x_maxes = x_maxes + x_max
        line_up, line_down = bounding_lines(x_maxes, m_err, c_err)
    return x_maxes

def get_t_start(Xtm):
    """From a given map of points, work out what time the first
    wave appears at. This is just the first non empty list (with
    no noise...).
    """
    for t in range(len(Xtm)):
        if len(Xtm[t]) != 0:
            return t
        else: pass

def conv(Xt, arg):
    Xta = Xt[arg]
    return [[p[0] for p in Xta[i]] for i in range(len(Xta))]

def get_waves(t_start, t_end, Xtm, n=1):
    """Get the first n waves from Xtm. After a wave is detected
    its points are removed from the dataset so that the next wave
    can be detected.

    Xtm has the form of a list of lists, such that Xtm[1] gives a list
    of the positions of the detected thing* for t=1s. It is shorthand
    for [[p[0] for p in T] for T in Xt['max']], where Xt is the object
    returned by conjoin_data.

    *the 'thing' could be maxima, minima, front, provided it is only a
    list of x positions and not a list of tuples.

    Return: waves, a list of lists of the (x,t) of points in the waves
            such that waves[2] is a list of the points in the third wave.
    """
    waves = []
    for i in range(n):
        t_start=get_t_start(Xtm)
        x_maxes = track(t_start, t_end, Xtm, m_err=0, c_err=1)
        Xtm = remove_from(Xtm, x_maxes)
        waves.append(x_maxes)
        # put a line through the wave
        X,T = zip(*x_maxes)
        m,c = np.polyfit(X,T,1)
        me = m
        ce = c + 0.8
        # remove left over points
        Xtm = [[x for x in Xtm[t] if x < (t - ce) / me] \
                                    for t in range(len(Xtm))]
    return waves

# plotting for testing the wave detection
def plot_lines(line):
    x = np.linspace(0, 12, 30)
    t = line[0] * x + line[1]
    plt.plot(x,t)

def plot_xtm(Xtm):
    # make a list of (x,t) tuples over all Xtm
    xt = [(x,t) for t in range(len(Xtm)) for x in Xtm[t]]
    x,t = zip(*xt)
    plt.plot(x, t, 'bo')

def plot_waves(waves):
    """Take a given list of waves and plot them with different colours."""
    for wave in waves:
        x,t = zip(*wave)
        plt.plot(x, t, 'ro')

def test(n=1, run='r11_7_06c'):
    """Produce a plot for visual checking of wave detection.
    TODO: WHAT SHOULD THIS LOOK LIKE??
    """
    start = 2
    end = 25
    no_waves = n
    Xtm = wave(run)
    nXtm = [[p[0] for p in T] for T in Xtm]

    plot_xtm(nXtm)
    waves = get_waves(start, end, nXtm, no_waves)
    plot_waves(waves)

def set():
    plt.xlim(0, 12)
    plt.ylim(0, 50)
    plt.xlabel("Distance (lock lengths)")
    plt.ylabel("Time (s)")

def fit_waves(waves):
    for wave in waves:
        x,t = zip(*wave)
        m,c = np.polyfit(x,t,1)
        T = np.linspace(1, 25, 25)
        X = (T - c) / m
        c = 1 / m
        plt.plot(X, T, label="c=%.2f /s" %c)
    
def plot_front(run='r11_7_06c', data=None, fmt=None):
    if data is None:
        data_storage = '/home/eeaol/code/lab-waves/data/data/data_store_'
        data = read_data(data_storage + run)

    if fmt is None:
        fmt = 'ko'

    f={}
    t={}
    for cam in ['cam1', 'cam2']:
        cam_data = data[run][cam]
        frames = sorted(cam_data.keys())

        f[cam] = [[p[0] for p in cam_data[frame]['front']] for frame in frames]
        t[cam] = range(len(cam_data.keys()))

    # f['cam1'] = f['cam1'][0:25]
    # t['cam1'] = t['cam1'][0:25]
    # f['cam2'] = f['cam2'][20:45]
    # t['cam2'] = t['cam2'][20:45]

    plt.plot(f['cam1'], t['cam1'], fmt)
    plt.plot(f['cam2'], t['cam2'], fmt, label = 'g.c. front')

    plt.xlim(0,13)
