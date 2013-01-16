# detection of constant speed coherent structures (i.e. waves)
# some stuff

# Trying here to clean it all up. Specifically by using a class.
import sys

import numpy as np
# import matplotlib as mpl
# Ensure no errors when there is no display.
# Must be done before pyplot import.
# mpl.use('Agg')
import matplotlib.pyplot as plt

from aolcore import read_data, get_parameters
from config import data_storage, pdir, plots_dir
from config import paramf
import functions as f

class RunData(object):
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

class Conjoin(RunData):
    ###FIXME This is broken. It can't deal with non continuous time
    ### data. Need to put the time reference into the actual object
    ### rather than relying on the index of the list.
    def conjoin_data(self, data_file=None):
        run = self.index
        if run.split('r')[0] == run:
            run = 'r' + run
        if data_file is None:
            data_file = data_storage + run
        data = read_data(data_file)

        Xt = {}
        Xtn = {}
        for arg in ['max', 'min', 'front']:
            xt = {}
            for cam in ['cam1', 'cam2']:
                cam_data = data[run][cam]
                frames = sorted(cam_data.keys())
                # xt[cam] = [[p for p in cam_data[frame][arg]] for frame in frames]

                # If e.g. frames are 3-5 and 7-9: frames will
                # include [3,4,5,7,8,9]. Then for each of these,
                # xt[cam] will have a list of the frame data
                # appended. Because the frames are not consecutive,
                # this will lead to gaps in the conjoined data. The
                # solution is to construct xt as a dict instead,
                # with the keys as the frame number. This changes
                # the implementation of anything that uses xt, and I
                # suspect non trivially.
                #
                # The alternative is to make sure that all of the
                # frames are included, i.e. make frames from
                # something other than the key list, or put in
                # filler frames when the frame isn't in the key
                # list.

                xt[cam] = {str(frame): [p for p in cam_data[frame][arg]] \
                                                        for frame in frames}

                # then we access frames by e.g.

                # xt['cam1']['0003']

            # how to conjoin now?
            # Xt[arg]= {k: xt['cam1'][k] + xt['cam2'][k] for k in xt['cam2']}
            #
            # this won't work. only inserts keys that are present in
            # cam2. more explicitly:
            #
            # determine the maximum frame number.

            cam1f = [int(k) for k in xt['cam1'].keys()]
            try:
                cam1max = max(cam1f)
            except ValueError:
                cam1max = 0
            cam2f = [int(k) for k in xt['cam2'].keys()]
            try:
                cam2max = max(cam2f)
            except ValueError:
                cam2max = 0
            fmax = max(cam1max, cam2max)

            Xt[arg] = {}
            Xtn[arg] = []
            for f in range(fmax):
                F = '%04d' % f
                if F in xt['cam1'] and F in xt['cam2']:
                    Xt[arg][F] = xt['cam1'][F] + xt['cam2'][F]
                elif F in xt['cam1']:
                    Xt[arg][F] = xt['cam1'][F]
                elif F in xt['cam2']:
                    Xt[arg][F] = xt['cam2'][F]
                if F in Xt[arg]:
                    Xtn[arg].append(Xt[arg][F])
                else:
                    # stick a blank placeholder in
                    Xtn[arg].append([])

        # strip out non physical points (faster then 1:1)
        #for arg in ['max', 'min']:
        #    Xt[arg] = [[x for x in Xt[arg] if x[0] < t] for t in range(len(Xtm))]
        return Xtn

    def plot_xt(self, arg, fmt):
        Xt = self.conjoin_data()
        Xtm = conv(Xt, arg)
        xt = [(x, t) for t in range(len(Xtm)) for x in Xtm[t]]
        try:
            x, t = zip(*xt)
        except ValueError:
            x, t = 999, 999
        plt.plot(x, t, fmt, label=arg)
        p = self.params
        title = """Wave maxima and current front for %s
                   D = %s, alpha = %s, h1 / H = %s
                   """ % (self.index, p['D/H'], p['alpha'], p['h_1/H'])
        labels = ('maxima', 'g.c. front')
        plt.legend(labels, loc=4)
        plt.title(title)

    def plot(self, args=None, fmts=None):
        """Plot the list of things given in args with the
        given list of formats.
        """
        if args is None:
            args = ['max', 'front']
        elif args is not type(list):
            args = [args]
        if fmts is None:
            fmts = ['bo', 'ko']
        elif fmts is not type(list):
            fmts = [fmts]

        for arg, fmt in zip(args, fmts):
            self.plot_xt(arg, fmt)

        set_plot()
        #rundir = pdir + '/' + self.index
        #fname = rundir + '/plot_' + self.index + '.png'
        fname = "%s/%s.png" % (plots_dir, self.index)
        #plt.show()
        plt.savefig(fname)
        plt.close()

    def plot_speed(self, c):
        t = np.linspace(0, 50, 51)
        x = c * (t - 1)
        plt.plot(x,t)


def conv(Xt, arg):
    Xta = Xt[arg]
    return [[p[0] for p in Xta[i]] for i in range(len(Xta))]

def main(run):
    r = Conjoin(run)
    print "plotting", run, "...\r",
    sys.stdout.flush()
    r.plot()


# # what is the furthest spatial coord?
# Xtm[0] # means list of (x,t) for frame 0
def wave(run='r11_7_06c', data_storage_file=None):
    """Joins together the cam1 and cam2 data to give
    a single data set in (x,t) for a given run.
    Inputs: run_data dictionary
    Returns: Xtm, which is a list of x for given t, i.e. Xtm[0]
    is a list of all the measured x for t=0.
    """
    #data_dir = '/home/eeaol/code/lab-waves/data/'

    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = data_storage + run

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


def bounding_init(t_start, Xtm):
    """Initialise the list of coordinates with the first
    three that are furthest in x. This gives an initial
    bounding region in which to search for more
    points.
    """
    x_maxes = []
    for t in range(t_start, t_start + 3):
        try:
            x_max = max(Xtm[t])
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
    x, T = zip(*x_maxes)
    m, c = np.polyfit(x, T, 1)
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

    return [(x, t) for x in Xtm[t] if (x_down < x < x_up)]

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
        else:
            pass

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
        t_start = get_t_start(Xtm)
        x_maxes = track(t_start, t_end, Xtm, m_err=0, c_err=1)
        Xtm = remove_from(Xtm, x_maxes)
        waves.append(x_maxes)
        # put a line through the wave
        X, T = zip(*x_maxes)
        m, c = np.polyfit(X, T, 1)
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
    plt.plot(x, t)

def plot_xtm(Xtm):
    # make a list of (x,t) tuples over all Xtm
    xt = [(x, t) for t in range(len(Xtm)) for x in Xtm[t]]
    x, t = zip(*xt)
    plt.plot(x, t, 'bo')

def plot_waves(waves):
    """Take a given list of waves and plot them with different colours."""
    for wave in waves:
        x, t = zip(*wave)
        plt.plot(x, t, 'ro')

def test(n=1, run='r11_7_06c', arg='max', start=0, end=25):
    """Produce a plot for visual checking of wave detection.
    TODO: WHAT SHOULD THIS LOOK LIKE??
    """
    no_waves = n
    Xt = conjoin_data(run)
    #nXtm = [[p[0] for p in T] for T in Xtm]
    nXtm = conv(Xt, arg)

    plot_xtm(nXtm)
    waves = get_waves(start, end, nXtm, no_waves)
    plot_waves(waves)

def set_plot():
    plt.xlim(0, 12)
    plt.ylim(0, 50)
    plt.xlabel("Distance (lock lengths)")
    plt.ylabel("Time (s)")
    plt.grid()

def fit_waves(waves):
    for wave in waves:
        x, t = zip(*wave)
        m, c = np.polyfit(x, t, 1)
        T = np.linspace(1, 25, 25)
        X = (T - c) / m
        c = 1 / m
        plt.plot(X, T, label="c=%.2f /s" % c)

def plot_front(run='r11_7_06c', data=None, fmt=None):
    if data is None:
        #data_storage = '/home/eeaol/code/lab-waves/data/data/data_store_'
        data = read_data(data_storage + run)

    if fmt is None:
        fmt = 'ko'

    f = {}
    t = {}
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
    plt.plot(f['cam2'], t['cam2'], fmt, label='g.c. front')

    plt.xlim(0, 13)

