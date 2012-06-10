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
from collections import namedtuple

def points(arg):
    arg_list = []
    point = namedtuple(arg, 'x, z, t')
    for cam in ['cam1', 'cam2']:
        cdata = rdata[cam]
        for f in cdata.keys()
            arg_list += [point(e[0], e[1], int(f)) for e in cdata[f][arg]]
    return arg_list

# TODO: Decide whether to use a tuple (x,z,t) or a dict {'x':,
# 'z':,'t':} for each of the points.

# ANSWER: Use a named tuple. We then know which arg each point came
# from in the first place (can we actually access this though??) and
# the point can be accessed as an object.

indata = read_data(data_file)
rdata = indata[run]

# We can then plot all of the maxima
maxima = points('max')
X, Z, T = zip(*maxima)
plt.plot(X, T, 'ro')

# Or indeed the maxima and the front as indistinguishable points
all_args = points('max') + points('front')
X, Z, T = zip(*all_args)
plt.plot(X, T, 'bo')

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
    # plot the argument
    X, Z, T = zip(*arg)
    plt.plot(X, T, 'bo')
    # get user input points
    pts = ginput(2,0)
    # calculate the limits
    hi, lo = bounding_lines(pts, 0, 0.5)
    # select points within limits
    line = [p for p in maxima \
                        if ((p.t - hi.c) / hi.m <  p.x < (p.t - lo.c) / lo.m)]
    # plot again
    Xs, Zs, Ts = zip(*line)
    plt.plot(Xs, Ts, 'ro')
    plt.close()
    return line

