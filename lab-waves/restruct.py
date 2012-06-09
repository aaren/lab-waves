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

indata = read_data(data_file)
rdata = indata[run]

def points(arg):
    arg_list = []
    point = namedtuple(arg, 'x, z, t')

    for cam in ['cam1', 'cam2']:
        cdata = rdata[cam]
        for f in cdata.keys()
            arg_list += [point(e[0], e[1], int(f)) for e in cdata[f][arg]]

    return arg_list

# Now big_list contains a list of all the points that come out for a
# given arg in the form (x, z, t).

# TODO: Decide whether to use a tuple (x,z,t) or a dict {'x':,
# 'z':,'t':} for each of the points.

# ANSWER: Use a named tuple. We then know which arg each point came
# from in the first place (can we actually access this though??) and
# the point can be accessed as an object.

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
