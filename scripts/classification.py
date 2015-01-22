# Helper script for run classification

# Each run has a load of characteristics that aren't completely
# straightforward to diagnose numerically. The idea here is to
# present each run and ask some questions about it.

# QUESTIONS
#
# 1) Wave of depression or elevation?
#
# 2) Significant amplitude?
#
# 3) Faster than current?

# The wave amplitude should be diagnosable numerically, it just
# requires some computing time.

# In fact, all of these things are possible numerically. Doing
# them by eye is prone to error and not as quantitative, but it
# does have some benefits: qualitative information can be noted
# for each run and the automated results can be validated.

# However these results are obtained, they need to be presented
# in some manner beyond a boring table. At the very least, a map
# of parameter space (alpha, h1) indicating super / sub critical
# and the significance of any produced waves.
#
# In this map, data points are degenerate in alpha with a variety
# of density profiles giving the same value. These data will need
# to be presented clearly as they overlap. Is there another way
# to map the data that would obviate this need? i.e. give data
# points a singular point in parameter space? Alpha just tells
# us about the buoyancy *ratio*, not the *absolute* densities
# involved that determine the speed of waves on the interface.

# Without doing any wave tracking, we can calculate the peak
# interface displacement in a given run. We know the baseline,
# so just look for the largest displacements from this in the
# interface data over all of the time steps.

# Speed relative to the current requires wave tracking, but can
# be done qualitatively by eye.

# Front speed alone is relatively easy. This only requires that
# the offset between the runs is properly set. We could go for
# a plot of front speed for each run, with some detection of 
# any kinks in the data indicating an abrupt change in speed
# and a front speed for the different regions.

# But looking at the front speed leads to lots of questions.
# The transisition in the front speed is predicted by Hoyler
# (see RS83 eq 4.1), but this gets modified variously by the
# two layer setup. All currents eventually slow down, as they
# are constant volume, and so all waves will eventually outrun
# the current.

# Simplest thing to look at is the direct atmospheric relevance,
# i.e. the amplitude of the waves. Then we can look at speeds.
# 
# Compute the run data, but no wave tracking. Then look through
# the max / min data across an entire run and find the largest
# deviation from the baseline for that run. This single number
# is what I'm looking for for each run. 

# max / min have format (pos, amp). 

# **
# actually, the amplitude information might provide a way to
# track the waves, assuming that their amplitude doesn't change.
# **

# this will be sped up by introducing a cut off for each run
# in the number of images that are processed. this can either
# be a set number (~30) or vary with run by inspection.

from __future__ import division

import glob
import os

def nondimensional(px):
    bottom = 540
    ruler = 435
    nd = (bottom - px) / ruler
    return nd

def get_simple_amplitudes(rundir, run):
    try:
        f = open('simple_amplitudes', 'r')
        if run in [line.split(',')[0] for line in f.readlines()]:
            return
        f.close()
    except IOError:
        pass

    command = 'gimp -s -f -d %s/%s/join/img_0012.jpg &' % (rundir, run)
    os.system(command)

    print "A different image? Return if ok, 999 to skip run"
    other_image = raw_input('> ')
    if other_image == '999':
        return
    # elif other_image:
    #     command = 'gimp -s -f -d processed/%s/join/img_%04d.jpg &' \
    #                                             % (run, int(other_image))
    print "What is the baseline?"
    baseline = int(raw_input('> '))
    print "What is the amplitude of the wave peak?"
    amplitude = int(raw_input('> '))

    # convert px to dimensionless
    amp = nondimensional(amplitude)
    base = nondimensional(baseline)
    dev = amp - base

    entry = "%s,%.3f,%.3f,%.3f\n" % (run, base, amp, dev)
    f = open('simple_amplitudes', 'a')
    f.write(entry)
    f.close()

def main():
    rundir = '/home/eeaol/lab/data/flume1/working/processed_11_12_2'
    runs = [path.split('/')[-1] for path in glob.glob(rundir + '/*')]
    for run in runs:
        get_simple_amplitudes(rundir,run)

main()
