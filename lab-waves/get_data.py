# V2 of get_data. this is a refactoring of v1.
# want to separate out thresholding and more advanced processing.
# the thresholding is the slowest part presently and it is quite 
# stable and well behaved now --> makes sense to only do it once
# and store the data in some intermediate file.
# Have implemented this. intermediate files are 'basic/basic_%s' % run
# basic refers to the data that threshold extracts from the image.


# thresholding: goes through an image column by column applying a logical test
# to each pixel value in the column to determine whether the pixel represents
# green fluid or not. then, from the top down, the first pixel that passes
# this test has it's y coordinate output to a list.
# 
# so thresholding returns a list that has the length of the pixel array width.
# 
# peakdetect is a module that includes a method to make a list of the
# coordinates of maxima and minima in some signal.
#
# the measured interface and inferred peaks are then overlaid onto the target
# image for inspection.
#
# TODO 1) Wavelength measurement
# this is actually a bit involved. to calculate fwhm, need to know deviation
# from baseline and whilst this can be calculated for an initial baseline, it
# doesn't deal with the case of an undular bore, in which the baseline is
# shifted vertically.
# however, a baseline is needed anyway
#      2) baseline detection

from __future__ import division
import glob

import matplotlib.pyplot as plt

from aolcore import pull_col, pull_line, write_data, read_data
from aolcore import get_parameters
import threshold
import peakdetect
import sanity

from config import *

def parallax_corr(xin, cam, p):
    """ Lab images suffer from parallax due to the static cameras.
    This is easily corrected for by assuming that features are 2d
    and homogeneous in the y coord (widthways across the tank."""
    # xin is in units of lock-lengths
    scale = 0.25 
    x = xin * scale
    # basic tank geometry
    c = centre[cam]
    d = 1.45
    w = 0.20
    f = (w / (2 * d + w))
    # given pos in tank x, cam pos c, distance from cam to tank d
    # and width of tank w, the correction, when projected onto the
    # centreline of the tank, is
    corr = p * (x - c) * f
    # whether this is positive or negative depends on the position of
    # the front w.r.t the cam position
    if x < c:
        x_corr = x + corr
    elif x > c:
        x_corr = x - corr
    else:
        x_corr = x
    # back into lock-lengths
    scale_x_corr = x_corr / scale

    return scale_x_corr

def norm(inlist, camera, p=0):
    """ takes a list of tuples and normalises them according to
    pre-defined camera offsets and scalings. p=0 (default), means
    no parallax correction. p!=0 switches it on (for the x-value),
    scaling the parallax correction factor by p.
    """
    def norm_tuple(input):
        offsets = camera_offsets[camera]
        scale = scales[camera]
        input_norm = [(o - i)/s for o,i,s in zip(offsets, input, scale)]
        # apply the parallax correction to the tuple x
        if p != 0:
            input_norm[0] = parallax_corr(input_norm[0], camera, p)
        input_norm = tuple(input_norm)
        return input_norm
    inlist_norm = [norm_tuple(tupe) for tupe in inlist]
    return inlist_norm

def iframe(image):
    """From an image filename, e.g. img_0001.jpg, get just the
    0001 bit and return it.
    """
    frame = image.split('_')[-1].split('.')[0].split('_')[-1]
    return frame

def irun(image):
    run = image.split('/')[-3]
    return run

def get_basic_frame_data(image):
    # get the list of interface depths, with the depth for the current
    # different for varying lock depth
    run = irun(image)
    params = get_parameters(run, paramf)
    if params['D/H'] == '0.4':
        front_depth = 525
    elif params['D/H'] == '1':
        front_depth = 505

    print("thresholding image %s..." % image)
    interface, current, mix_current, front_coord\
            = threshold.main(image, region, rulers, thresh_values, front_depth)

    basic_data = {}
    basic_data['interface'] = interface
    basic_data['core_current'] = current
    basic_data['mixed_current'] = mix_current
    basic_data['front_coord'] = front_coord

    return basic_data

def get_basic_run_data(run):
    """grabs all basic data from a run"""
    # run = '11_7_06c'
    # run = run.split('r')[-1]
    basic_run_data = {}
    for camera in ('cam1', 'cam2'):
        basic_run_data[camera] = {}
        cam_data = basic_run_data[camera]
        for image in sorted(glob.glob(data_dir + run+'/' + camera + '/*jpg')):
            frame = iframe(image)
            cam_data[frame] = get_basic_frame_data(image)
    return basic_run_data

def get_basic_data(runs=None):
    if runs is None:
        runs = ['r11_7_06c']
    elif type(runs) is not list:
        runs = [runs]
    if 'r' not in runs[0]:
        print "runs must lead with an r!!"
        return 0
    for run in runs:
        basic_run_data = get_basic_run_data(run)
        file = data_dir + 'basic/basic_%s' % run
        write_data(basic_run_data, file)

def get_frame_data(image, run_data_container):
    """gets the data for a single image.
    runs the external threshold module to produce the data,
    then normalises it and puts it in a dictionary for storage"""
    
    sp = image.split('/')
    frame = iframe(image)
    camera = sp[-2]
    
    basic_data = run_data_container[camera][frame] 
    interface = basic_data['interface']
    current = basic_data['current']
    front_coord = basic_data['front_coord']

    # get the lists of the positions of maxima and minima.
    # at this point they are maxima in DEPTH! so MINIMA in height.
    # print("detecting the peaks")
    _min, _max = peakdetect.peakdetect(interface, None, 100, 10)

    # check that the front and wave peaks make sense by overlaying
    # measured positions onto the source image and writing this out
    # to the sanity directories
    sanity.sanity_check(interface, _max, _min, front_coord, image, current)

    # make a container for the data and populate it
    frame_data = {}
    # need the baseline when doing amplitude deviations
    #FIXME frame identity incorrect
    if frame == '0001':

        # calculate the baseline, putting it in the same list/tuple
        # format as the other data
        baseline = [(0,sum(interface)/len(interface))]
        frame_data['baseline'] = norm(baseline, camera)

    # put the interface into the standard format
    interface = list(enumerate(interface))
    current = list(enumerate(current))
     
    frame_data['interface'] = norm(interface, camera)    
    frame_data['current'] = norm(current, camera)
    frame_data['max'] = norm(_max, camera, 0.5)
    frame_data['min'] = norm(_min, camera, 0.5)
    frame_data['front'] = norm(front_coord, camera, 1)

    return frame_data

def get_run_data(run):
    """grabs all data from a run"""
    # run = '11_7_06c'
    basic_run_data = read_data(data_dir + 'basic/basic_%s' % run)
    run_data = {}
    for camera in ('cam1', 'cam2'):
        run_data[camera] = {}
        cam_data = run_data[camera]
        for image in sorted(glob.glob(data_dir + run+'/' + camera + '/*jpg')):
            frame = iframe(image)
            cam_data[frame] = get_frame_data(image, basic_run_data)
    return run_data

def main(runs=None):
    # define the runs to collect data from
    if runs is None:
        runs = ['r11_7_06c']
    elif type(runs) is not list:
        runs = [runs]
    for run in runs:
        # make container for all the run data
        data = {}
        run_data = get_run_data(run)
        data[run] = run_data
        file = data_storage_file + run
        print "writing the data to", file
        write_data(data, file)
