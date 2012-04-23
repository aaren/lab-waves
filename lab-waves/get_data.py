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

from __future__ import division
import glob
import sys
from multiprocessing import Pool
import time

import numpy as np

from sgolay import savitzky_golay as sgolay
import peakdetect

from aolcore import write_data, read_data
from aolcore import get_parameters
import threshold
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
    def norm_tuple(intupe):
        offsets = camera_offsets[camera]
        scale = scales[camera]
        input_norm = [(o - i) / s for o, i, s in zip(offsets, intupe, scale)]
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

def icam(image):
    cam = image.split('/')[-2]
    return cam

def reject_outliers(inter, r, w, degree=2):
    """Running through inter, if any element is outside range r
    either side of the previous element then replace it with the
    value of the previous element.

    The problem with this method is that it can only deal with
    anomalous spikes and not ramps. If the transition to anomaly
    is sufficiently smooth then we get stuck > r away from truth.

    We could make r small (~5), but this would cause problems when
    there are steeper slopes.

    Try calculating the gradient of the last few points and
    extrapolating from there.

    This will lead to problems at extrema. Could try quadratic
    extrapolation.

    w determines the width of window over which to calculate
    the prediction curve.

    r and w are dependent on the form of the data. w should be
    set to the length-scale over which the data have the approximate
    form of a parabola.

    r should be large enough to accommodate the normal variability
    of the data.

    This still struggles with steps.
    """
    # Initialise with an average of the first few points, which we
    # hope is sensible.
    def fit(inter, j, k, deg):
        x = range(j, k)
        h = inter[j:k]
        return np.polyfit(x, h, deg)

    def pred(inter, i, w, deg):
        """Predict the value of cell i, based on polynomial extrapolation
        of degree deg over the last w cells for series inter.
        """
        coeff = fit(b, i - w, i, deg)
        [i ** n * coeff[::-1][n] for n in range(len(coeff))]
        pred_i = sum([i ** n * coeff[::-1][n] for n in range(len(coeff))])
        return pred_i

    # initialise
    m, n = fit(inter, 0, w, 1)
    init = [i * m + n for i in range(w)]
    b = init[:]

    # list comp won't work here as b has to refer to its old self
    #b = [inter[i] if (pred(b, i, w, d) - r < inter[i] < pred(b, i, w, d) + r) \
    #                else pred(b, i, w, d) for i in range(w, len(inter))]

    for i in range(w, len(inter)):
        pred_i = pred(b, i, w, degree)
        if (pred_i - r < inter[i] < pred_i + r):
            b.append(inter[i])
        else:
            b.append(pred_i)

    return b

def smooth(inter, window):
    """Apply Savitzky-Golay smoothing to the given data (1D)
    over a given window width.
    """
    x = np.asarray(inter)
    y = sgolay(x, window, 2)
    smoothed_inter = list(y)
    return smoothed_inter

def get_basic_frame_data(image):
    # get the list of interface depths, with the depth for the current
    # different for varying lock depth
    run = irun(image)
    camera = icam(image)
    frame = iframe(image)

    params = get_parameters(run, paramf)
    front_depth = front_depths[params['D/H']]

    #print "thresholding processed", \
    #        run, camera, frame, "of"
    #sys.stdout.flush()

    interface, current, mixed_current, core_front_coord, mix_front_coord\
            = threshold.main(image, region, rulers, thresh_values, front_depth)

    basic_data = {}
    basic_data['interface'] = interface
    basic_data['core_current'] = current
    basic_data['mixed_current'] = mixed_current
    basic_data['core_front_coord'] = core_front_coord
    basic_data['mix_front_coord'] = mix_front_coord

    frame_data = (frame, basic_data)
    return frame_data

def get_basic_run_data(run, processors=1):
    """grabs all basic data from a run"""
    # run = '11_7_06c'
    # run = run.split('r')[-1]
    basic_run_data = {}
    for camera in ('cam1', 'cam2'):
        images = sorted(glob.glob('/'.join([path,
                            'processed', run, camera, '*jpg'])))
        tot = "%03d" % len(images)
        if len(images) == 0:
            break
        p = Pool(processes=processors)
        # for i, _ in enumerate(p.map_async(get_basic_frame_data, images), 1):
                # # print "Thresholding... %02d%%\r"
                # sys.stderr.write('\rThresholding {0:%}'.format(i/len(images)))
        result = p.map_async(get_basic_frame_data, images)

        # while True:
            # if result.ready():
                # break
            # remain = "%03d" % result._number_left
            # print "Thresholding...", run, remain, "left of", tot, "\r",
            # sys.stdout.flush()
            # time.sleep(0.1)

        p.close()
        p.join()

        basic_run_data[camera] = {k: v for k,v in result.get()}
    print ""
    return basic_run_data

def get_basic_data(runs=None, processors=1):
    if runs is None:
        runs = ['r11_7_06c']
    elif type(runs) is not list:
        runs = [runs]
    if 'r' not in runs[0]:
        print "runs must lead with an r!!"
        return 0
    for run in runs:
        basic_run_data = get_basic_run_data(run, processors)
        f = data_dir + 'basic/basic_%s' % run
        print "writing the data to", f
        write_data(basic_run_data, f)

def get_frame_data(image, run_data_container):
    """gets the data for a single image.
    runs the external threshold module to produce the data,
    then normalises it and puts it in a dictionary for storage"""

    sp = image.split('/')
    frame = iframe(image)
    camera = sp[-2]

    basic_data = run_data_container[camera][frame]
    interface = basic_data['interface']
    core_current = basic_data['core_current']
    mixed_current = basic_data['mixed_current']
    core_front_coord = basic_data['core_front_coord']
    mix_front_coord = basic_data['mix_front_coord']

    # Make the front_coord the front_coord furthest from the lock.
    front_coord = min(core_front_coord, mix_front_coord)

    # Outlier rejection. arg[1] is point to point variability;
    # arg[2] is window over which interface can be considered
    # parabolic in form.
    fixed_interface = reject_outliers(interface, 20, 50)
    # SMOOTHING (Savitzky-Golay). Preferable to moving avg as it
    # doesn't phase shift or crush peaks. Supplied number is the
    # window.
    smoothed_interface = smooth(fixed_interface, 301)
    # current profile is a bit too messy for the rejection to work
    #core_current = reject_outliers(core_current, 50, 20)
    #mixed_current = reject_outliers(mixed_current, 50, 20)

    # get the lists of the positions of maxima and minima.
    # at this point they are maxima in DEPTH! so MINIMA in height.
    _min, _max = peakdetect.peakdetect(smoothed_interface, None, 200, 10)
    core_min, core_max = peakdetect.peakdetect(core_current, None, 200)
    mix_min, mix_max = peakdetect.peakdetect(mixed_current, None, 200)

    # put the interfaces into the standard format
    interface = list(enumerate(interface))
    fixed_interface = list(enumerate(fixed_interface))
    smoothed_interface = list(enumerate(smoothed_interface))
    core_current = list(enumerate(core_current))
    mixed_current = list(enumerate(mixed_current))

    # SANITY CHECKING
    # overlay given interfaces and points onto the images with specified
    # colours. images are saved to the sanity dirs.
    interfaces = [interface, core_current, \
            mixed_current, smoothed_interface, fixed_interface]
    icolours = ['black', 'blue', 'cyan', 'orange', 'red']
    points = [_max, _min, core_front_coord, mix_front_coord, core_max, mix_max]
    pcolours = ['green', 'purple', 'blue', 'cyan', 'blue', 'cyan']
    sanity.sanity_check(interfaces, points, image, icolours, pcolours)

    # make a container for the data and populate it
    frame_data = {}
    # need the baseline when doing amplitude deviations
    if frame == iframe('img_0001.jpg'):
        baseline = [(0, sum(zip(*smoothed_interface)[1])\
                                         / len(smoothed_interface))]
        frame_data['baseline'] = norm(baseline, camera)

    frame_data['interface'] = norm(interface, camera)
    frame_data['core_current'] = norm(core_current, camera)
    frame_data['mixed_current'] = norm(mixed_current, camera)
    frame_data['max'] = norm(_max, camera, 0)
    frame_data['min'] = norm(_min, camera, 0)
    frame_data['core_max'] = norm(core_max, camera, 0)
    frame_data['core_min'] = norm(core_min, camera, 0)
    frame_data['mix_max'] = norm(mix_max, camera, 0)
    frame_data['mix_min'] = norm(mix_min, camera, 0)
    frame_data['core_front'] = norm(core_front_coord, camera, 1)
    frame_data['mix_front'] = norm(mix_front_coord, camera, 1)
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
        images = sorted(glob.glob('/'.join([path,
                            'processed', run, camera, '*jpg'])))
        for image in images:
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
        file = data_storage + run
        print "\nwriting ", file, "...\r",
        sys.stdout.flush()
        write_data(data, file)
        print "writing ", file, "...done"
        sys.stdout.flush()
