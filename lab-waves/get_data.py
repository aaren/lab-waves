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

def wave_para(inlist, camera):
    # compute the gradient of the inlist
    def get_gradient(inlist):
        X, Z = zip(*inlist)
        # FIXME: are these x values evenly distributed???
        # do we need to reinterpolate?
        gradient = np.gradient(Z)
        grad_list = zip(X, gradient)
        return grad_list

    g_inlist = get_gradient(inlist)
    outlist = []
    # measurements
    w = 0.20
    c = centre[camera]
    # FIXME: need to find out what this was!!
    h = height[camera]

    # how to transform things
    def transform(xa, za):
        x = xa + w/d * (xa - c)
        z = za + w/d * (za - h)
        return x, z

    for xa, za in inlist:
        # gradient of the line that goes through xa, za
        dl = (za - h) / (xa - c)
        D = g_inlist - dl
        if D * (xa - c) > 0:
            # seeing the back of the wave
            x, z = transform(xa, za)
        elif D * (xa -c) < 0:
            # seeing the front of the wave
            x, z = xa, za
        elif D == 0:
            # TODO: or near zero - what are the limits??
            # discard
            x, z = -9999999, -9999999
        else:
            sys.exit("I don't know what's going on")
        outlist.append((x,z))
    return outlist


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
    form of a parabola. Increasing w increases smoothness up to a
    point - that at which a cubic approximation is better.

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

    print "\rthresholding", run, camera, frame,
    sys.stdout.flush()

    interface, current, mixed_current, core_front_coords, mix_front_coords\
        = threshold.main(image, region, rulers, thresh_values, front_depth)
    # print "done"

    basic_data = {}
    basic_data['interface'] = interface
    basic_data['core_current'] = current
    basic_data['mixed_current'] = mixed_current
    basic_data['core_front_coords'] = core_front_coords
    basic_data['mix_front_coords'] = mix_front_coords
    frame_data = (frame, basic_data)

    return frame_data

# multiprocessing and serial implementation
def get_basic_run_data(run, processors=1):
    """grabs all basic data from a run"""
    # run = '11_7_06c'
    # run = run.split('r')[-1]
    cameras = ['cam1', 'cam2']

    def serial(images, camera):
        print "processing", run, camera, len(images), "images"
        result = [get_basic_frame_data(image) for image in images]
        return result

    def parallel(images, camera):
        if processors == 0:
            p = Pool()
        else:
            p = Pool(processes=processors)
        result = p.map_async(get_basic_frame_data, images)
        p.close()
        p.join()
        return result.get()

    basic_run_data = {}
    for camera in cameras:
        images = sorted(glob.glob('/'.join([path,
                            'processed', run, camera, '*jpg'])))
        tot = "%03d" % (len(images))
        if len(images) == 0:
            print "\nno images in", camera
            break
        else:
            pass
        if processors == 1:
            result = serial(images, camera)
        else:
            result = parallel(images, camera)

        basic_run_data[camera] = {k: v for k,v in result}

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
        fname = f.split('/')[-1]
        print "writing", fname
        sys.stdout.flush()
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
    core_front_coords = basic_data['core_front_coords']
    mix_front_coords = basic_data['mix_front_coords']

    # Outlier rejection. arg[1] is point to point variability;
    # arg[2] is window over which interface can be considered
    # parabolic in form.
    fixed_interface = reject_outliers(interface, 20, 200)
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


    # TODO / FIXME: deal with red fluid at bottom!

    # use the current interface from the first image to define a
    # region in which there is red fluid. then filter out all
    # current front coords that fall within thcore_interface

    # OR, use the present images interface. Look for a point where
    # there is a sharp change in the interface height. Again, this
    # has problems with shallow currents as this will appear pretty
    # smooth.

    # This should perhaps be done in the thresholding stage.
    # but here we have access to the first images interface, in
    # run_data_container[camera]['0001'], where in thresholding this
    # data hasn't been written to disk yet.

    # If we take the first image core_interface
    core_i = run_data_container[camera]['0001']['core_current']
    # put into correct format
    core_i = list(enumerate(core_i))
    # bad front coords are those that fall between this and the
    # tank bottom for all subsequent images.
    # front coords consists of a list of (x,z) tuples, measured in
    # image pixels.
    # the tank base is crop[camera][3] pixels from the bottom
    # of the image
    # something like
    for coord in core_front_coords:
        if coord[0] < 0:
            pass
        elif coord[0] >= len(core_i):
            # case that front has been (wrongly) detected behind the
            # lock gate
            core_front_coords.remove(coord)
        elif core_i[coord[0]] < coord[1] - 5 < region[1]:
            # ignore the point if in bad region, plus some buffer
            core_front_coords.remove(coord)
        elif region[0] < coord[1] - 5 < core_i[coord[0]]:
            # accept the point
            pass
        else:
            print "Something has gone wrong!"
            sys.exit('Check the front coords')

    # same for the mixed fluid
    mix_i = run_data_container[camera]['0001']['mixed_current']
    mix_i = list(enumerate(mix_i))
    for coord in mix_front_coords:
        if coord[0] < 0:
            pass
        elif coord[0] > len(mix_i):
            # case that front has been (wrongly) detected behind the
            # lock gate
            mix_front_coords.remove(coord)
        elif mix_i[coord[0]] < coord[1] - 5 < region[1]:
            # ignore the point if in bad region, plus some buffer
            mix_front_coords.remove(coord)
        elif region[0] < coord[1] - 5 < mix_i[coord[0]]:
            # accept the point
            pass
        else:
            print "Something has gone wrong!"
            sys.exit('Check the front coords')

    # need to catch case that there is fluid along the bottom
    # reject the outliers? NO. won't work for front that is shallow.
    # def _reject_outliers(data):
        # return [e for e in data if abs(e[0] - np.mean(data)) < np.std(data)]
    # fcfc = reject_outliers(core_front_coord)
    # fmfc = reject_outliers(core_front_coord)
    # Make the front_coord the front_coord furthest from the lock.
    min_core_front_coord = min(core_front_coords, key=lambda k: abs(k[0]))
    min_mix_front_coord = min(mix_front_coords, key=lambda k: abs(k[0]))
    front_coord = [min(min_core_front_coord, min_mix_front_coord)]

    def find_head(f_coords, thresh=100):
        # find the head of the current by looking for a flat bit
        for i,p in enumerate(f_coords):
            try:
                p1 = f_coords[i+1]
                if p[0] < 0:
                    pass
                elif thresh < p1[0] - p[0] < 99999:
                    return [p]
                elif p1[0] - p[0] < -99999:
                    return [p]
                else:
                    pass
            except IndexError:
                pass
        return [(-999999, 0)]
    core_head = find_head(core_front_coords)
    mix_head = find_head(mix_front_coords)
    head_coord = core_head

    # SANITY CHECKING: overlay given interfaces and points onto the
    # images with specified colours. images are saved to the sanity
    # dirs.
    interfaces = [interface, core_current, \
            mixed_current, fixed_interface, smoothed_interface]
    icolours = ['black', 'blue', 'cyan', 'orange', 'red']
    points = [_max, _min, \
            core_front_coords, mix_front_coords, \
            core_max, mix_max,
            head_coord, front_coord]
    pcolours = ['green', 'purple', \
            'blue', 'cyan', \
            'green', 'purple', \
            'black', 'orange']
    threshold.sanity_check(interfaces, points, image, icolours, pcolours)

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
    frame_data['core_front'] = norm(core_front_coords, camera, 2)
    frame_data['mix_front'] = norm(mix_front_coords, camera, 2)
    frame_data['front'] = norm(front_coord, camera, 2)
    frame_data['head'] = norm(head_coord, camera, 2)

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
        f = data_storage + run
        # print "\nwriting ", file, "...\r",
        sys.stdout.flush()
        write_data(data, f)
        fname = f.split('/')[-1]
        print "wrote ", fname
        sys.stdout.flush()
