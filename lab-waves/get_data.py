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
import pickle

import matplotlib.pyplot as plt

from aolcore import pull_col, pull_line
import threshold
import peakdetect
import sanity

#####CONFIG#####

# perhaps this is a good place to use a class instead of all of these dicts?
# How about a Camera class, with attributes of rulers, offsets, scales, etc.

# where is the parameters file?
paramf = '/home/eeaol/lab/data/flume1/working/parameters'

# specify a vertical region of the image in which to search for the interface
region = (130, 540)

# specify the threshold values to use. fiddling with these has a strong impact
# on the quality of the interface signal.
thresh_green = (80, 120, 50)
thresh_red = (100, 20, 20)
thresh_values = (thresh_green, thresh_red)

# specify the positions of rulers and other vertical features that
# obscure the fluid.
rulers = {}
rulers['cam1'] = [(80, 105), (870, 950), (1630, 1660), (1740, 1810)]
rulers['cam2'] = [(80, 130), (950, 1000), (1820, 1890), \
                  (2665, 2695), (2705, 2725)]

# specify the offsets that each of the cameras have, for normalisation of
# pixel measurements
camera_offsets = {}
camera_offsets['cam1'] = (2650, 543)
## TODO the 2750 value here is the same as a value in threshold module
## they need to be the same.
## the 2600 value is the distance from the identical place in cam1 to 
## the lock in cam1
camera_offsets['cam2'] = (2750+2600, 543)

# specify the scale, i.e how many pixels to some real measurement in the
# images. in y we want this to be the total fluid depth. in x make it the
# lock length for now (25cm).
fluid_depth = 543-109
lock_length = 440
scales = {}
scales['cam1'] = (lock_length, fluid_depth)
scales['cam2'] = (lock_length, fluid_depth)

# where is the data going to be stored?? (filename)
data_dir = '/home/eeaol/code/lab-waves/data/'
data_storage_file = data_dir + 'data/data_store_'

# specify where the centre of the camera was pointing to.
# used in parallax_corr
centre = {}
centre['cam1'] = 0.75
centre['cam2'] = 2.25

#####/CONFIG#####

def write_data(data_dict, filename):
    """uses pickle to write a dict to disk for persistent storage"""
    output = open(filename, 'wb') # b is binary
    pickle.dump(data_dict, output)
    output.close()

def read_data(filename):
    """reads in a dict from filename and returns it"""
    input = open(filename, 'rb')
    data_dict = pickle.load(input)
    input.close()
    return data_dict

def parallax_corr(xin, cam, p):
    """ Lab images suffer from parallax due to the static cameras.
    This is easily corrected for by assuming that features are 2d
    and homogeneous in the y coord (widthways across the tank."""
    # lock_displacement is chosen to make the reference x a distance
    # of l/2 from the midpoint.
    # unscaled = [s * x for s,x in zip(scales[cam], xy_tuple)] 
    scale = 0.25 # lock-lengths
    unscaled_x = xin * scale
    x_wrt_to_lock = unscaled_x
    x = x_wrt_to_lock
    l = 2 * centre[cam] 
    d = 1.45
    w = 0.20
    # given pos in tank x, fov width l, distance from cam to tank d
    # and width of tank w, the correction is
    corr = p * (x - l/2) * (w / (w + d))
    x_corr = x + corr
    # corr_unscaled = (x_corr, unscaled[1])
    # corr_xy_tuple = tuple([x / s for x,s in zip(corr_unscaled, scales[cam])])
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
    frame = image.split('_')[-1].split('.')[0].split('_')[-1]
    return frame

def irun(image):
    run = image.split('/')[-3]
    return run

def get_parameters(run): 
    p_runs = pull_col(0, paramf) 
    run_params = pull_line(p_runs.index(run), paramf)
    headers = pull_line(0, paramf)
    parameters = dict(zip(headers, run_params))
    return parameters

def get_basic_frame_data(image):
    # get the list of interface depths, with the depth for the current
    # different for varying lock depth
    run = irun(image)
    params = get_parameters(run)
    if params['D/H'] == 0.4:
        front_depth = 525
    elif params['D/H'] == 1:
        front_depth = 505

    print("thresholding image %s..." % image)
    interface, current, front_coord\
            = threshold.main(image, region, rulers, thresh_values, front_depth)

    basic_data = {}
    basic_data['interface'] = interface
    basic_data['current'] = current
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
    if frame == 'img_0001.jpg':
        # calculate the baseline, putting it in the same list/tuple
        # format as the other data
        baseline = [(0,sum(interface)/len(interface))]
        frame_data['baseline'] = norm(baseline, camera)

    # put the interface into the standard format
    interface = [(i,interface[i]) for i in range(len(interface))]
     
    frame_data['interface'] = norm(interface, camera)    
    frame_data['current'] = norm(current, camera)
    frame_data['max'] = norm(_max, camera, 0.5)
    frame_data['min'] = norm(_min, camera, 0.5)
    frame_data['front'] = norm(front_coord, camera, 0.5)

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

## MAIN function
def main(runs=None):
    # make container for all the data
    data = {}
    # define the runs to collect data from
    if runs is None:
        runs = ['r11_7_06c']
    elif type(runs) is not list:
        runs = [runs]
    for run in runs:
        run_data = get_run_data(run)
        data[run] = run_data
        file = data_storage_file + run
        print "writing the data to", file
        write_data(data, file)

def obj_dic(d):
    """a useful method for turning a dict into an object, so that
    d['blah']['bleh'] is the same as d.blah.bleh.
    will work with any level of nesting inside the dict.
    """
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i, type(j)(obj_dic(sj)\
                    if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def get_offset_from_front():
    """Looks at the front trajectory and determines the offset
    between the cameras by minimising the step in value between
    the two cameras."""
    f1xt = [[data[run]['cam1'][fr]['front'][0][0], fr]\
            for fr in sorted(data[run]['cam1'].keys())]
    f2xt = [[data[run]['cam2'][fr]['front'][0][0], fr]\
            for fr in sorted(data[run]['cam2'].keys())]

    f1x = zip(*f1xt)[0]
    f1t = zip(*f1xt)[1]

    f2x = zip(*f2xt)[0]
    f2t = zip(*f2xt)[1]

    

    


