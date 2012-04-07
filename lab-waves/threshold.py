# force division to be floating by default (as in python 3).
# floored division is done using //
from __future__ import division
import Image

def thresh_img(image, thresh_values=None):
    # Threshold the image, determining the fluid type of each individual
    # pixel

    # open the image and load it to memory.
    im = Image.open(image)
    source = im.load()

    # get the image dimensions
    w, h = im.size

    # defaults
    thresh_green = (80, 110, 50)
    thresh_red = (60, 20, 20)
    mixed_red = (75, 20, 20)
    if thresh_values:
        thresh_green, thresh_red, mixed_red = thresh_values
    
    # determine the fluid type throughout the image. remember brackets
    # around the logical expressions!!!!!
    # fluid_type is a list of lists. would this not be better implemented as
    # a numpy array?
    fluid_type = [[0 if (source[i,j][0] > thresh_red[0]) &\
                        (source[i,j][1] < thresh_red[1]) &\
                        (source[i,j][2] < thresh_red[2]) 
                    else 0.5 if (source[i,j][0] > mixed_red[0]) &\
                                (source[i,j][1] < mixed_red[1]) &\
                                (source[i,j][2] < mixed_red[2])\
                    else 1 if (source[i,j][0] > thresh_green[0]) &\
                              (source[i,j][1] > thresh_green[1]) &\
                              (source[i,j][2] < thresh_green[2])\
                           else 'other' \
                    for j in range(h)] for i in range(w)]

    # i wonder if this can be implemented using numpy arrays and map?
    # np arrays are much more compact than list of lists. could use map with
    # an explicit lamda function??
    # this can be done, but the problem with np arrays is that they can't
    # be indexed in the same way as lists, so the process function below
    # would need to be modified as indexing the first occurence of '1' etc.
    # is the present way to detect the interface.
    # maybe not necessary. the list of lists works fast enough so why bother
    # fiddling with it??
    # list comp is evaluated in c anyway, so the performance benefit is 
    # probably not worth it, especially when the code is as clear as it is.

    # output the list of lists that specifies the fluid type of individual
    # pixels throughout the image.
    return fluid_type


def process(image, fluid_type_lists, region, fluid, rulers):
    # Now make lists of the index at which a particular fluid type first
    # occurs.
    # range is a two number tuple, specifying the bounds on the threshold
    # process, if any. 
    # TODO defaults?************
    # the tuple is (upper pixel in the image, lower pixel in the image),
    # i.e. the first one should be a *lower* number.
    # fluid is the type of the fluid, specified by 0 or 1
    top, bottom = region
    depth = []

    # set the horizontal bounds dependent on the camera 
    camera = image.split('/')[-2]

    if camera == 'cam1':
        l_lim = 20 
        r_lim = 2700 # this is approximate and discards lock parallax
    elif camera == 'cam2':
        l_lim = 0
        r_lim = 2800

    # pad depth out with zero so that the index of depth is equivalent
    # to the pixel image coordinate in x.
    for i in range(l_lim):
        depth.append(0)

    for i in range(l_lim, r_lim): 
        # range does matter as we don't want to go into the lock fluid
        # this has been standardised across images though so simple to 
        # specify the range, although it varies from cam1 to cam2
        # print "indexing column %s" %i
        try:
            # if looking where a ruler is, just copy the previous value
            cond_list = [(ruler[0] < i < ruler[1]) for ruler in rulers[camera]]
            if any(cond_list):
                pixel = depth[-1]
                # interpolation requires another pass, in a separate function
                # below
            else:
                pixel = fluid_type_lists[i].index(fluid, top, bottom)

        # sometimes none of the pixels match (e.g. with a ruler in the way)
        except ValueError:
            # copy the previous value and hope we don't start off with rubbish
            try:
                pixel = depth[-1]
            except IndexError:
                pixel = 0
        # print "column %s has pixel depth %s for fluid %s" % (i, pixel, fluid)
        # put the computed interface depth into the list of depths by pixel
        depth.append(pixel) 

    # change the first l_lim depths into something more sensible than zero
    for i in range(l_lim):
        section = depth[l_lim + 1: l_lim + 25]
        depth[i] = sum(section)/len(section)
        
    return depth

def interpolate(image, in_list, rulers):
    """
    Takes a list of interface depths and changes locations with a ruler
    present from a constant value to an interpolation between the two edges.
    
    arguments:
    image -- the image being operated on (only to find out camera type from
        the path).
    in_list -- the list of interface depths, as outputted by process()

    return -- a list of interface depths with the interpolation applied.
    """
    interface = in_list

    def interp((x1, x2)):
        y1 = interface[x1]
        y2 = interface[x2]
        for i in range(x1, x2):
            interface[i] = (y2 - y1) * ((i - x1) / (x2 - x1)) + y1 

    camera = image.split('/')[-2]
    for ruler in rulers[camera]:
        interp(ruler)

    return interface

def main(image, region, rulers, thresh_values=None, front_depth=None):
    top, bottom = region
    # generate fluid type list of lists
    # print('generating threshold array...')
    fluid_type = thresh_img(image, thresh_values)
    
    # detect and interpolate the interface
    # print('detecting the interface')
    interface = process(image, fluid_type, region, 1, rulers)
    interp_interface = interpolate(image, interface, rulers) 

    # detect the current front
    # print('detecting the front...')
    # fluid_type_transpose = zip(*fluid_type)
    # front_pos = fluid_type_transpose[510].index(0)
    # much faster, given we know the row we want (70ms / 200us)
    # but the zip transpose is still very quick.
    if not front_depth:
        front_depth = 510
    try:
        tot = 0
        d = 5
        for n in range(d):
            front_pos = [i[front_depth - n] for i in fluid_type].index(0)
            tot += front_pos
        front_pos = int(tot / d)
    # if the front isn't found
    except ValueError:
        front_pos = -99999
    # catch the case that the front has neared the end of the tank
    if front_pos < 110:
        front_pos = -99999
    front_coord = [(front_pos, front_depth)]

    # detect and interpolate the current
    current = process(image, fluid_type, region, 0, rulers)
    # remove silly current values
    current[:front_pos] = [bottom]*front_pos
    interp_current = interpolate(image, current, rulers)

    mix_current = process(image, fluid_type, region, 0.5, rulers)
    mix_current[:front_pos] = [bottom]*front_pos
    interp_mix_current = interpolate(image, mix_current, rulers)

    out = (interp_interface, interp_current, interp_mix_current, front_coord)

    return out
