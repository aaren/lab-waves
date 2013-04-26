# force division to be floating by default (as in python 3).
# floored division is done using //
from __future__ import division
import os
import sys

import Image
import ImageDraw
from numpy import polyfit

from config import crop

def thresh_img(image, thresh_values=None):
    # TODO: this can definitely be replaced with an algorithm from
    # skimage.

    # Threshold the image, determining the fluid type of each
    # individual pixel. Return the list of lists that specifies the
    # fluid type of individual pixels throughout the image.

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

    # remember brackets around the logical expressions!!!!!
    fluid_type = [[0 if (source[i,j][0] > thresh_red[0]) &\
                        (source[i,j][1] < thresh_red[1]) &\
                        (source[i,j][2] < thresh_red[2])
                    else 3 if (source[i,j][0] > mixed_red[0]) &\
                                (source[i,j][1] < mixed_red[1]) &\
                                (source[i,j][2] < mixed_red[2])\
                    else 1 if (source[i,j][0] > thresh_green[0]) &\
                              (source[i,j][1] > thresh_green[1]) &\
                              (source[i,j][2] < thresh_green[2])\
                           else 'other' \
                    for j in range(h)] for i in range(w)]

    # surely this can be written as a mapping? just need to find a
    # way of indexing the result.
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
        r_lim = 2350 # this is approximate and discards lock parallax
    elif camera == 'cam2':
        l_lim = 20
        r_lim = crop['cam2'][1] - crop['cam2'][0]

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
    # extrapolation of the subsequent trend
    def extrapolate(in_depths, in_depths_low, in_depths_hi, extra_point):
        x1, x2 = in_depths_low, in_depths_hi
        x = range(x1, x2)
        d = in_depths[x1:x2]
        m, c = polyfit(x, d, 1)
        extra_depth = m * extra_point + c
        return extra_depth

    for i in range(l_lim):
        epoint = extrapolate(depth, l_lim + 1, l_lim +50, i)
        depth[i] = int(round(epoint))

    # this just sets them to an average of the next 25 depths.
    # for i in range(l_lim):
    #     section = depth[l_lim + 1: l_lim + 25]
    #     depth[i] = sum(section)/len(section)

    return depth


def interpolate(image, in_list, rulers):
    """ Takes a list of interface depths and changes locations with a ruler
    present from a constant value to an interpolation between the two edges.

    arguments:
    image -- the image being operated on (only to find out camera type from
        the path).
    in_list -- the list of interface depths, as outputted by process()

    return -- a list of interface depths with the interpolation applied.
    """
    interface = in_list[:]

    def interp((x1, x2)):
        y1 = interface[x1]
        y2 = interface[x2]
        for i in range(x1, x2):
            interface[i] = (y2 - y1) * ((i - x1) / (x2 - x1)) + y1

    camera = image.split('/')[-2]
    for ruler in rulers[camera]:
        interp(ruler)

    return interface


def main(image, region, rulers, thresh_values=None, front_depths=None):
    top, bottom = region
    # generate fluid type list of lists
    # print('generating threshold array...')
    fluid_type = thresh_img(image, thresh_values)

    # detect the current front
    # print('detecting the front...')
    # fluid_type_transpose = zip(*fluid_type)
    # front_pos = fluid_type_transpose[510].index(0)
    # much faster, given we know the row we want (70ms / 200us)
    # but the zip transpose is still very quick. tradeoff comes when
    # we want to get more than 140 rows.
    def get_front_pos(fluid, f_depth=425, d=5):
        # find the front position at front_depth over d adjacent
        # depths and average.
        try:
            tot = 0
            for n in range(d):
                f_pos = [i[f_depth - n] for i in fluid_type].index(fluid)
                tot += f_pos
            f_pos = int(tot / d)
        # if the front isn't found
        except ValueError:
            f_pos = -999999
        # catch the case that the front has neared the end of the tank
        # FIXME: is this still necessary or are we throwing away
        # data?
        if f_pos < 20:
            f_pos = -999999
        return f_pos

    front_coord_core = [(get_front_pos(0, front_depth), front_depth) \
                        for front_depth in front_depths]
    front_coord_mix = [(get_front_pos(3, front_depth), front_depth) \
                        for front_depth in front_depths]

    # detect the interfaces
    interface = process(image, fluid_type, region, 1, rulers)
    core_current = process(image, fluid_type, region, 0, rulers)
    mix_current = process(image, fluid_type, region, 3, rulers)

    # qc the data by ignoring values that are outside the tank
    for i,d in enumerate(core_current):
        if top < d < bottom:
            pass
        else:
            core_current[i] = bottom
    for i,d in enumerate(mix_current):
        if top < d < bottom:
            pass
        else:
            mix_current[i] = bottom

    # interpolate
    interp_interface = interpolate(image, interface, rulers)
    interp_core_current = interpolate(image, core_current, rulers)
    interp_mix_current = interpolate(image, mix_current, rulers)

    out = (interp_interface, interp_core_current, interp_mix_current, \
                    front_coord_core, front_coord_mix)

    return out


def sanity_check(interfaces, points, image, icolours, pcolours):
    """produces an image that has the calculated interface
    and the inferred peak coordinates and front position overlaid.

    args: interfaces -- a list of interface depth lists
          maxima -- the list of maxima coordinates
          minima -- the list of minima coordinates
          front -- the coordinate of the current front
          colous -- a list of colours with which to draw the interfaces

    return: none. Saves an image in the sanity_dir
    """

    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    # plot the measured interface depth onto the image
    for inter, colour in zip(interfaces, icolours):
        draw.line(inter, fill = colour, width = 5)

    # plot squares onto the image at the given points
    rectangle_size = (7,7)
    for point, colour in zip(points, pcolours):
        for coord in point:
            xy = [(coord[0] + rectangle_size[0], coord[1] + rectangle_size[1]), \
                    (coord[0] - rectangle_size[0], coord[1] - rectangle_size[1])]
            draw.rectangle(xy, fill = colour)

    run = image.split('/')[-3]
    camera = image.split('/')[-2]
    frame = image.split('/')[-1]

    # derive the root data directory from the image filename that is passed
    root_data_dir = ('/').join(image.split('/')[:-3]) + '/'
    sanity_dir = root_data_dir + run + '/' + camera + '_sanity/'
    if not os.path.exists(sanity_dir):
        os.makedirs(sanity_dir)
    else:
        pass

    im.save(sanity_dir + frame)
    print "wrote ", run, camera, "sanity ", frame,"\r",
    sys.stdout.flush()
