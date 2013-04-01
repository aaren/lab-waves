"""Collection of pure functions that operate on individual image objects

Functions:

- barrel_correct: correct barrel distortion

"""
from __future__ import division

# necessary IO modules for using external functions
import os
import subprocess

import numpy as np
import Image
import ImageDraw
import ImageFont

import config


def barrel_correct(im, coeffs, verbose=False, tmp_in='/tmp/bctmpin', tmp_out='/tmp/bctmpout', tmp_fmt='bmp'):
    """Uses Imagemagick convert to apply a barrel correction to
    an image.

    Inputs: im - a PIL Image object
            coeffs - a list of four coefficients [a, b, c, d] which
                     can be str, float or int, will be converted to
                     str anyway.
            verbose - print output to screen?
            tmp_in - input temp file
            tmp_out - output temp file
            tmp_fmt - temp file format, default is raw bitmap

    Outputs: A corrected PIL Image object.

    Barrel distortion is radial. For a given output pixel a
    distance r_dest from the centre of the image, the corresponding
    source pixel is found at a radius r_src using a set of four
    coefficients:

    a, b, c, d = coeffs
    r_src = r_dest * (a * r_dest ** 3 + b * r_dest ** 2 + c * r_dest + d)

    This function is a bit annoying as it calls an external function
    to do the work, which means doing IO in a supposedly pure
    function.

    The solution is to create a temporary file and re-read it
    to an image object. This won't be as fast as it could be,
    but consistency of these low level functions is more important
    than performance as we can solve the latter by using a bigger
    computer.
    """
    # check image is RGB
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    # create temp files
    tin = tmp_in + '.' + tmp_fmt
    tout = tmp_out + '.' + tmp_fmt
    im.save(tin, tmp_fmt)

    # format coefficients for convert
    scoeffs = ' '.join([str(c) for c in coeffs])

    cmd = ["convert",
           "-verbose",
           "-distort", "Barrel",
           scoeffs,
           tin,
           tout]
    if verbose:
        subprocess.call(cmd)
    elif not verbose:
        subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_im = Image.open(tout)

    # cleanup
    os.remove(tin)
    os.remove(tout)

    return out_im


def perspective_transform(im, coeffs):
    """Apply a perspective transform to an image.

    Inputs: im - a PIL image object

            coeffs - a sequence of perspective coefficients
                     (a, b, c, d, e, f, g, h) that correspond
                     to the transform matrix

    Output: a transformed image object

    The coefficients can be generated by another method,
    perspective_coefficients.
    """
    args = {'size':     im.size,
            'method':   Image.PERSPECTIVE,
            'data':     coeffs,
            'resample': Image.BILINEAR}
    trans = im.transform(*args)
    return trans


def perspective_coefficients(X, x):
    """Calculates the perspective coefficients that would affect
    a four point perspective transform on a given list of
    four input points, x, and output X.

    Inputs:
        X - (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)
            output points
        x - (x1, y1), (x2, y2), (x3, y3), (x4, y4)
            input points

    Returns:
        Vector of perspective transform coefficients,
        (a, b, c, d, e, f, g, h)

    N.B. This calculates the coefficients needed to obtain the
    corresponding INPUT to a given OUTPUT, which is what we need
    for PIL transform.

    Solves the equation Ac = X.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = x
    (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = X

    A = np.matrix([[x1, y1, 1,  0,  0, 0, -X1 * x1, -X1 * y1],
                   [0,   0, 0, x1, y1, 1, -Y1 * x1, -Y1 * y1],
                   [x2, y2, 1,  0,  0, 0, -X2 * x2, -X2 * y2],
                   [0,   0, 0, x2, y2, 1, -Y2 * x2, -Y2 * y2],
                   [x3, y3, 1,  0,  0, 0, -X3 * x3, -X3 * y3],
                   [0,   0, 0, x3, y3, 1, -Y3 * x3, -Y3 * y3],
                   [x4, y4, 1,  0,  0, 0, -X4 * x4, -X4 * y4],
                   [0,   0, 0, x4, y4, 1, -Y4 * x4, -Y4 * y4]])

    c = np.solve(A, np.asarray(X).flatten())

    return c


def run_perspective_coefficients(run):
    """Generate the cam1 and cam2 perspective transform coefficients
    for a given run.

    Inputs: run - string, the run index

    Outputs: dictionary of the camera coefficients
             d.keys() = ['cam1', 'cam2']
             d['cam1'] = (a, b, c, d, e, f, g, h)
    """
    run_data = get_run_data(run)

    lock_0 = int(run_data['l0x']), int(run_data['l0y'])
    lock_surf = int(run_data['lsx']), int(run_data['lsy'])
    join1_0 = int(run_data['j10x']), int(run_data['j10y'])
    join1_surf = int(run_data['j1sx']), int(run_data['j1sy'])

    join2_0 = int(run_data['j20x']), int(run_data['j20y'])
    join2_surf = int(run_data['j2sx']), int(run_data['j2sy'])
    ruler_0 = int(run_data['r0x']), int(run_data['r0y'])
    ruler_surf = int(run_data['rsx']), int(run_data['rsy'])
    # need some standard vertical lines in both cameras.
    # cam1: use lock gate and tank join
    # cam2: tank join and ruler at 2.5m
    # (checked to be vertical, extrapolate to surface)
    # so for each camera, 4 locations (8 numbers) need
    # to be recorded.

    x1 = (lock_0, lock_surf, join1_0, join1_surf)
    X1 = (lock_0,
          (lock_0[0], lock_0[1] - config.ideal_25),
          (lock_0[0] - config.ideal_base_1, lock_0[1]),
          (lock_0[0] - config.ideal_base_1, lock_0[1] - config.ideal_25))

    x2 = (join2_0, join2_surf, ruler_0, ruler_surf)
    X2 = (join2_0,
          (join2_0[0], join2_0[1] - config.ideal_25),
          (join2_0[0] - config.ideal_base_2, join2_0[1]),
          (join2_0[0] - config.ideal_base_2, join2_0[1] - config.ideal_25))

    cam1_coeff = tuple(perspective_coefficients(x1, X1))
    if run_data['j20x'] == '0':
        cam2_coeff = 0
    else:
        cam2_coeff = tuple(perspective_coefficients(x2, X2))

    out = {'cam1': cam1_coeff, 'cam2': cam2_coeff}
    return out


def apply_perspective_transform(run):
    # TODO: flesh out
    coeffs = run_perspective_coefficients(run)
    for camera, image in run:
        im = Image.open(image)
        trans = perspective_transform(im, coeffs[camera])
        trans.save('blah')


## State needed for add_text ##
run, cam, frame = image.split('/')[-3:]
# get the orientation bits for a run, prompting for measurement
# if needs be.
# TODO: put in specific IO module
run_data = get_run_data(run)

# 'time' that a frame represents
# TODO: update this with 25hz camera in mind. need some time
# generating function
time = int(frame.split('.')[0]) - 1
# TODO: put this in config
config.font = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf',
# TODO: put in specific IO module
parameters = get_parameters(run, config.paramf)


def add_text(im, run_data, cam, time, fonts=config.font, parameters=parameters):
    """Put black backgrounded bars into im and print text
    over them using given font

    fonts:
     this is PLATFORM DEPENDENT
     in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide.
     "University of Leeds" is 520 px wide.

    Then crop the image to specified size and offsets.
    """
    # TODO: move calls to config to a container function,
    # can't have these inside a pure function - need to be
    # supplied as arguments

    odd = {'cam1': run_data['odd_1'], 'cam2': run_data['odd_2']}
    if odd[cam] == '999':
        return

    # define the box to crop the image to relative to the
    # invariant point in the projection transform (se).
    l0x = int(run_data['l0x'])
    l0y = int(run_data['l0y'])
    j20x = int(run_data['j20x'])
    j20y = int(run_data['j20y'])
    ref = {'cam1': (l0x, l0y), 'cam2': (j20x, j20y)}
    left = ref[cam][0] + config.crop[cam][0]
    right = ref[cam][0] + config.crop[cam][1]
    upper = ref[cam][1] - config.ideal_25 + config.crop[cam][2]
    lower = ref[cam][1] + config.crop[cam][3]

    font = ImageFont.truetype(fonts, 40)
    author_text = "Aaron O'Leary, University of Leeds"
    param = ("run {run}, "
             "t={t}s: "
             "h_1 = {h_1}, "
             "rho_0 = {r0}, "
             "rho_1 = {r1}, "
             "rho_2 = {r2}, "
             "alpha = {a}, "
             "D = {D}")
    params = dict(run=parameters['run_index'],
                  t=time,
                  h_1=parameters['h_1/H'],
                  r0=parameters['rho_0'],
                  r1=parameters['rho_1'],
                  r2=parameters['rho_2'],
                  a=parameters['alpha'],
                  D=parameters['D/H'])
    param_text = param.format(**params)
    upper_text, lower_text = param_text, author_text

    # position of the text
    upper_text_pos = (left, upper)
    lower_text_pos = (left, lower - config.bottom_bar)
    # position of the background
    upper_background_pos = (left, upper, right, upper + config.top_bar)
    lower_background_pos = (left, lower - config.bottom_bar, right, lower)
    # text colour
    text_colour = 'white'
    # background colour
    background_colour = 'black'

    draw = ImageDraw.Draw(im)
    draw.rectangle(upper_background_pos, fill=background_colour)
    draw.rectangle(lower_background_pos, fill=background_colour)
    draw.text(upper_text_pos, upper_text, font=font, fill=text_colour)
    draw.text(lower_text_pos, lower_text, font=font, fill=text_colour)

    #box is a 4-tuple defining the left, upper, right and lower pixel
    box = (left, upper, right, lower)
    cropped = im.crop(box)

    return cropped
