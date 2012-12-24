## Script to standardise lab images
## AOL Nov 2011

from __future__ import division
import os
import glob
import sys

import Image
import ImageFont
import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage.interpolation import geometric_transform

from aolcore import pull_col, pull_line
from aolcore import get_parameters
from config import path, paramf, procf, crop
from config import ideal_25, ideal_base_1, ideal_base_2
from config import top_bar, bottom_bar

# We don't just want to operate on a single image, we want to operate on
# many. But in the first instance, in determining the offsets for cropping
# and checking the scale for barrel correction and determining the rotation
# correction angle, we need only look at the first image in a given run.
# This is what bc1(run) does; measure(run) will prompt for measurements
# from the images bc1 creates.

def measure(run):
    proc = []
    proc.append(run)
    for camera in ['cam1', 'cam2']:
        plt.figure(figsize=(16,12))
        img1 = '%s/barrel_corr/%s/%s/img_0001.jpg' % (path, run, camera)
        try:
            im = Image.open(img1)
        except IOError:
            print "Bad image for %s %s" % (run, camera)
            break
        plt.imshow(im)
        plt.xlim(2500,3000)
        plt.ylim(750, 1500)
        plt.draw()
        print "Select lock base and surface"
        pt1 = plt.ginput(3, 0)
        if camera == 'cam1':
            plt.xlim(0,500)
        elif camera == 'cam2':
            plt.xlim(750, 1250)
        plt.draw()
        print "Select join base and surface"
        pt2 = plt.ginput(3, 0)

        pts = pt1[0:2] + pt2[0:2]
        for x,y in pts:
            proc.append(int(x))
            proc.append(im.size[1] - int(y))

        plt.xlim(0,3000)
        plt.draw()
        if camera == 'cam1':
            print "What is the extent of lock leakage?"
            leak = plt.ginput(2,0)[0][0]
            proc.append(int(pt1[0][0] - leak))
        print "Weird (y/n)"
        proc.append(raw_input('> '))
        plt.close()

    proc = [str(e) for e in proc]
    entry = ','.join(proc) + '\n'
    f = open(procf, 'a')
    f.write(entry)
    f.close()

def get_run_data(run):
    proc_runs = pull_col(0, procf, ',')
    try:
        line_number = proc_runs.index(run)
        # print "%s is in proc_data" % run
    except ValueError:
        print "%s is not in the procf (%s)" % (run, procf)
        print "get the proc_data for this run now? (y/n)"
        A = raw_input('> ')
        if A == 'y':
            measure(run)
            get_run_data(run)
        elif A == 'n':
            return 0
        else:
            print "y or n!"
            get_run_data(run)

    run_data = get_parameters(run, procf, ',')

    return run_data

def bc1(run):
    """Just barrel correct the first image of a run"""
    indir = '%s/synced/%s' % (path, run)
    outdir = 'bc1'
    for camera in ['cam1', 'cam2']:
        dirs = '/'.join([path, outdir, run, camera])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print "made " + dirs
        else:
            # print "using " + dirs
            pass
        image1 = '%s/%s/img_0001.jpg' % (indir, camera)
        barrel_corr(image1, outdir)

def barrel_corr(image, outimage, Null=None):
    # run = image.split('/')[-3]
    # cam = image.split('/')[-2]
    # frame = image.split('/')[-1]
    cam = 'cam1'

    cam1corr = (0.000658776, -0.0150048, -0.00123339, 1.01557914)
    cam2corr = (0.023969, -0.060001, 0, 1.036032)

    if cam == 'cam1':
        corr = cam1corr
    elif cam == 'cam2':
        corr = cam2corr
    else:
        print "Camera must be cam1 or cam2"

    im = ndimage.imread(image)
    dim = im.shape[0:2]
    imcorr = geometric_transform(im, f, extra_arguments=(dim, corr))
    return imcorr
    # write out

    # print "Correcting", run, cam, frame, "\r",
    sys.stdout.flush()

def r_from_centre((x, y), (w, h)):
    """Calculate the radius from the centre of an image of a point (x, y)
    in the image.
    """
    r = ((w / 2 - x) ** 2 + (h / 2 - y) ** 2) ** .5
    return r

def barrel_dist_r((x, y), (w, h), (a, b, c, d)):
    """For given point x, y in the desination, calculate the radius
    to the equivalent point in the source.

    The radius is normalised such that the described circle will fit
    entirely into the image.

    i.e. half of the smallest of h, w is used to normalise.
    """
    r_dest = r_from_centre((x, y), (w, h))
    norm = min(w,h) / 2
    norm_r_dest = r_dest / norm
    norm_r_src = barrel_dist(norm_r_dest, (a, b, c, d))
    r_src = norm_r_src * norm
    return r_src

def barrel_dist(r, (a, b, c, d)):
    """For given radius in the destination, calculate the equivalent
    radius in the source for given coefficients.

    r has to be in units normalised to half the minimum side length.

    That is, the circle that r describes must fit entirely into the image.
    """
    r_src = r * (a * r ** 3 + b * r ** 2 + c * r + d)
    return r_src

def f((x,y,z), (w, h), (a,b,c,d)):
    """Map destination pixels to source pixels for barrel
    distortion coefficients a, b, c, d and an image of
    dimensions (w, h).

    There are two equations to solve.

    Given an (x, y) input we can calculate the radius of this point
    in both the source and desination images.

    Secondly, we conserve the direction of the vector from the centre
    of the image to the point in both images.

    i.e. (w / 2 - x_src) / (h / 2 - y_src) = (w / 2 - x_dest) / (h / 2 - y_dest)

    This gives us a quadratic in y_src.
    """
    if x == w / 2 and y == h / 2:
        return x, y, z
    elif y == h / 2:
        x_src = barrel_dist_r((x, y), (w, h), (a, b, c, d))
        y_src = y
        return x_src, y_src, z

    tan = (w / 2 - x) / (h / 2 - y)

    r = barrel_dist_r((x, y), (w, h), (a, b, c, d))
    # coefficients A x ** 2 + B * x + C = 0
    A = 1 + tan ** 2
    B = (h * (1 + tan ** 2) - 2 * w * tan)
    C = w ** 2 / 2 + (h ** 2 / 4) * (1 + tan) + (w * h / 2) * tan - r ** 2

    roots = np.roots((A, B, C))
    # this is the correct root selection. It comes out like this
    # because of the simple formula used to equalise the angles.
    if x < w / 2 and y < h / 2:
        y_src = roots[1]
    elif x < w / 2 and y > h / 2:
        y_src = roots[0]
    elif x >= w / 2 and y < h / 2:
        y_src = roots[0]
    elif x >= w / 2 and y >= h / 2:
        y_src = roots[0]
    else:
        exit("I don't know what you're talking about")

    x_src = tan * (y_src - h / 2) + w / 2

    return x_src, y_src, z

def barrel_corrections(run, run_data=None):
    # Barrel correct
    bc_out = 'barrel_corr'
    proc_images(barrel_corr, run, path + '/synced', 'barrel_corr', None, None)

def perspective_coefficients(X,x):
    """Calculates the perspective coeffcients for a given list of
    four input points, x, and output X.

    N.B. This calculates the coefficients needed to obtain the
    corresponding INPUT to a given OUTPUT, which is what we need
    for PIL transform.

    Solves the equation Ac = X.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = x
    (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = X

    A = np.matrix([[x1,y1,1,0,0,0,-X1*x1,-X1*y1],\
                   [0,0,0,x1,y1,1,-Y1*x1,-Y1*y1],\
                   [x2,y2,1,0,0,0,-X2*x2,-X2*y2],\
                   [0,0,0,x2,y2,1,-Y2*x2,-Y2*y2],\
                   [x3,y3,1,0,0,0,-X3*x3,-X3*y3],\
                   [0,0,0,x3,y3,1,-Y3*x3,-Y3*y3],\
                   [x4,y4,1,0,0,0,-X4*x4,-X4*y4],\
                   [0,0,0,x4,y4,1,-Y4*x4,-Y4*y4]])

    c = np.linalg.solve(A, np.asarray(X).flatten())

    return c

def transform(image, outimage, coeffs):
    im = Image.open(image)
    run, cam, frame = image.split('/')[-3:]
    print "Perspective transform", run, cam, frame, "\r",
    sys.stdout.flush()
    trans = im.transform(im.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    trans.save(outimage)

def std_corrections(run, run_data=None):
    if run_data is None:
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

    x1 = lock_0, lock_surf, join1_0, join1_surf
    X1 = lock_0, \
         (lock_0[0], lock_0[1] - ideal_25), \
         (lock_0[0] - ideal_base_1, lock_0[1]), \
         (lock_0[0] - ideal_base_1, lock_0[1] - ideal_25)

    x2 = join2_0, join2_surf, ruler_0, ruler_surf
    X2 = join2_0, \
         (join2_0[0], join2_0[1] - ideal_25), \
         (join2_0[0] - ideal_base_2, join2_0[1]), \
         (join2_0[0] - ideal_base_2, join2_0[1] - ideal_25)

    # print x1
    # print X1
    # print x2
    # print X2
    cam1_coeff = tuple(perspective_coefficients(x1,X1))
    if run_data['j20x'] == '0':
        cam2_coeff = 0
    else:
        cam2_coeff = tuple(perspective_coefficients(x2,X2))

    # transform('img_0001.jpg', cam1_coeff)
    proc_images(transform, run, path + '/barrel_corr', 'std_corr', \
                                                cam1_coeff, cam2_coeff)

def add_text(image, outimage, Null=None):
    # opens and crops an image to the box given.
    im = Image.open(image)

    run, cam, frame = image.split('/')[-3:]
    run_data = get_run_data(run)
    frame = frame.split('_')[-1]
    time = int(frame.split('.')[0]) - 1
    print "Crop / text", run, frame, "\r",
    sys.stdout.flush()

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
    left = ref[cam][0] + crop[cam][0]
    right = ref[cam][0] + crop[cam][1]
    upper = ref[cam][1] - ideal_25 + crop[cam][2]
    lower = ref[cam][1] + crop[cam][3]

    draw = ImageDraw.Draw(im)
    draw.rectangle((left, upper, right, upper + top_bar), fill='black')
    draw.rectangle((left, lower - bottom_bar, right, lower), fill='black')

    # this is PLATFORM DEPENDENT
    # in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide.
    # "University of Leeds" is 520 px wide.
    fonts = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(fonts, 40)

    p = get_parameters(run, paramf)
    author_text = "Aaron O'Leary, University of Leeds"
    param_a = "run %s, t=%ss: h_1 = %s, rho_0 = %s, rho_1 = %s, rho_2 = %s, "
    param_b = "alpha = %s, D = %s"
    param_t = param_a + param_b
    param_text = param_t % (p['run_index'], time, p['h_1/H'], p['rho_0'],\
            p['rho_1'], p['rho_2'], p['alpha'], p['D/H'])

    text_hi_pos = (left, upper)
    text_low_pos = (left, lower - bottom_bar)
    text_hi, text_low = param_text, author_text

    draw.text(text_hi_pos, text_hi, font=font, fill="white")
    draw.text(text_low_pos, text_low, font=font, fill="white")

    #box is a 4-tuple defining the left, upper, right and lower pixel
    box = (left, upper, right, lower)
    cropped = im.crop(box)
    cropped.save(outimage)

def text_crop(run, run_data=None):
    # Add text and crop
    proc_images(add_text, run, path + '/std_corr', 'processed', None, None)

def proc_images(proc, run, source, output, arg1, arg2):
    def cam_proc(arg):
        path = image.split('/')
        run = path[-3]
        cam = path[-2]
        frame = path[-1]

        path[-4] = output
        outimage = '/'.join(path)
        outdirpath = '/'.join(path[:-1])
        if not os.path.exists(outdirpath):
            os.makedirs(outdirpath)
        #print "performing",proc,"on",image,"\r",
        #sys.stdout.flush()
        proc(image, outimage, arg)

    for image in sorted(glob.glob('%s/%s/cam1/*jpg' % (source, run))):
        cam_proc(arg1)
    for image in sorted(glob.glob('%s/%s/cam2/*jpg' % (source, run))):
        cam_proc(arg2)
    # print "\n"
