## Script to standardise lab images
## AOL Nov 2011

from __future__ import division
import os
import glob
import sys

import Image
import ImageFont
import ImageDraw
from numpy import matrix
from numpy import asarray
from numpy.linalg import solve

from aolcore import pull_col, pull_line
from aolcore import get_parameters
from config import path, paramf, procf, crop

# We don't just want to operate on a single image, we want to operate on
# many. But in the first instance, in determining the offsets for cropping
# and checking the scale for barrel correction and determining the rotation
# correction angle, we need only look at the first image in a given run.
# This is what bc1(run) does; measure(run) will prompt for measurements
# from the images bc1 creates.

def load_img(image):
    im = Image.open(image)
    im.load()

    source = im.split()

    red = source[0].load()
    green = source[1].load()
    blue = source[2].load()

    w, h = im.size

    return red, green, blue, w, h

def barrel_corr(image, outdir):
    run = image.split('/')[-3]
    cam = image.split('/')[-2]
    frame = image.split('/')[-1]

    cam1corr = '"0.000658776 -0.0150048 -0.00123339 1.01557914"'
    cam2corr = '"0.023969 -0.060001 0 1.036032"'

    if cam == 'cam1':
        corr = cam1corr
    elif cam == 'cam2':
        corr = cam2corr
    else:
        print "Camera must be cam1 or cam2"

    outdirpath = '%s/%s/%s/%s/' % (path, outdir, run, cam)
    if not os.path.exists(outdirpath):
        os.makedirs(outdirpath)

    infile = '%s/synced/%s/%s/%s' % (path, run, cam, frame)
    outfile = outdirpath + frame
    command = 'convert -distort Barrel %s %s %s' % (corr, infile, outfile)

    #print cam,"has barrel correction coefficients"
    #print corr
    print "Correcting", run, cam, frame, "\r",
    sys.stdout.flush()
    os.system(command)

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
            print "using " + dirs
        image1 = '%s/%s/img_0001.jpg' % (indir, camera)
        barrel_corr(image1, outdir)

def measure(run):
    proc = []
    proc.append(run)
    for camera in ['cam1', 'cam2']:
        command = 'gimp -s -f -d %s/bc1/%s/%s/img_0001.jpg &'\
                                         % (path, run, camera)
        os.system(command)

        print "What is the rotation correction for %s? (as measured)" % camera
        proc.append(raw_input('> '))

        print "What is the offset for %s?" % camera
        proc.append(raw_input('> '))

        print "What is the position of the water surface for %s?" % camera
        proc.append(raw_input('> '))

        print "What is the position of the tank floor for %s?" % camera
        proc.append(raw_input('> '))

        if camera == 'cam1':
            print "Where is the 25 in %s" % camera
            proc.append(raw_input('> '))
            print "What is the lock position?"
            proc.append(raw_input('> '))
            print "What is the extent of lock leakage?"
            proc.append(raw_input('> '))

        print "Weird (y/n)"
        proc.append(raw_input('> '))

    entry = ','.join(proc) + '\n'
    f = open(procf, 'a')
    f.write(entry)
    f.close()

def bcm():
    # easy testing
    measure('r11_7_06g')

def get_run_data(run):
    proc_runs = pull_col(0, procf, ',')
    try:
        line_number = proc_runs.index(run)
        print "%s is in proc_data" % run
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

def rotation_corr(image, angle):
    # rotates the image by a given angle and saves, overwriting the original
    im = Image.open(image)
    rot = im.rotate(angle)
    rot.save(image)

def rescale(image, ratio):
    im = Image.open(image)
    w, h = im.size
    h_new = int(h * ratio)
    w_new = int(w * ratio)
    frame = image.split('_')[-1]
    print "\rrescaling", frame,
    sys.stdout.flush()
    re = im.resize((w_new, h_new))
    re.save(image)

def add_text(image, (scale, data)):
    # opens and crops an image to the box given.
    im = Image.open(image)
    # have to change the out image
    dirs = image.split('/')[:]
    dirs[-4] = 'processed'
    outimage = '/'.join(dirs)
    outdir = '/'.join(dirs[:-1])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ratio, run_data = scale, data
    run = image.split('/')[-3]
    cam = image.split('/')[-2]
    frame = image.split('_')[-1]
    time = int(frame.split('.')[0]) - 1
    print "Crop / text", run, frame, "\r",
    sys.stdout.flush()

    odd = {'cam1': run_data['odd_1'], 'cam2': run_data['odd_2']}
    if odd[cam] == '999':
        return

    # define the box to crop the image to.
    l0x = int(get_data['l0x'])
    l0y = int(get_data['l0y'])
    # FIXME: work for both cameras??
    left = l0x + crop[cam][0]
    right = l0x + crop[cam][1]
    upper = l0y - ideal_25 + crop[cam][2]
    lower = l0y + crop[cam][3]

    draw = ImageDraw.Draw(im)
    draw.rectangle((left, upper, right, upper + 100), fill='black')
    draw.rectangle((left, lower - 100, right, lower), fill='black')

    # this is PLATFORM DEPENDENT
    # in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide.
    # "University of Leeds" is 520 px wide.
    fonts = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(fonts, 45)

    p = get_parameters(run, paramf)
    author_text = "Aaron O'Leary, University of Leeds"
    param_a = "run %s, t=%ss: h_1 = %s, rho_0 = %s, rho_1 = %s, rho_2 = %s, "
    param_b = "alpha = %s, D = %s"
    param_t = param_a + param_b
    param_text = param_t % (p['run_index'], time, p['h_1/H'], p['rho_0'],\
            p['rho_1'], p['rho_2'], p['alpha'], p['D/H'])

    text_hi_pos = (left, upper)
    text_low_pos = (left, lower - 100)
    text_hi, text_low = param_text, author_text

    draw.text(text_hi_pos, text_hi, font=font, fill="white")
    draw.text(text_low_pos, text_low, font=font, fill="white")

    #box is a 4-tuple defining the left, upper, right and lower pixel
    box = (left, upper, right, lower)
    cropped = im.crop(box)
    cropped.save(outimage)

def proc_images(proc, run, source, arg1, arg2):
    #print '%s, %s, cam1, %s' % (proc, run, source)
    for image in sorted(glob.glob('%s/%s/cam1/*jpg' % (source, run))):
        #print "performing",proc,"on",image,"\r",
        #sys.stdout.flush()
        proc(image, arg1)
    #print '%s, %s, cam2, %s' % (proc, run, source)
    for image in sorted(glob.glob('%s/%s/cam2/*jpg' % (source, run))):
        #print "performing",proc,"on",image,"\r",
        #sys.stdout.flush()
        proc(image, arg2)

def std_corrections(run, run_data=None):
    if run_data is None:
        run_data = get_run_data(run)
    # Barrel correct
    bc_out = 'std_corr'
    # proc_images(barrel_corr, run, path + '/synced', bc_out, bc_out)

    ideal_25 = 435

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
    ideal_base_1 = int(5.88 * ideal_25)
    ideal_base_2 = int(7.96 * ideal_25)

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

    print x1
    print X1
    cam1_coeff = tuple(perspective_coefficients(x1,X1))
    cam2_coeff = tuple(perspective_coefficients(x2,X2))

    transform('img_0001.jpg', cam1_coeff)
    # proc_images(transform, run, path + '/' + bc_out, cam1_coeff, cam2_coeff)

def text_crop(run, run_data=None):
    if run_data is None:
        run_data = get_run_data(run)
    # Resize the images to standard
    ideal_ruler = 435
    ruler = int(run_data['bottom_1']) - int(run_data['ruler_25'])
    ruler_ratio = ideal_ruler / ruler
    depth_1 = int(run_data['bottom_1']) - int(run_data['surface_1'])
    depth_2 = int(run_data['bottom_2']) - int(run_data['surface_2'])
    cam1_ratio = ruler_ratio
    if depth_2 != 0:
        depth_ratio = depth_1 / depth_2
        cam2_ratio = ruler_ratio * depth_ratio
    elif depth_2 == 0:
        cam2_ratio = 1
    # Add text and crop
    proc_images(add_text, run, path + '/std_corr', \
            (cam1_ratio, run_data), (cam2_ratio, run_data))

def perspective_coefficients(x,X):
    """Calculates the perspective coeffcients for a given list of
    four input points, x, and output X.

    Solves the equation Ac = X.
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = x
    (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = X

    A = matrix([[x1,y1,1,0,0,0,-X1*x1,-X1*y1],\
                [0,0,0,x1,y1,1,-Y1*x1,-Y1*y1],\
                [x2,y2,1,0,0,0,-X2*x2,-X2*y2],\
                [0,0,0,x2,y2,1,-Y2*x2,-Y2*y2],\
                [x3,y3,1,0,0,0,-X3*x3,-X3*y3],\
                [0,0,0,x3,y3,1,-Y3*x3,-Y3*y3],\
                [x4,y4,1,0,0,0,-X4*x4,-X4*y4],\
                [0,0,0,x4,y4,1,-Y4*x4,-Y4*y4]])

    c = solve(A, asarray(X).flatten())
    print c

    return c

def transform(image, coeffs):
    im = Image.open(image)
    print coeffs
    trans = im.transform(im.size, 2, coeffs)
    trans.save('out.jpg')
