## Script to standardise lab images
## AOL Nov 2011

from __future__ import division
import os
import glob

import Image
import ImageFont, ImageDraw

from aolcore import pull_col, pull_line
from aolcore import get_parameters
from config import path, paramf, procf

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

    infile = '%s/synced/%s/%s/%s' % (path, run, cam, frame)
    outfile = '%s/%s/%s/%s/%s' % (path, outdir, run, cam, frame)
    command = 'convert -distort Barrel %s %s %s' % (corr, infile, outfile)

    print "Barrel correcting %s, %s, %s with 18mm coefficients,\n%s"\
                                        % (run, cam, frame, corr)
    os.system(command)

def bc1(run):
    """Just barrel correct the first image of a run"""
    indir = '%s/synced/%s' % (path, run)
    outdir = 'bc1'
    for camera in glob.glob(indir + '/cam*'):
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
    f = open('procf', 'a')
    f.write(entry)
    f.close()

def bcm():
    # easy testing
    measure('r11_7_06g')

def get_run_data(run):
    proc_runs = pull_col(0, procf, ',') 
    try:
        line_number = proc_runs.index(run)
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
    re = im.resize((w_new, h_new))
    re.save(image)

def add_text(image, (scale, data)):
    # opens and crops an image to the box given.     
    im = Image.open(image)

    ratio, run_data = scale, data

    author_text = "Aaron O'Leary, University of Leeds"
    param_a = "run %s, t=%ss: h_1 = %s, rho_0 = %s, rho_1 = %s, rho_2 = %s, "
    param_b = "alpha = %s, D = %s" 
    param_t = param_a + param_b

    run = image.split('/')[-3]
    param_runs = pull_col(0, paramf)
    line_number = param_runs.index(run) 
    parameter = pull_line(line_number, paramf)

    camera = image.split('/')[-2]
    time = int(image.split('_')[-1].split('.')[0]) - 1
    param_text = param_t % (parameter[0], time, parameter[1], parameter[2],\
            parameter[3], parameter[4], parameter[5], parameter[6])
    scale_depth = 440

    if camera == 'cam1':
        cam1_ratio = ratio
        offset = int(run_data['off_1'])
        bottom1 = int(run_data['bottom_1'])

        left = int((offset * cam1_ratio) - 60)
        right = int((offset * cam1_ratio) + 2750)
        upper = int((bottom1 * cam1_ratio) - (scale_depth + 100))
        lower = int((bottom1 * cam1_ratio) + 150)

    elif camera == 'cam2':
        cam2_ratio = ratio
        offset = int(run_data['off_2'])
        bottom1 = int(run_data['bottom_2'])

        left = int((offset * cam2_ratio) - 2750)
        right = int((offset * cam2_ratio) + 60)
        upper = int((bottom1 * cam2_ratio) - (scale_depth + 100))
        lower = int((bottom1 * cam2_ratio) + 150)

    draw = ImageDraw.Draw(im)
    draw.rectangle((left, upper, right, upper + 100), fill = 'black')
    draw.rectangle((left, lower - 100, right, lower), fill = 'black')

    # this is PLATFORM DEPENDENT
    # in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide.
    # "University of Leeds" is 520 px wide.
    fonts = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(fonts, 45)

    text_hi_pos = (left, upper)
    text_low_pos = (left, lower - 100)
    text_hi, text_low = param_text, author_text

    draw.text(text_hi_pos, text_hi, font=font, fill="white")
    draw.text(text_low_pos, text_low, font=font, fill="white")

    #box is a 4-tuple defining the left, upper, right and lower pixel
    box = (left, upper, right, lower)
    cropped = im.crop(box)
    cropped.save(image)

def proc_images(proc, run, source, arg1, arg2):
    for image in glob.glob('%s/%s/cam1/*jpg' % (source, run)):
        print "performing %s on %s" % (proc, image)
        proc(image, arg1)
    for image in glob.glob('%s/%s/cam2/*jpg' % (source, run)):
        print "performing %s on %s" % (proc, image)
        proc(image, arg2)

def std_corrections(run, run_data=None):
    if run_data is None:
        run_data = get_run_data(run)
    # Barrel correct
    bc_out = 'processed'
    proc_images(barrel_corr, run, path + '/synced', bc_out, bc_out)

    # Rotation correct 
    theta1 = -float(run_data['rot_1'])
    theta2 = -float(run_data['rot_2'])
    proc_images(rotation_corr, run, path + '/' + bc_out, theta1, theta2)
 
def text_crop(run, run_data=None):
    if run_data is None:
        run_data = get_run_data(run)
    # Resize the images to standard
    ideal_ruler = 435
    ruler = int(run_data['bottom_1']) - int(run_data['ruler_25'])
    ruler_ratio = ideal_ruler / ruler
    depth_1 = int(run_data['bottom_1']) - int(run_data['surface_1'])
    depth_2 = int(run_data['bottom_2']) - int(run_data['surface_2'])
    depth_ratio = depth_1 / depth_2
    cam1_ratio = ruler_ratio
    cam2_ratio = ruler_ratio * depth_ratio

    # Rescale to common size
    # rescaling means that the offsets, lock_pos etc. are rescaled too.
    proc_images(rescale, run, path + '/processed', cam1_ratio, cam2_ratio)

    # Add text and crop
    proc_images(add_text, run, path + '/processed', \
            (cam1_ratio, run_data), (cam2_ratio, run_data))