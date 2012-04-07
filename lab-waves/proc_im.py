## Script to process lab images
## AOL Nov 2011

from __future__ import division
import glob
import Image
import ImageFont, ImageDraw
from sys import argv
import os
import math

from aolcore import pull_col, pull_line


# script, run = argv
from get_data import path

paramf = path + '/parameters'
procf = path + '/proc_data'

# Initialisation:
#
# list the files
print "Initialising..."

# We don't just want to operate on a single image, we want to operate on
# many. But in the first instance, in determining the offsets for cropping
# and checking the scale for barrel correction and determining the rotation
# correction angle, we need only look at the first image in a given run.
#
# The list of runs is in the first column of the parameters file and these are
# the same as the directory names that the images are stored in (check!!!).
# 
# I want to add the scale, offets and rotation angle to the parameters file.
# Rethinking this, perhaps I should just create a new file which these data
# are added to. They aren't scientific variables, just means to making an
# easily useable product that I would like to note down somewhere. The new
# file can always be tacked onto parameters if it's really necessary.

def load_img(image):
    im = Image.open(image)
    im.load()

    source = im.split()

    red = source[0].load()
    green = source[1].load()
    blue = source[2].load()

    w, h = im.size
    
    return red, green, blue, w, h

def barrel_corr(image, null1):
    # reduce the path down to just the filename
    folder = image.split('/')[-3]
    camera = image.split('/')[-2]
    image = image.split('/')[-1]

    cam1corr = '"0.000658776 -0.0150048 -0.00123339 1.01557914"'
    cam2corr = '"0.023969 -0.060001 0 1.036032"'
    
    if camera == 'cam1':
        corr = cam1corr
        cam = 'cam1'
    elif camera == 'cam2':
        corr = cam2corr
        cam = 'cam2'
    else:
        print "Camera must be cam1 or cam2"

    infile = '%s/synced/%s/%s/%s' % (path, folder, cam, image)
    outfile = '%s/processed/%s/%s/%s' % (path, folder, cam, image)
    command = 'convert -distort Barrel %s %s %s' % (corr, infile, outfile)

    print "Barrel correcting %s with %s 18mm coefficients,\n%s" % (image, cam, corr)
    os.system(command)


def barrel_corr_measure(run):
    proc = []
    proc.append(run)
    for camera in ['cam1', 'cam2']:
        command = 'gimp -s -f -d %s/processed/%s/%s/img_0001.jpg &'\
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

# easy testing
def bcm():
    barrel_corr_measure('11_7_06g')

def bc1(run):
    for camera in ['cam1', 'cam2']:
        barrel_corr('%s/synced/%s/%s/img_0001.jpg' % (path, run, camera), None)

def get_run_data(run):
    lines = pull_col(0, procf, ',') 
    line_number = lines.index(run)
    line = pull_line(line_number, procf, ',')

    headers = pull_line(0, procf, ',')
    
    run_data = {}

    for header in headers:
        index = headers.index(header)
        run_data[header] = line[index]

    return run_data

def rotation_corr(image, angle):
    # rotates the image by a given angle and saves, overwriting the original
    im = Image.open(image)
    rot = im.rotate(angle)
    rot.save(image)

def get_measurements(run):
    proc1 = barrel_corr_measure('cam1', run)
    proc2 = barrel_corr_measure('cam2', run)

    proc_string = '\t'.join(proc1 + proc2)
    print proc_string

    f = open(procf, 'a')
    f.write(proc_string)
    f.close()

def rescale(image, ratio):
    im = Image.open(image)
    w, h = im.size
    h_new = int(h * ratio)
    w_new = int(w * ratio)
    re = im.resize((w_new, h_new))
    
    re.save(image)

def add_text(image, (scale, data)):
    # opens and crops an image to the box given. box is a 4-tuple defining
    # the left, upper, right and lower pixel
    # in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide.
    # "University of Leeds" is 520 px wide.
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

    box = (left, upper, right, lower)

    draw = ImageDraw.Draw(im)
    draw.rectangle((left, upper, right, upper + 100), fill = 'black')
    draw.rectangle((left, lower - 100, right, lower), fill = 'black')

    # Want to write something to join the two images together. Will probably
    # mean splitting this function up to separate the text adding and cropping
    # parts - hard, as this requires a change of order.

    # this is platform specific!
    fonts = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(fonts, 45)

    text_hi_pos = (left, upper)
    text_low_pos = (left, lower - 100)

    text_hi, text_low = param_text, author_text

    draw.text(text_hi_pos, text_hi, font=font, fill="white")
    draw.text(text_low_pos, text_low, font=font, fill="white")

    cropped = im.crop(box)
    cropped.save(image)

def proc_images(proc, run, source, arg1, arg2):
    for image in glob.glob('%s/%s/cam1/*jpg' % (source, run)):
        print "performing %s on %s" % (proc, image)
        proc(image, arg1)
    for image in glob.glob('%s/%s/cam2/*jpg' % (source, run)):
        print "performing %s on %s" % (proc, image)
        proc(image, arg2)

def main(run, run_data=None):
    if run_data is None:
        run_data = get_run_data(run)
    # Barrel correct
    proc_images(barrel_corr, run, path + '/synced', None, None)

    # Pull the correction angles from the run data
    theta1 = -float(run_data['rot_1'])
    theta2 = -float(run_data['rot_2'])

    # Rotation correct the images
    proc_images(rotation_corr, run, path + '/processed', theta1, theta2)
 
    # Resize the images to standard
    ideal_ruler = 435
    ruler = int(run_data['bottom_1']) - int(run_data['ruler_25'])
    ruler_ratio = ideal_ruler / ruler
    depth_1 = int(run_data['bottom_1']) - int(run_data['surface_1'])
    depth_2 = int(run_data['bottom_2']) - int(run_data['surface_2'])
    depth_ratio = depth_1 / depth_2
    cam1_ratio = ruler_ratio
    cam2_ratio = ruler_ratio * depth_ratio

    proc_images(rescale, run, path + '/processed', cam1_ratio, cam2_ratio)
    # rescaling means that the offsets, lock_pos etc. are rescaled too.
    # Add text and crop
    proc_images(add_text, run, path + '/processed', (cam1_ratio, run_data), (cam2_ratio, run_data))

# MAIN SEQUENCE
#
#runs = [path.split('/')[-1] for path in glob.glob('synced/*')]
# runs = ['11_7_05c']

#for run in runs:
#    bc1(run)
#for run in runs:
#    barrel_corr_measure(run)
# for run in runs:
#     run_data = get_run_data(run)
#     main(run, run_data)
