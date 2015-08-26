#####CONFIG#####

### **PATH** ###
# what is the root directory for all of this?
path = '/home/eeaol/lab/data/flume1/working'
# where is the parameters file?
paramf = path + '/parameters'
# proc_data?
procf = path + '/proc_data'
# input directory
indir = 'synced'
# output directory
outdir = 'processed'
# where is the data going to be stored?? (filename)
data_dir = path + '/data/'
data_storage = data_dir + 'data/data_store_'
pdir = path + '/processed'
plots_dir = path + '/plots'
### /PATH ###


### **BASIC SETTINGS** ###
# Note: Modifying these settings requires re-acquiring the basic_data,
# which is computationally intensive.

# specify the threshold values to use. fiddling with these has a strong impact
# on the quality of the interface signal.
thresh_green = (80, 120, 50)
#thresh_green = (60, 100, 20)
core_red = (100, 50, 10)
mixed_red = (100, 75, 10)
thresh_values = (thresh_green, core_red, mixed_red)
# thresh_red[1] controls how mixed the detected current is, as this strongly
# varies the greenness. Inside the core current this might be <50 (across the
# whole tank, the stuff against the front wall is <5), but in more mixed zones
# more like <75, or greater depending on the mixing. Two values for thresh_red
# gives an idea of the thickness of the mixed layer on the current.

# Scale for the images (number of pixels per 0.25m)
ideal_25 = 200
ideal_m = ideal_25 * 4
ideal_base_1 = int(1.47 * ideal_m)
ideal_base_2 = int(0.99 * ideal_m)

# Thickness of border above and below images.
top_bar = 50
bottom_bar = 60

# Specify a vertical region of the image in which to search for the interface.
# Units are pixels, starting from top of image.
region = (top_bar + 10, top_bar + ideal_25)

# Depths at which to scan for the current front. 0.4 is partial, 1
# is full depth (fractions of H, i.e. non dimensional).
# What height to scan for front up to?
h = 0.6
s = [i / 100. for i in range(0, int(h * 100 + 1))]
d = {'0.4': s, '1': s}
front_depths = {k: [int(top_bar + (1 - i) * ideal_25) for i in d[k]]
                for k in d}

### /BASIC SETTINGS ###

### **CAMERA SETUP** ###
# Settings here relate to the positioning of the cameras relative to the
# tank and each other. This allows for conversion from pixel dimensions
# to S.I and for the correction of parallax.
#
# where to crop the images? (left, right, upper, lower), all relative to
# lock position for left, right; scale_depth for upper; tank bottom for
# lower. ensure that these are consistent with the source images!
# e.g. by comparison with proc_data.
# the borders (top / bottom bar) are added *outside* these values
crop = {'cam1': {'left':   1.80,
                 'right': -0.20,
                 'upper':  0.25,
                 'lower': -0.00},
        'cam2': {'left':   3.50,
                 'right':  1.39,
                 'upper':  0.25,
                 'lower': -0.00}
        }

## Perspective grid
# explicit measurements for the (x, y) position of the lower right,
# upper right, lower left, upper left perspective reference points,
# relative to the lower lock gate.
perspective_ref_points = {'old':   {'cam1': ((0.00, 0.00),
                                             (0.00, 0.25),
                                             (1.47, 0.00),
                                             (1.47, 0.25)),

                                    'cam2': ((1.51, 0.00),
                                             (1.51, 0.25),
                                             (2.50, 0.00),
                                             (2.50, 0.25))},

                          'new_1': {'cam1': ((-0.006, 0.00),
                                             (0.00, 0.25),
                                             (1.62, 0.00),
                                             (1.60, 0.25)),

                                    'cam2': ((1.80, 0.00),
                                             (1.80, 0.25),
                                             (3.40, 0.00),
                                             (3.40, 0.25))},

                          'new_2': {'cam1': ((-0.006, 0.00),
                                             (0.00, 0.25),
                                             (1.60, 0.00),
                                             (1.60, 0.25)),

                                    'cam2': ((1.80, 0.00),
                                             (1.80, 0.25),
                                             (3.40, 0.00),
                                             (3.40, 0.25))}
                          }

# a measurement where the cameras overlap
overlap = {'old':   1.40,
           'new_1': 1.80,
           'new_2': 1.80}

# Specify the positions of rulers and other vertical features that
# obscure the fluid. Measured in metres from the lock.

real_rulers = {}
real_rulers['cam1'] = [(0.49, 0.535), (0.99, 1.02)]
real_rulers['cam2'] = [(1.46, 1.54), (1.99, 2.02), (2.49, 2.52), (2.99, 3.02)]

rulers = {}
for cam in ['cam1', 'cam2']:
    rulers[cam] = [(int((crop[cam]['left'] - y) * ideal_m),
                    int((crop[cam]['left'] - x) * ideal_m))
                   for x, y in real_rulers[cam]]

# Specify the offsets that each of the cameras have, for
# normalisation of pixel measurements
## the cam1 offset is the distance between wherever zero is in cam1
## and the *left* edge of cam1.
## the cam2 offset is the distance between wherever zero is in cam1
## and the *left* edge of *cam2*
camera_offsets = {cam: (crop[cam]['left'] * ideal_m, ideal_25 + top_bar)
                  for cam in ('cam1', 'cam2')}

# specify the scale, i.e how many pixels to some real measurement in the
# images. in y we want this to be the total fluid depth. in x make it the
# lock length for now (25cm).
fluid_depth = ideal_25
lock_length = ideal_25
scales = {cam: (lock_length, fluid_depth) for cam in ('cam1', 'cam2')}

# specify where the centre of the camera was pointing to.
# used in parallax_corr
centre = {'cam1': 0.75, 'cam2': 2.25}
# where was the camera centred in the vertical?
# units are fractions of the fluid depth
# 0.3 seems sensible, based on measurements of images.
height = {'cam1': 0.3, 'cam2': 0.3}

# Barrel distortion coefficients for the cameras
# Sourced from lensfun-0.2.5/data/db/slr-nikon.xml
# two lenses used, Nikkor 18-55mm f/3.5-5.6G AF-S DX VR and
# Nikkor 18-70mm f/3.5-4.5G ED-IF AF-S DX
# coeffs = (a, b, c, d)
# d calculated as  a + b + c + d = 1
nikkor_1855 = (0.000658776, -0.0150048, -0.00123339, 1.01557914)
nikkor_1870 = (0.023969, -0.060001, 0.0, 1.036032)
barrel_coeffs = {'old':   {'cam1': nikkor_1855,
                           'cam2': nikkor_1870},
                 'new_1': {'cam1': nikkor_1855,
                           'cam2': nikkor_1855},
                 'new_2': {'cam1': nikkor_1855,
                           'cam2': nikkor_1855}
                 }
### /CAMERA SETUP ###

### **FONTS** ###
# font used in the processed images
# this is PLATFORM DEPENDENT
# in 15 pt Liberation Regular, "Aaron O'Leary" is 360 px wide,
# "University of Leeds" is 520 px wide.
font = '/usr/share/fonts/liberation/LiberationMono-Regular.ttf'
alt_font = '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf'
fontsize = ideal_25 / 10
# text that appears in the processed images
author_text = "Aaron O'Leary, University of Leeds"
param_text = ("run {run_index}, "
              "t = {time}s, "
              "h_1 = {h_1}, "
              "rho_0 = {rho_0}, "
              "rho_1 = {rho_1}, "
              "rho_2 = {rho_2}, "
              "alpha = {alpha}, "
              "D = {D}")
### /FONTS** ###

#####/CONFIG#####
