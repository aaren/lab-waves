#####CONFIG#####

### **PATH** ###
# what is the root directory for all of this?
path = '/home/eeaol/lab/data/flume1/working'
# where is the parameters file?
paramf = path + '/parameters'
# proc_data?
procf = path + '/proc_data'
# where is the data going to be stored?? (filename)
data_dir = path + '/data/'
data_storage = data_dir + 'data/data_store_'
pdir = path + '/processed'
plots_dir = path + '/plots'
### /PATH ###

### **BASIC SETTINGS** ###
# Note: Modifying these settings requires re-acquiring the basic_data,
# which is computationally intensive.

# Specify a vertical region of the image in which to search for the interface.
# Units are pixels, starting from top of image.
region = (130, 540)

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

# Depths at which to scan for the current front. 0.4 is partial, 1 is full
# depth.
front_depths = {'0.4': 520, '1': 495}

# specify the positions of rulers and other vertical features that
# obscure the fluid.
rulers = {}
rulers['cam1'] = [(30, 65), (820, 900), (1580, 1610), (1690, 1760)]
rulers['cam2'] = [(80, 130), (950, 1000), (1820, 1890), \
                  (2665, 2695), (2705, 2725)]
### /BASIC SETTINGS ###

### **CAMERA SETUP** ###
# Settings here relate to the positioning of the cameras relative to the
# tank and each other. This allows for conversion from pixel dimensions
# to S.I and for the correction of parallax.
#
# where to crop the images? (left, right, upper, lower), relative to
# offset for left, right; scale_depth for upper; tank bottom for
# lower.
crop = {}
crop['cam1'] = (-10, 2750, -100, 150)
crop['cam2'] = (-2700, 150, -100, 150)

# distance from offset mark to zero point (lock side of lock gate)
# in cam1.
zero_offset = 2600
# Specify the offsets that each of the cameras have, for
# normalisation of pixel measurements
camera_offsets = {}
## the cam1 offset is the distance between wherever zero is in cam1
## and the left edge of cam1.
camera_offsets['cam1'] = (zero_offset - crop['cam1'][0], 543)
## the cam2 offset is the distance between wherever zero is in cam1
## and the left edge of *cam2*
camera_offsets['cam2'] = (zero_offset - crop['cam2'][0], 543)

# specify the scale, i.e how many pixels to some real measurement in the
# images. in y we want this to be the total fluid depth. in x make it the
# lock length for now (25cm).
fluid_depth = 543 - 109
lock_length = 440
scales = {}
scales['cam1'] = (lock_length, fluid_depth)
scales['cam2'] = (lock_length, fluid_depth)

# specify where the centre of the camera was pointing to.
# used in parallax_corr
centre = {}
centre['cam1'] = 0.75
centre['cam2'] = 2.25
### /CAMERA SETUP ###

#####/CONFIG#####
