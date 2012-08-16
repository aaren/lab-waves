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

# Scale for the images
ideal_25 = 400
ideal_m = ideal_25 * 4
ideal_base_1 = int(1.47 * ideal_m)
ideal_base_2 = int(0.99 * ideal_m)

# Thickness of border above and below images.
top_bar = 50
bottom_bar = 60

# Specify a vertical region of the image in which to search for the interface.
# Units are pixels, starting from top of image.
region = (top_bar + 10, top_bar + 400)

# Depths at which to scan for the current front. 0.4 is partial, 1
# is full depth (fractions of H, i.e. non dimensional).
# What height to scan for front up to?
h = 0.6
s = [i / 100. for i in range(0, int(h * 100 + 1))]
d = {'0.4': s, '1': s}
front_depths = \
        {k: [int(top_bar + (1 - i) * ideal_25) for i in d[k]] for k in d}

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

cl = {'cam1': 1.46, 'cam2': 3.06}
cr = {'cam1': -0.05, 'cam2': 1.39}

# Some reference points for cropping. Should be the same as the x
# coord of the se corner of the reference box chosen in
# measurements, i.e. the invariant point in the perspective
# transformation.
crop_ref = {'cam1': 0.00, 'cam2': 1.51}

crop= {cam: (int(-(cl[cam] - crop_ref[cam]) * ideal_m), \
             int(-(cr[cam] - crop_ref[cam]) * ideal_m), \
            -50, \
             110) for cam in ('cam1', 'cam2')}

# Specify the positions of rulers and other vertical features that
# obscure the fluid. Measured in metres from the lock.

real_rulers = {}
real_rulers['cam1'] = [(0.48, 0.525), (0.99, 1.02)]
real_rulers['cam2'] = [(1.46, 1.54), (1.99, 2.02), (2.49, 2.52), (2.99, 3.02)]

rulers = {}
for cam in ['cam1', 'cam2']:
    rulers[cam] = [( int((cl[cam] - y) * ideal_m) , \
                     int((cl[cam] - x) * ideal_m) )
                        for x, y in real_rulers[cam]]

# Specify the offsets that each of the cameras have, for
# normalisation of pixel measurements
## the cam1 offset is the distance between wherever zero is in cam1
## and the *left* edge of cam1.
## the cam2 offset is the distance between wherever zero is in cam1
## and the *left* edge of *cam2*
camera_offsets = {cam: (cl[cam] * ideal_m, ideal_25 + top_bar)\
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
### /CAMERA SETUP ###

#####/CONFIG#####
