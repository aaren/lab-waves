#####CONFIG#####

# perhaps this is a good place to use a class instead of all of these dicts?
# How about a Camera class, with attributes of rulers, offsets, scales, etc.

# what is the root directory for all of this?
path = '/home/eeaol/lab/data/flume1/working'
# where is the parameters file?
paramf = path + '/parameters'
# proc_data?
procf = path + '/proc_data'

# where is the data going to be stored?? (filename)
data_dir = '/home/eeaol/code/lab-waves/data/'
data_storage_file = data_dir + 'data/data_store_'

# specify a vertical region of the image in which to search for the interface
region = (130, 540)

# specify the threshold values to use. fiddling with these has a strong impact
# on the quality of the interface signal.
thresh_green = (80, 120, 50)
thresh_red = (100, 50, 10)
mixed_red = (100, 75, 10)
thresh_values = (thresh_green, thresh_red, mixed_red)
# TODO: Two values for thresh red? 
# thresh_red[1] controls how mixed the detected current is, as this strongly
# varies the greenness. Inside the core current this might be <50 (across the
# whole tank, the stuff against the front wall is <5), but in more mixed zones
# more like <75, or greater depending on the mixing. Two values for thresh_red
# would give an idea of the thickness of the mixed layer on the current.

# specify the positions of rulers and other vertical features that
# obscure the fluid.
rulers = {}
rulers['cam1'] = [(80, 105), (870, 950), (1630, 1660), (1740, 1810)]
rulers['cam2'] = [(80, 130), (950, 1000), (1820, 1890), \
                  (2665, 2695), (2705, 2725)]

# specify the offsets that each of the cameras have, for normalisation of
# pixel measurements
camera_offsets = {}
camera_offsets['cam1'] = (2650, 543)
## TODO the 2650 value here is the same as a value in threshold module
## they need to be the same. DO THEY ACTUALLY???
## the 2600 value is the distance from the identical place in cam1 to 
## the lock in cam1
camera_offsets['cam2'] = (camera_offsets['cam1'][0] + 2600, 543)

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

#####/CONFIG#####
