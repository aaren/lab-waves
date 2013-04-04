from __future__ import division

import glob

import Image
import ImageFont

import config

import processing

# TODO: these don't exist yet
import get_run_data
import get_parameters


class RawRun(object):
    def __init__(self, run):
        self.index = run
        # self.parameters = get_parameters(run, config.paramf)
        # self.run_data = get_run_data(run)

    @property
    def runfiles(self):
        """Build a list of the files from a run, along with the camera
        that they come from.

        Returns a list of tuples
        e.g. [('cam1', 'path/to/image1'), ('cam1', 'path/to/image2'), ...]
        """
        impaths = self.imagepaths()
        cams = [self.icam(impath) for impath in impaths]
        return zip(cams, impaths)

    def iframe(impath):
        """From an image filename, e.g. img_0001.jpg, get just the
        0001 bit and return it.

        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        frame = impath.split('_')[-1].split('.')[0].split('_')[-1]
        return frame

    def icam(impath):
        """Given a path to an image, extract the corresponding camera.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        cam = impath.split('/')[-2]
        return cam

    def irun(impath):
        """Given a path to an image, extract the corresponding run.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        run = impath.split('/')[-3]
        return run

    @property
    def imagepaths(self):
        """Return a list of the full path to all of the images
        in the run.
        """
        # TODO: put synced in config or something
        # TODO: composition of path names??
        # TODO: put these re in config?
        rundir = config.path + 'synced' + self.index
        im_re = 'img*jpg'
        cam_re = 'cam*'
        im_cam_re = cam_re + '/' + im_re
        imagelist = glob.glob(rundir + im_cam_re)
        return imagelist

    ### Applicators
    # Only a function of the run, generate all context based on this.
    # These do actual, persistent IO
    # Should go into a class really
    # multiprocessing starts to come in here too
    def barrel_correct(self):
        for camera, image in self.runfiles:
            coeffs = config.barrel_coeffs[camera]
            im = Image.open(image)
            out = processing.barrel_correct(im, coeffs)
            out.save('blah')

    def perspective_transform(self):
        # TODO: flesh out
        coeffs = self.run_perspective_coefficients()
        for camera, image in self.runfiles:
            im = Image.open(image)
            trans = processing.perspective_transform(im, coeffs[camera])
            trans.save('blah')

    def crop_text(self):
        run_data = self.run_data

        for image, cam in self.runfiles:
            cim = processing.crop(image, run_data, cam)

            time = self.gen_time(image)
            param_text = self.gen_image_text(time)

            kwargs = {'upper_text': param_text,
                      'lower_text': config.author_text,
                      'upper_bar': config.top_bar,
                      'lower_bar': config.bottom_bar,
                      'font': ImageFont.truetype(config.font, 40),
                      'text_colour': 'white',
                      'bg_colour': 'black'}
            dcim = processing.draw_text(cim, **kwargs)

            dcim.save('blah')

    ### argument generators ###
    # No side effects, but rely on external context (config, run_data)
    def crop_box(self, cam):
        """Calculate box needed to crop an image to standard
        orientation.

        Inputs: run_data - a dictionary of measurements
                cam - string, the camera used for the image, e.g. 'cam1'

        Output: a tuple of four integers defining the box
                (left, upper, right, lower)
        """
        run_data = self.run_data
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
        return (left, upper, right, lower)

    def gen_image_text(self, time):
        parameters = self.parameters
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
        return param_text

    def run_perspective_coefficients(self):
        """Generate the cam1 and cam2 perspective transform coefficients
        for a given run.

        Inputs: run - string, the run index

        Outputs: dictionary of the camera coefficients
                d.keys() = ['cam1', 'cam2']
                d['cam1'] = (a, b, c, d, e, f, g, h)
        """
        run_data = self.run_data

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

        cam1_coeff = tuple(self.perspective_coefficients(x1, X1))
        if run_data['j20x'] == '0':
            cam2_coeff = 0
        else:
            cam2_coeff = tuple(self.perspective_coefficients(x2, X2))

        return {'cam1': cam1_coeff, 'cam2': cam2_coeff}

    def gen_time(self, image):
        # XXX: stub
        # 'time' that a frame represents
        # TODO: update this with 25hz camera in mind. need some time
        # generating function
        run, cam, frame = image.split('/')[-3:]
        time = int(frame.split('.')[0]) - 1
        return time


# this class is at the run level.
# the other level that we could conceive of using is the image level
# say by extending the PIL Image class by composition.
# then we would do things like
#
# LabImage = compose(Image)

# im = LabImage.open(file)
# im.barrel_correct()
# im.crop_text()

# this isn't consistent with the design of my pure functions so far,
# which operate on a Image object.

# It doesn't really make sense to process images in isolation - the
# context for doing so is determined by the run that they belong to
# anyway. Therefore, stick with this run level class.
