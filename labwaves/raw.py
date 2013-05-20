from __future__ import division

import glob
import os

import Image
import ImageFont
import numpy as np
import matplotlib.pyplot as plt

import config

import processing

import util


def read_parameters(run, paramf):
    """Read in data from the parameters file, which has
    the headers and data types as in the list data.
    'S10' is string, 'f4' is float.

    Reading in gives a numpy recarray. Convert to a dictionary
    and return this.
    """
    data = [('run_index',       'S10'),
            ('h_1',             'f8'),
            ('rho_0',           'f8'),
            ('rho_1',           'f8'),
            ('rho_2',           'f8'),
            ('alpha',           'f8'),
            ('D',               'f8'),
            ('sample',          'f8')]
    names, types = zip(*data)
    p = np.genfromtxt(paramf, dtype=types, names=names)
    index = np.where(p['run_index'] == run)
    params = p[index]
    if len(params) == 0:
        # run is not in parameters file
        print("{run} is not in {paramf}".format(run=run, paramf=paramf))
        raise KeyError
    else:
        # convert to dictionary
        names = params.dtype.names
        p_dict = dict(zip(names, params[0]))
        return p_dict


def read_run_data(run, paramf):
    """Read in data from proc file, which has headers and data
    types as in the list data. 'S10' is string, 'i4' is integer.

    Read in as numpy recarray. Convert this to a dictionary for output.
    """
    data = [('run_index',  'S10'),
            ('l0x',        'i4'),
            ('l0y',        'i4'),
            ('lsx',        'i4'),
            ('lsy',        'i4'),
            ('j10x',       'i4'),
            ('j10y',       'i4'),
            ('j1sx',       'i4'),
            ('j1sy',       'i4'),
            ('leakage',    'i4'),
            ('odd_1',      'S10'),
            ('j20x',       'i4'),
            ('j20y',       'i4'),
            ('j2sx',       'i4'),
            ('j2sy',       'i4'),
            ('r0x',        'i4'),
            ('r0y',        'i4'),
            ('rsx',        'i4'),
            ('rsy',        'i4'),
            ('odd_2',      'S10')]
    names, dtypes = zip(*data)
    rd = np.genfromtxt(paramf, skip_header=1,
                       dtype=dtypes,
                       delimiter=',',
                       names=names)
    index = np.where(rd['run_index'] == run)
    rdp = rd[index]
    # convert to dictionary
    rdp_dict = dict(zip(rdp.dtype.names, rdp[0]))
    return rdp_dict


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


class RawImage(object):
    """Represents an individual image from a lab run.

    We want to do a series of things to a raw lab image:

        - barrel correct (self.barrel_correct)
        - perspective correct (self.perspective_correct)
        - crop and add borders (self.crop_text)

    Each of these only makes sense to do if the previous steps have
    been carried out already. Therefore when one of the above
    methods is called, the preceding methods are called as well.

    self.process is a wrapper function that returns a completely
    processed image object

    After all these steps have been performed we want to write
    the image to persistent storage.

    self.write_out - writes the image to disk, the location determined
                     by the RawRun instance path and the config file.
    """
    def __init__(self, path, run):
        """A RawImage is a member of a RawRun - images don't exist
        outside of a run. To initialise a RawImage, a RawRun instance
        must be passed as input.

        Inputs: path - path to an image file
                run - a RawRun instance

        Each image originates from a specific camera and has a
        frame number, both of which are encoded in the file path.

        The camera determines the coefficients used in the correction
        routines.

        The frame number to determine the time that an image
        corresponds to.
        """
        self.path = path
        self.fname = os.path.basename(path)
        self.dirname = os.path.dirname(path)

        self.frame = iframe(path)
        self.cam = icam(path)
        self.run_index = run.index

        self.processed_path = os.path.join(run.path,
                                           config.outdir,
                                           run.index,
                                           self.cam,
                                           self.fname)

        self.im = Image.open(path)

        self.barrel_coeffs = config.barrel_coeffs[self.cam]
        self.perspective_coefficients = run.perspective_coefficients[self.cam]
        self.crop_box = run.crop_box(self.cam)
        self.parameters = run.parameters
        self.param_text = config.param_text.format(time=self.time,
                                                   **self.parameters)

    @property
    def time(self):
        """Determine the time that an image corresponds to."""
        index = int(self.frame) - 1
        sample_interval = self.parameters['sample']
        time_stamp = index * sample_interval
        return time_stamp

    def barrel_correct(self):
        """Barrel correct an image. Returns barrel corrected image
        as Image object.
        """
        bc_im = processing.barrel_correct(self.im, self.barrel_coeffs)
        return bc_im

    def perspective_transform(self):
        """Perspective correct an image."""
        bc_im = self.barrel_correct()
        coeffs = self.perspective_coefficients
        trans = processing.perspective_transform(bc_im, coeffs)
        return trans

    def crop_text(self):
        """Crop, add borders and add text."""
        trans_im = self.perspective_transform()
        crop_box = self.crop_box
        crop_im = processing.crop(trans_im, crop_box)

        kwargs = {'upper_text': self.param_text,
                  'lower_text': config.author_text,
                  'upper_bar': config.top_bar,
                  'lower_bar': config.bottom_bar,
                  'font': ImageFont.truetype(config.font, 40),
                  'text_colour': 'white',
                  'bg_colour': 'black'}

        dcim = processing.draw_text(crop_im, **kwargs)
        return dcim

    def process(self):
        return self.crop_text()

    def write_out(self):
        """Write the processed image to disk."""
        processed_im = self.process()
        util.makedirs_p(os.path.dirname(self.processed_path))
        processed_im.save(self.processed_path)


class RawRun(object):
    """Represents a lab run in its raw state.

    A lab run consists of a set of raw images and some run metadata.

    The metadata is contained in a parameters file and a run_data file.

    This class uses the run metadata to create arguments for the functions
    in processing and uses the RawImage class to process a whole run.
    """
    def __init__(self, run, parameters_f=None, run_data_f=None, path=None):
        """

        Inputs: run - string, a run index, e.g. 'r11_05_24a'
                parameters_f - optional, a file containing run parameters
                run_data_f - optional, a file containing run_data
                path - optional, root directory for this run
        """
        self.index = run
        self.config = config
        if not path:
            self.path = config.path
        else:
            self.path = path
        self.input_dir = os.path.join(self.path, config.indir, self.index)
        if not parameters_f:
            self.parameters = read_parameters(run, config.paramf)
        else:
            self.parameters = read_parameters(run, parameters_f)

        if not run_data_f:
            run_data_f = config.procf
        self.run_data_f = run_data_f

        self.bc1_outdir = 'tmp/bc1'
        self.cameras = ['cam1', 'cam2']

    @property
    def run_data(self):
        """Get the data for the run from the given file (procf).

        If the data doesn't exist in the file, prompt to measure
        the data and call if user says yes.
        """
        procf = self.run_data_f
        proc_runs = util.pull_col(0, procf, ',')
        try:
            proc_runs.index(self.index)
        except ValueError:
            print "%s is not in the procf (%s)" % (self.index, procf)
            print "get the proc_data for this run now? (y/n)"
            A = raw_input('> ')
            if A == 'y':
                self.measure(procf)
                self.get_run_data(procf)
            elif A == 'n':
                return 0
            else:
                print "y or n!"
                self.get_run_data(procf)

        run_data = read_run_data(self.index, procf)

        return run_data

    def measure(self, procf=config.procf):
        #TODO: doc
        # TODO: make sense in class
        proc = []
        proc.append(self.index)
        for camera in ['cam1', 'cam2']:
            plt.figure(figsize=(16, 12))
            # TODO: fix this path
            simg1 = '{path}/barrel_corr/{run}/{cam}/img_0001.jpg'
            img1 = simg1.format(path=self.path, run=self.index, cam=camera)
            try:
                im = Image.open(img1)
            except IOError:
                print "Bad image for %s %s" % (self.index, camera)
                break
            plt.imshow(im)
            plt.xlim(2500, 3000)
            plt.ylim(750, 1500)
            plt.draw()
            print "Select lock base and surface"
            pt1 = plt.ginput(3, 0)
            if camera == 'cam1':
                plt.xlim(0, 500)
            elif camera == 'cam2':
                plt.xlim(750, 1250)
            plt.draw()
            print "Select join base and surface"
            pt2 = plt.ginput(3, 0)

            pts = pt1[0:2] + pt2[0:2]
            for x, y in pts:
                proc.append(int(x))
                proc.append(im.size[1] - int(y))

            plt.xlim(0, 3000)
            plt.draw()
            if camera == 'cam1':
                print "What is the extent of lock leakage?"
                leak = plt.ginput(2, 0)[0][0]
                proc.append(int(pt1[0][0] - leak))
            print "Weird (y/n)"
            proc.append(raw_input('> '))
            plt.close()

        proc = [str(e) for e in proc]
        entry = ','.join(proc) + '\n'
        f = open(procf, 'a')
        f.write(entry)
        f.close()

    def bc1(self):
        """Just barrel correct the first image from each camera from
        a run and save it to the folder self.bc1_outdir under
        self.path.
        """
        for camera in self.cameras:
            ref_image_path = self.ref_image_path(camera)
            im1 = RawImage(ref_image_path, self)
            bim1 = im1.barrel_correct()

            bc1_outdir = self.bc1_outdir
            dirname = os.path.join(self.path, bc1_outdir, self.index, camera)
            basename = 'img_0001.jpg'
            outf = os.path.join(dirname, basename)

            util.makedirs_p(dirname)
            bim1.save(outf)

    def ref_image_path(self, cam):
        """Return the path to the first image from given camera.

        Inputs: cam - a string, e.g. 'cam1'
        """
        rundir = self.input_dir
        im_re = 'img_0001.jpg'
        cam_re = cam
        im_cam_re = cam_re + '/' + im_re
        image_path = glob.glob(os.path.join(rundir, im_cam_re))[0]
        return image_path

    @property
    def imagepaths(self):
        """Return a list of the full path to all of the images
        in the run.
        """
        rundir = self.input_dir
        # TODO: put these re in config?
        im_re = 'img*jpg'
        cam_re = 'cam*'
        im_cam_re = cam_re + '/' + im_re
        imagelist = glob.glob(os.path.join(rundir, im_cam_re))
        return sorted(imagelist)

    @property
    def images(self):
        """Returns a list of RawImage objects, corresponding to each
        image in the run.
        """
        paths = self.imagepaths
        return [RawImage(p, run=self) for p in paths]

    @property
    def perspective_coefficients(self):
        """Generate the cam1 and cam2 perspective transform coefficients
        for a given run.

        Inputs: run - string, the run index

        Outputs: dictionary of the camera coefficients
                d.keys() = ['cam1', 'cam2']
                d['cam1'] = (a, b, c, d, e, f, g, h)
        """
        run_data = self.run_data

        lock_0 = (run_data['l0x']), (run_data['l0y'])
        lock_surf = (run_data['lsx']), (run_data['lsy'])
        join1_0 = (run_data['j10x']), (run_data['j10y'])
        join1_surf = (run_data['j1sx']), (run_data['j1sy'])

        join2_0 = (run_data['j20x']), (run_data['j20y'])
        join2_surf = (run_data['j2sx']), (run_data['j2sy'])
        ruler_0 = (run_data['r0x']), (run_data['r0y'])
        ruler_surf = (run_data['rsx']), (run_data['rsy'])
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

        cam1_coeff = tuple(processing.perspective_coefficients(x1, X1))
        if run_data['j20x'] == 0:
            cam2_coeff = 0
        else:
            cam2_coeff = tuple(processing.perspective_coefficients(x2, X2))

        return {'cam1': cam1_coeff, 'cam2': cam2_coeff}

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
        l0x = run_data['l0x']
        l0y = run_data['l0y']
        j20x = run_data['j20x']
        j20y = run_data['j20y']
        ref = {'cam1': (l0x, l0y), 'cam2': (j20x, j20y)}
        left = ref[cam][0] + config.crop[cam][0]
        right = ref[cam][0] + config.crop[cam][1]
        upper = ref[cam][1] - config.ideal_25 + config.crop[cam][2]
        lower = ref[cam][1] + config.crop[cam][3]
        return (left, upper, right, lower)

    def process(self):
        """Process the images, transforming them from their raw state.

        Can easily multiprocess this bit.
        """
        for image in self.images:
            image.write_out()
