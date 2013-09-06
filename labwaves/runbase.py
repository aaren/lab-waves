from __future__ import division

import glob
import os
import itertools
from itertools import imap

import Image
import ImageDraw
import ImageFont
import numpy as np
import matplotlib.pyplot as plt

from gc_turbulence.gc_turbulence.util import parallel_process, parallel_stub

import config
import util

import processing
# module for extracting interfaces from image objects
# TODO: make this
import interface


def lazyprop(fn):
    """Decorator to allow lazy evaluation of class properties

    http://stackoverflow.com/questions/3012421/python-lazy-property-decorator

    usage:

        class Test(object):

            @lazyprop
            def a(self):
                print 'generating "a"'
                return range(5)

    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


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
            ('sample',          'f8'),
            ('perspective',     'S10')]
    names, types = zip(*data)
    try:
        p = np.genfromtxt(paramf, dtype=types, names=names)
    except ValueError:
        print("parameters file is malformed. Should have headings: np.dtype")
        print({k: v for k, v in data})

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
    try:
        rd = np.genfromtxt(paramf, skip_header=1,
                           dtype=dtypes,
                           delimiter=',',
                           names=names)
    except ValueError:
        print("run data file is malformed. Should have headings: np.dtype")
        print({k: v for k, v in data})

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

def real_to_pixel(x, y, cam='cam2'):
    """Convert a real measurement (in metres, relative to the lock
    gate) into a pixel measurement (in pixels, relative to the top
    left image corner, given the camera being used (cam).
    """
    x_ = (config.crop[cam]['left'] - x) * config.ideal_m
    h = config.crop[cam]['upper'] - config.crop[cam]['lower']
    y_ = config.top_bar + (h - y) * config.ideal_m
    return int(x_), int(y_)

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
        self.run = run

        self.fname = os.path.basename(path)
        self.dirname = os.path.dirname(path)

        self.frame = iframe(path)
        self.cam = icam(path)

        self.outpath = os.path.join(run.output_dir, self.cam, self.fname)

        f = open(path, 'rb')
        self.im = Image.open(f)
        if run.dump_file:
            # explicitly close the image file after loading to memory
            self.im.load()
            f.close()

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
        self.barrel_coeffs = config.barrel_coeffs[self.run.style][self.cam]
        bc_im = processing.barrel_correct(self.im, self.barrel_coeffs)
        return bc_im

    def perspective_transform(self):
        """Perspective correct an image."""
        bc_im = self.barrel_correct()
        self.perspective_coeffs = self.run.perspective_coefficients[self.cam]
        return processing.perspective_transform(bc_im, self.perspective_coeffs)

    def crop_text(self):
        """Crop, add borders and add text."""
        trans_im = self.perspective_transform()
        self.crop_box = self.run.crop_box(self.cam)
        crop_im = processing.crop(trans_im, self.crop_box)

        kwargs = {'upper_text': self.param_text,
                  'lower_text': config.author_text,
                  'upper_bar': config.top_bar,
                  'lower_bar': config.bottom_bar,
                  'font': ImageFont.truetype(config.font, config.fontsize),
                  'text_colour': 'white',
                  'bg_colour': 'black'}

        dcim = processing.draw_text(crop_im, **kwargs)
        return dcim

    def process(self):
        return self.crop_text()

    @lazyprop
    def processed(self):
        """Attribute for storing processed image in memory."""
        return self.process()

    def write_out(self):
        """Write the processed image to disk. Doesn't store anything
        in memory."""
        processed_im = self.process()
        util.makedirs_p(os.path.dirname(self.outpath))
        processed_im.save(self.outpath)

class ProcessedImage(object):
    def __init__(self, path, run):
        """A ProcessedImage is a member of a ProcessedRun - images
        don't exist outside of a run. To initialise a
        ProcessedImage, a ProcessedRun instance must be passed as
        input.

        Inputs: path - path to an image file
                run - a ProcessedRun instance

        Each image originates from a specific camera and has a
        frame number, both of which are encoded in the file path.

        The camera determines the coefficients used in the correction
        routines.

        The frame number to determine the time that an image
        corresponds to.
        """
        self.path = path
        self.run = run

        self.fname = os.path.basename(path)
        self.dirname = os.path.dirname(path)

        self.frame = iframe(path)
        self.cam = icam(path)

        self.output_dir = os.path.join(run.path, config.outdir, run.index)

        f = open(path, 'rb')
        self.im = Image.open(f)
        if run.dump_file:
            # explicitly close the image file after loading to memory
            self.im.load()
            f.close()

class StitchedImage(object):
    def __init__(self, im1, im2, join):
        """Create a Stitched Image, which is the joining of two
        images from different cameras.

        im1 -  cam1 image, ProcessedImage object
        im2 -  cam2 image, ProcessedImage object
        join - position in metres of the point at which to
               connect the images together

        Alternately, im1 and im2 can be PIL Image objects.
        """
        self.stitched_im = self.stitch(im1.im, im2.im, join)

        if type(im1) is ProcessedImage and type(im2) is ProcessedImage:
            output_dir = im1.output_dir
            self.fname = os.path.basename(im1.path)
        else:
            output_dir = './'
            self.fname = 'stitched_image.png'

        self.outpath = os.path.join(output_dir, 'stitched', self.fname)

    @staticmethod
    def stitch(im1, im2, join):
        """Stitch two images together, where im1 is taken by cam1
        and im2 is taken by cam2.

        im1 and im2 are PIL Image objects.
        Returns a PIL Image object.
        """
        c = config
        # the output size is determined by the outer edges of the
        # camera crop regions
        width = c.crop['cam2']['left'] - c.crop['cam1']['right']
        # and we'll use the full height of the images
        outsize = (int(width * c.ideal_m), im1.size[1])

        # you can join the images together anywhere you like as
        # long as is in the overlap between the two cameras
        # where is the join in cam2?
        join_c2 = real_to_pixel(join, 0, 'cam2')[0]
        # where is the join in cam1?
        join_c1 = real_to_pixel(join, 0, 'cam1')[0]

        # crop the images down
        cim1 = im1.crop((join_c1, 0, im1.size[0], im1.size[1]))
        cim2 = im2.crop((0, 0, join_c2, im2.size[1]))
        # output boxes for the cropped images to go in
        b2 = (0, 0, join_c2, cim2.size[1])
        b1 = (join_c2, 0, join_c2 + cim1.size[0], cim1.size[1])

        out = Image.new(mode='RGB', size=outsize)
        out.paste(cim1, box=b1)
        out.paste(cim2, box=b2)
        return out

    def draw_rectangle(self, box, fill, linewidth):
        """Draw a rectangle, defined by the coordinates in box, over
        the stitched image.

        box - (left, upper), (right, lower), pixel coordinates of
              the upper left and lower right rectangle corners.
        fill - colour to draw the lines with
        linewidth - width of the lines in pixels
        """
        draw = ImageDraw.Draw(self.stitched_im)
        (left, upper), (right, lower) = box
        draw.line((left, upper, right, upper), width=linewidth, fill=fill)
        draw.line((right, upper, right, lower), width=linewidth, fill=fill)
        draw.line((right, lower, left, lower), width=linewidth, fill=fill)
        draw.line((left, lower, left, upper), width=linewidth, fill=fill)
        return self

    def draw_rectangles(self, boxes, fill='red', linewidth=5):
        """Draw multiple rectangles."""
        for box in boxes:
            self.draw_rectangle(box, fill, linewidth)
        return self

    def write_out(self, path=None):
        """Save the stitched image to disk."""
        if path is None:
            path = self.outpath
        util.makedirs_p(os.path.dirname(path))
        self.stitched_im.save(path)


class BaseRun(object):
    """Base class for a lab run.

    Each run has associated metadata, contained in a parameters file
    and a run_data file.

    Each run is also stored within a specific base path, with
    the structure

        path/
            run1
            run2
            run3/
                cam1/
                    img_0001.jpg
                    img_0002.jpg
                    ...
                cam2/
                    ...
            ...

    """
    def __init__(self, run, parameters_f=None, run_data_f=None,
            path=None, dump_file=False):
        """
        Inputs: run - string, a run index, e.g. 'r11_05_24a'
                parameters_f - optional, a file containing run parameters
                run_data_f - optional, a file containing run_data
                path - optional, root directory for this run
                dump_file - default False, whether to explicitly load images
                            to memory before processing. The only reason that
                            this is needed is that parallel_process doesn't
                            deal with generators. TODO: Fix this upstream.
        """
        self.index = run

        self.path = path or config.path
        self.parameters_f = parameters_f or config.paramf
        self.run_data_f = run_data_f or config.procf

        self.dump_file = dump_file

        self.parameters = read_parameters(run, self.parameters_f)

    @property
    def input_dir(self):
        """Path to get input images from."""
        return os.path.join(self.path, self.indir, self.index)

    @property
    def output_dir(self):
        """Path to save output images to."""
        return os.path.join(self.path, self.outdir, self.index)

    @property
    def cameras(self):
        """Determine cameras used in run from input path."""
        camera_paths = glob.glob(os.path.join(self.input_dir, 'cam[0-9]'))
        return [os.path.basename(c) for c in camera_paths]

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
        return (self.Image(p, run=self) for p in paths)

    @property
    def style(self):
        """Returns a string, the perspective style used for the run."""
        return self.parameters['perspective']

class RawRun(BaseRun):
    """Represents a lab run in its raw state.

    A lab run consists of a set of raw images, represented
    by RawImage.
    """
    Image = RawImage
    bc1_outdir = 'tmp/bc1'
    indir = config.indir
    outdir = config.outdir

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
            elif A == 'n':
                return 0
            else:
                print "y or n!"
                run_data = self.run_data

        run_data = read_run_data(self.index, procf)

        return run_data

    def measure_camera(self, camera):
        plt.figure(figsize=(16, 12))
        # load up the barrel corrected first image
        bc1_path = self.bc1_image_path(camera)
        # if there isn't anything there, barrel correct the
        # first image
        # TODO: catch case that there is a missing camera
        if not os.path.exists(bc1_path):
            self.bc1()
        try:
            bc_im = Image.open(bc1_path)
        except IOError:
            print "Bad image for {run}, {cam}".format(run=self.index,
                                                      cam=camera)
            if camera == 'cam1':
                return [0, 0, 0, 0, 0, 0, 0, 0, 0, 999]
            elif camera == 'cam2':
                return [0, 0, 0, 0, 0, 0, 0, 0, 999]
            else:
                return None

        plt.imshow(bc_im)

        help_text = ("Select {} \nClick again to finish or "
                     "right click to cancel point.").format

        # set limits to zoom in on rough target area (lock)
        w, h = bc_im.size
        plt.xlim((w * 5) / 6, w)
        # ylim is inverted to make image right way up
        plt.ylim((h * 6) / 8, (h * 2) / 8)

        if camera == 'cam1':
            plt.title(help_text("lock base and surface"))

        elif camera == 'cam2' and self.style == 'old':
            plt.title(help_text("inner join base and surface"))

        elif camera == 'cam2' and 'new' in self.style:
            plt.title(help_text("right projection markers"))

        plt.draw()
        # ask for three points - the third indicates to move on
        pt1 = plt.ginput(3, 0)

        # set limits to zoom in on rough target area
        # this is the join for cam1 and the ruler for cam2
        if camera == 'cam1' and self.style == 'old':
            plt.xlim(0, w / 6)
            plt.title(help_text("inner join base and surface"))

        elif camera == 'cam2' and self.style == 'old':
            plt.xlim(w / 4, w / 2)
            plt.title(help_text("inner ruler base and projection to surface"))

        elif 'new' in self.style:
            plt.xlim(0, w / 6)
            plt.title(help_text("left projection markers"))

        plt.draw()
        # ask for three points - the third indicates to move on
        pt2 = plt.ginput(3, 0)

        # discard the third point
        pts = pt1[0:2] + pt2[0:2]
        points = list(itertools.chain.from_iterable([(int(x), int(y))
                                                     for x, y in pts]))

        plt.xlim(0, w)
        plt.draw()
        if camera == 'cam1':
            plt.title("What is the extent of lock leakage? \n"
                      "Click inside lock if none. \n"
                      "Click again to finish or right click to cancel point.")
            leak = plt.ginput(2, 0)[0][0]
            leakage = int(pt1[0][0] - leak)
            points += [leakage]

        print("Weird? (y/n)")
        weird = raw_input('> ')
        points += [weird]
        plt.close()

        return points

    def measure(self, procf=None):
        """Interactive tool for selecting perspective correction
        reference points in a run.

        Input: procf - path to a file to append the results to.
                       Default None will just return the result
                       list.

        Returns: a list of points used for perspective correction
                 and some strings.

        The returned list has entries corresponding to these
        headers in the proc file

            keys = ['run_index',
                    'l0x',
                    'l0y',
                    'lsx',
                    'lsy',
                    'j10x',
                    'j10y',
                    'j1sx',
                    'j1sy',
                    'leakage',
                    'odd_1',
                    'j20x',
                    'j20y',
                    'j2sx',
                    'j2sy',
                    'r0x',
                    'r0y',
                    'rsx',
                    'rsy',
                    'odd_2']

        Internally, this uses matplotlib ginput to display an image
        and allow user input.

        The image used is the barrel corrected first image from each
        camera.

        There are two styles of image: old and new.

        Old style images - from lab runs 2011 and earlier. No fixed
                           markers in the image so have to rely on
                           features of the tank. Specifically, the
                           lock gate and the right edge of the tank
                           join for cam1; the left edge of the tank
                           join and a ruler for cam2. These are the
                           markers in the horizontal - each yields
                           two points; one at the bottom of the tank
                           and one at the surface of the water.

        New style images - have markers made of brown tape at measured
                           points on the front of the tank.

                           There are actually two kinds of new style
                           image as one of the markers was incorrectly
                           positioned for a few runs.

        TODO: test this somehow
        """
        proc = []
        proc.append(self.index)
        for camera in self.cameras:
            proc += self.measure_camera(camera)

        entry = ','.join([str(e) for e in proc]) + '\n'
        if not procf:
            return proc
        else:
            f = open(procf, 'a')
            f.write(entry)
            f.close()
            return proc

    def bc1(self):
        """Just barrel correct the first image from each camera from
        a run and save it to the folder self.bc1_outdir under
        self.path.
        """
        for camera in self.cameras:
            try:
                ref_image_path = self.ref_image_path(camera)
            except IndexError:
                break
            im1 = RawImage(ref_image_path, self)
            bim1 = im1.barrel_correct()

            # define output file and make directories for it
            outf = self.bc1_image_path(camera)
            dirname = os.path.dirname(outf)
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

    def bc1_image_path(self, camera):
        """Return the path to the first barrel corrected image
        from given camera.

        Inputs: cam - a string, e.g. 'cam1'
        """
        bc1_path = os.path.join(self.path,
                                self.bc1_outdir,
                                self.index,
                                camera,
                                'img_0001.jpg')
        return bc1_path

    def perspective_reference(self, reference_point, style, camera):
        """Return a tuple of pixel coordinates:

            (lower_right, upper_right, lower_left, upper_left)

        Each coordinate is a tuple of integers (x, y).

        Inputs:
            reference_point - (x, y) pixel units. The coordinates of the lower
                              left corner in pixels.
            style - string, e.g. 'old' will give a grid that fits old style
                            images
            camera - string, e.g. 'cam1', the camera to get the grid for.
        """
        points = config.perspective_ref_points[style][camera]
        ref_x, ref_y = reference_point
        # coordinates of se corner of perspective quad
        x0, y0 = points[0]
        # number of pixels per metre
        metre = config.ideal_m
        pixel_points = [(ref_x - int(metre * (x - x0)),
                         ref_y - int(metre * (y - y0)))
                        for x, y in points]

        return pixel_points

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

        lower_right_1 = (run_data['l0x']), (run_data['l0y'])
        upper_right_1 = (run_data['lsx']), (run_data['lsy'])
        lower_left_1 = (run_data['j10x']), (run_data['j10y'])
        upper_left_1 = (run_data['j1sx']), (run_data['j1sy'])

        lower_right_2 = (run_data['j20x']), (run_data['j20y'])
        upper_right_2 = (run_data['j2sx']), (run_data['j2sy'])
        lower_left_2 = (run_data['r0x']), (run_data['r0y'])
        upper_left_2 = (run_data['rsx']), (run_data['rsy'])
        # need some standard vertical lines in both cameras.
        # cam1: use lock gate and tank join
        # cam2: tank join and ruler at 2.5m
        # (checked to be vertical, extrapolate to surface)
        # so for each camera, 4 locations (8 numbers) need
        # to be recorded.

        style = self.style

        # units here are pixels, i.e in the coordinate system of
        # an image.
        x1 = (lower_right_1, upper_right_1, lower_left_1, upper_left_1)
        X1 = self.perspective_reference(lower_right_1, style, 'cam1')

        x2 = (lower_right_2, upper_right_2, lower_left_2, upper_left_2)
        X2 = self.perspective_reference(lower_right_2, style, 'cam2')

        cam1_coeff = tuple(processing.perspective_coefficients(x1, X1))
        if lower_right_2[0] == 0:
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

        # x coord of invariant point in the perspective transform (se corner)
        crop_ref_x = config.perspective_ref_points[self.style][cam][0][0]
        crop_ref_y = config.perspective_ref_points[self.style][cam][0][1]

        m = config.ideal_m

        crop = {'left':  int(-(config.crop[cam]['left'] - crop_ref_x) * m),
                'right': int(-(config.crop[cam]['right'] - crop_ref_x) * m),
                'upper': int(-(config.crop[cam]['upper'] - crop_ref_y) * m),
                'lower': int(-(config.crop[cam]['lower'] - crop_ref_y) * m)}

        # define the box to crop the image to relative to the
        # invariant point in the projection transform (se).
        if cam == 'cam1':
            ref_x = run_data['l0x']
            ref_y = run_data['l0y']
        elif cam == 'cam2':
            ref_x = run_data['j20x']
            ref_y = run_data['j20y']
        else:
            raise Exception('Invalid camera selected')

        left = ref_x + crop['left']
        right = ref_x + crop['right']
        upper = ref_y + crop['upper'] - config.top_bar
        lower = ref_y + crop['lower'] + config.bottom_bar
        return (left, upper, right, lower)

    def process(self):
        """Process the images, transforming them from their raw state.

        Can easily multiprocess this bit.
        """
        kwargs = [{'image': i} for i in self.images]
        parallel_process(process_raw, kwargs, processors=10)

class ProcessedRun(BaseRun):
    """Same init as RawRun. At some point these two will be merged
    into a single Run class.
    """
    Image = ProcessedImage
    indir = config.outdir

    @property
    def stitched_images(self):
        cam1_images = (im for im in self.images if im.cam == 'cam1')
        cam2_images = (im for im in self.images if im.cam == 'cam2')
        overlap = config.overlap[self.style]
        join = (overlap for im in self.images)
        return imap(StitchedImage, cam1_images, cam2_images, join)

    @staticmethod
    def visible_regions(stitched_images):
        """Yield a sequence of stitched images that have
        rectangles overlaid on the sections of the tank
        in which PIV experiments could be performed.

        In red are standard locations. In yellow are the equivalent
        locations were the lock to shift downstream by 0.75m.

        stitched_images is a sequence of StitchedImage objects

        Returns a generator.
        """
        # visible regions
        vis = (((1.18, 0.25), (0.71, 0.0)),
               ((2.11, 0.25), (1.70, 0.0)),
               ((3.08, 0.25), (2.70, 0.0)))

        # shifted visible regions
        vis_ = (((0.43, 0.25), (-0.04, 0.0)),
                ((1.36, 0.25), (0.95, 0.0)),
                ((2.33, 0.25), (1.95, 0.0)))

        boxes = [[real_to_pixel(*x, cam='cam2') for x in X] for X in vis]
        boxes_ = [[real_to_pixel(*x, cam='cam2') for x in X] for X in vis_]
        for si in stitched_images:
            si.draw_rectangles(boxes_, fill='yellow')
            si.draw_rectangles(boxes, fill='red')
            yield si

    def interface(self):
        """Grab all interface data from a run"""
        # TODO: multiprocessing
        # TODO: runfiles should point to processed data
        for im in self.images:
            interfaces = interface.interface(im)
            qc_interfaces = [interface.qc(i) for i in interfaces]
            save_interface(qc_interfaces)

    def write_out(self, images):
        """Write images to disk.

        images is a sequence of image objects that have a
        write_out() method

        e.g. self.write_out(self.stitched_images)
        """
        for im in images:
            im.write_out()


@parallel_stub
def process_raw(image):
    """External function to allow multiprocessing raw images."""
    image.write_out()
