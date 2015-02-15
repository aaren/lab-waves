from __future__ import division

import glob
import os
import itertools
from itertools import imap
import json

import Image
import ImageDraw
import ImageFont
import numpy as np
import matplotlib.pyplot as plt
from skimage import filter as skif
import h5py
import scipy.interpolate as interp

from parallelprogress import parallel_process, parallel_stub

import config
import util

import processing


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


def read_parameters(paramf):
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
            ('H',               'f8'),
            ('sample',          'f8'),
            ('perspective',     'S10')]
    names, types = zip(*data)
    try:
        p = np.genfromtxt(paramf, dtype=types, names=names)
    except ValueError:
        print("parameters file is malformed. "
                "Should have headings: {}".format(names))
        print({k: v for k, v in data})
    return p


class LabImage(object):
    """Base class for images that come from a lab run."""
    def __init__(self, path, run, im=None):
        """A LabImage is a member of a Run - images don't exist
        outside of a run. To initialise a RawImage, a Run instance
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

        self.frame = self.iframe(path)
        self.cam = self.icam(path)

        self.outformat = '.jpg'
        # set the image file format in the filename
        self.out_fname = os.path.splitext(self.fname)[0] + self.outformat
        self.outpath = os.path.join(run.output_dir, self.cam, self.out_fname)

        if os.path.exists(path) and not im:
            f = open(path, 'rb')
            self.im = Image.open(f)
            if run.dump_file:
                # explicitly close the image file after loading to memory
                self.im.load()
                f.close()

        elif im:
            self.im = im
            # NB reassignment of outpath when im is given
            # path has different meaning in this case
            self.outpath = path

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

    @staticmethod
    def iframe(impath):
        """From an image filename, e.g. img_0001.jpg, get just the
        0001 bit and return it.

        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        frame = impath.split('_')[-1].split('.')[0].split('_')[-1]
        return frame

    @staticmethod
    def icam(impath):
        """Given a path to an image, extract the corresponding camera.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        cam = impath.split('/')[-2]
        return cam

    @staticmethod
    def irun(impath):
        """Given a path to an image, extract the corresponding run.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        run = impath.split('/')[-3]
        return run


class RawImage(LabImage):
    """Represents an individual raw image from a lab run.

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

        try:
            font = ImageFont.truetype(config.font, config.fontsize)
        except IOError:
            font = ImageFont.truetype(config.alt_font, config.fontsize)

        kwargs = {'upper_text': self.param_text,
                  'lower_text': config.author_text,
                  'upper_bar': config.top_bar,
                  'lower_bar': config.bottom_bar,
                  'font': font,
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


class ProcessedImage(LabImage):
    def draw_rectangle(self, box, fill, linewidth):
        """Draw a rectangle, defined by the coordinates in box, over
        the stitched image.

        box - (left, upper), (right, lower), pixel coordinates of
              the upper left and lower right rectangle corners.
        fill - colour to draw the lines with
        linewidth - width of the lines in pixels
        """
        # you can draw outside the image box with PIL
        draw = ImageDraw.Draw(self.im)
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

    @property
    def with_visible_regions(self):
        """ Create rectangles overlaid on the sections of the tank
        in which PIV experiments could be performed.

        In red are standard locations. In yellow are the equivalent
        locations were the lock to shift downstream by 0.75m.

        Returns self so that you can use in a chain.
        """
        # visible region
        vis = ((3.08, 0.25), (2.70, 0.0))

        # shifted lock equivalent visible region
        vis_s = ((2.33, 0.25), (1.95, 0.0))

        # elongated lock visible region
        def elong(x, n=4, L=0.25):
            """Calculate equivalent x position given an elongation
            of the lock gate of length L by a factor of n, where
            x=0 is found at n=1."""
            return (x - (n - 1) * L) / n

        vis_e = (((elong(3.08, n=4), 0.25), (elong(2.70, n=4), 0.0)),
                 ((elong(3.08, n=3), 0.25), (elong(2.70, n=3), 0.0)),
                 ((elong(3.08, n=2), 0.25), (elong(2.70, n=2), 0.0)),)

        box = [[self.run.real_to_pixel(*x, cam='cam2') for x in vis]]
        box_s = [[self.run.real_to_pixel(*x, cam='cam2') for x in vis_s]]
        box_e = [[self.run.real_to_pixel(*x, cam='cam2') for x in X]
                                                            for X in vis_e]

        self.draw_rectangles(box, fill='red')
        self.draw_rectangles(box_s, fill='yellow')
        self.draw_rectangles(box_e, fill='cyan')

        return self

    @lazyprop
    def measurement_region(self):
        """Remove the bars from the top and bottom of a lab image."""
        return self.im.crop(self.run.measurement_box(self.cam))

    @lazyprop
    def imarray(self):
        """Numpy array of the measurement region of the image."""
        return np.asarray(self.measurement_region)

    @property
    def pixel_coords(self):
        """Return the pixel coordinate of each point in the
        measurement region.

        Outputs two arrays: the x coordinates and the y coordinates.
        """
        return self.run.pixel_coords(cam=self.cam)

    @property
    def real_coords(self):
        """For each pixel in the measurement region, compute
        the real coordinate.

        Outputs two arrays: the x coordinates and the y coordinates.
        """
        return self.run.real_coords(cam=self.cam)

    @property
    def channels(self):
        """Return the r, g, b channels of the lab image,
        normalised on the range [0, 1].
        """
        return [c.squeeze() / 255. for c in np.dsplit(self.imarray, 3)]

    @property
    def current_fluid(self):
        """Return the color space projection that best captures
        the gravity current fluid.

        The green channel picks it out the best as a single channel,
        but is contaminated by background noise.

        This noise is largely neutrally toned though, so we can
        subtract one of the other channels to remove it.
        """
        r, g, b = self.channels
        return r - g

    @property
    def masked_current_fluid(self):
        return np.ma.masked_where(self.current_mask, self.current_fluid)

    @property
    def wave_fluid(self):
        """Colour space projection that best captures
        the lower layer fluid.

        The blue channel provides a very clear signal.
        The green looking fluid is marked by an absence
        of blue.
        """
        r, g, b = self.channels
        return b

    @property
    def real_rulers(self):
        """Where are the rulers in real space."""
        cam = self.cam
        real_rulers = config.real_rulers[cam]

        style = self.run.style
        if style == 'old':
            pass
        elif 'new' in style and cam == 'cam1':
            # only one ruler in cam1
            real_rulers = [real_rulers[0]]
        elif 'new' in style and cam == 'cam2':
            # no rulers
            real_rulers = None

        return real_rulers

    @property
    def ruler_mask(self):
        """Mask for rulers in the image."""
        rx, ry = self.real_coords
        if not self.real_rulers:
            return np.zeros(rx.shape).astype(np.bool)
        # Truth wherever there is a ruler
        mask = reduce(np.logical_or, ((x1 < rx) & (rx < x2)
                                      for x1, x2 in self.real_rulers))

        return mask

    @property
    def bottom_mask(self):
        """Mask the bottom 5 pixels of the image."""
        ix, iy = self.pixel_coords
        mask = iy > self.measurement_region.size[1] - 5
        return mask

    @property
    def top_mask(self):
        """Mask the top 5cm of the image."""
        ix, iy = self.real_coords
        mask = iy > 0.20
        return mask

    @property
    def crop_mask(self):
        """Mask where the image has been cropped, i.e. there is black
        background."""
        return np.all(self.imarray < 0.01, axis=-1)

    @property
    def lock_mask(self):
        """Mask the region behind the lock gate."""
        return self.real_coords[0] < 0.05

    @property
    def wave_mask(self):
        """Combination of masks that masks out irrelevant features
        for the wave fluid detection."""
        return reduce(np.logical_or, (self.ruler_mask,
                                      self.top_mask,
                                      self.bottom_mask,
                                      self.crop_mask,
                                      self.lock_mask,))

    @property
    def current_mask(self):
        """Combination of masks that masks out irrelevant features
        for the lock fluid detection."""
        return reduce(np.logical_or, (self.ruler_mask,
                                      self.bottom_mask,
                                      self.crop_mask,
                                      self.current_fluid < 0.1,
                                      self.real_coords[0] < 0,))

    @property
    def wave_interface(self):
        """Pull out the wave interface."""
        mask = self.wave_mask
        array = (self.wave_fluid < 0.1).astype(np.float)
        x, y = self.canny_interface(array, sigma=5, mask=~mask)
        return x, y

    @property
    def current_interface(self):
        """Pull out the lock interface."""
        mask = self.current_mask
        array = (self.current_fluid < 0.5).astype(np.float)
        x, y = self.canny_interface(array, sigma=5, mask=~mask)
        return x, y

    @property
    def wave_profile(self):
        """Define the upper limit of the gravity current
        for each horizontal position in the visible region
        of the image.
        """
        x, y = self.wave_interface
        # TODO: do this a bit more elegantly
        # like with a mapper from image coords to measured image
        # coords
        y += config.top_bar
        ix, iy = self.pixel_coords
        w, h = self.measurement_region.size

        X = np.arange(w)
        Y = np.zeros((w,))
        # there must be a way to vectorize this...
        # if behind the front
        for i in xrange(w):
            _y = y[np.where(x == i)]
            # if no point, put a nan placemarker
            if _y.size == 0:
                Y[i] = np.nan
            # otherwise, take highest (lowest y pixel)
            elif _y.size >= 1:
                Y[i] = _y.min()

        # interpolate over gaps (nan) behind the front
        nans = np.isnan(Y)
        # numpy one dimensional interpolation - more in
        # scipy.interpolate
        Y[nans] = np.interp(X[nans], X[~nans], Y[~nans])

        return Y

    @property
    def current_profile(self):
        """Define the upper limit of the gravity current
        for each horizontal position in the visible region
        of the image.
        """
        x, y = self.current_interface
        # TODO: do this a bit more elegantly
        y += config.top_bar
        ix, iy = self.pixel_coords
        w, h = self.measurement_region.size

        # detect front, if exists
        try:
            front = x.min()
        except ValueError:
            front = w

        X = np.arange(w)
        Y = np.zeros((w,))
        # if ahead of the front, equal bottom of the tank
        Y[:front] = h + config.top_bar

        # there must be a way to vectorize this...
        # if behind the front
        for i in xrange(front, w):
            _y = y[np.where(x == i)]
            # if no point, put a nan placemarker
            if _y.size == 0:
                Y[i] = np.nan
            # otherwise, take highest (lowest y pixel)
            elif _y.size >= 1:
                Y[i] = _y.min()

        # interpolate over gaps (nan) behind the front
        nans = np.isnan(Y)
        # numpy one dimensional interpolation - more in
        # scipy.interpolate
        Y[nans] = np.interp(X[nans], X[~nans], Y[~nans])

        return Y

    @staticmethod
    def canny_interface(array, sigma=5, mask=None):
        """Perform canny edge detection on the given array, using
        gaussian smoothing of sigma, only considering the regions
        defined as True in mask.
        """
        cb = skif.canny(array, sigma=sigma, mask=mask)
        iy, ix = np.where(cb)
        s = ix.argsort()
        return ix[s], iy[s]

    @property
    def has_waves(self):
        # TODO: move to run level
        """True if the run that the image is from contains
        a two layer fluid."""
        return self.parameters['h_1'] != 0.0

    def plot_channels(self):
        """Convenience function to make a figure with the input
        image and the three colour channels."""
        r, g, b = self.channels

        fig, axes = plt.subplots(6, 2, sharex='col', sharey='row')

        axes[0, 0].set_title('Original')
        axes[0, 0].imshow(self.imarray)

        axes[1, 0].set_title('red')
        axes[1, 0].imshow(r)

        axes[2, 0].set_title('green')
        axes[2, 0].imshow(g)

        axes[3, 0].set_title('blue')
        axes[3, 0].imshow(b)

        axes[4, 0].set_title('current fluid (r-g)')
        axes[4, 0].imshow(self.current_fluid)

        axes[4, 1].set_title('masked current fluid (r-g)')
        axes[4, 1].imshow(self.masked_current_fluid)

        axes[5, 0].set_title('r - b')
        axes[5, 0].imshow(r - b)

        w, h = self.im.size
        for ax in fig.axes:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

        fig.tight_layout()

        return fig

    def write_out(self, path=None):
        """Save the processed image to disk."""
        if path is None:
            path = self.outpath
        util.makedirs_p(os.path.dirname(path))
        self.im.save(path)


class LabRun(object):
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

        self.parameters = self.read_parameters(run, self.parameters_f)

        self.run_json_fname = 'config.json'

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
        im_re = 'img_*'
        cam_re = 'cam[0-9]'
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
    def cam1_images(self):
        return (im for im in self.images if im.cam == 'cam1')

    @property
    def cam2_images(self):
        return (im for im in self.images if im.cam == 'cam2')

    @property
    def style(self):
        """Returns a string, the perspective style used for the run."""
        return self.parameters['perspective']

    @staticmethod
    def read_parameters(run, paramf):
        p = read_parameters(paramf)
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

    @staticmethod
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
        # convert to dictionary. tolist() is to convert to native
        # python types
        rdp_dict = dict(zip(rdp.dtype.names, rdp[0].tolist()))
        return rdp_dict


class RawRun(LabRun):
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

        run_data = self.read_run_data(self.index, procf)

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

        cam1_coeff = processing.perspective_coefficients(x1, X1).tolist()
        if lower_right_2[0] == 0:
            cam2_coeff = 0
        else:
            cam2_coeff = processing.perspective_coefficients(x2, X2).tolist()

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

    def process(self, parallel=10):
        """Process the images, transforming them from their raw state.

        Can easily multiprocess this bit.
        """
        kwargs = [{'image': i} for i in self.images]
        if parallel:
            parallel_process(process_raw, kwargs, processors=int(parallel))
        elif not parallel:
            for image in self.images:
                image.write_out()

        self.write_run_json()

    def write_run_json(self):
        """Write out a json file that contains the configuration
        used to process the run."""
        outpath = os.path.join(self.output_dir, self.run_json_fname)

        # pull out top level names from config file.
        config_dict = {k: getattr(config, k)
                       for k in dir(config)
                       if not k.startswith('_')}

        data = {'crop_box': {'cam1': self.crop_box('cam1'),
                             'cam2': self.crop_box('cam2')},
                'perspective_coeffs': self.perspective_coefficients,
                'parameters': self.parameters,
                'run_data': self.run_data,
                'config': config_dict,
                }

        out = open(outpath, 'w')
        out.write(json.dumps(data, indent=1))
        out.close()


class ProcessedRun(LabRun):
    """Same init as RawRun. At some point these two will be merged
    into a single Run class.
    """
    Image = ProcessedImage
    indir = config.outdir
    outdir = config.outdir

    # TODO: if you write a wrapper for images that lets
    # you broadcast a method call across all images,
    # then write a parallel_stub function that will
    # call any general method, you should be able to
    # implement multiprocessing on the image level.
    #
    # relies on util.parallel_progress being able
    # to work with generators in order to work with
    # out blasting too much memory.
    def __init__(self, *args, **kwargs):
        super(ProcessedRun, self).__init__(*args, **kwargs)
        self.config_data = self.read_run_json()
        # create a config object that mimics the config module
        self.config = type('config', (object,), self.config_data['config'])

    def measurement_box(self, cam):
        """The box that contains the image data from the run, i.e.
        the box (left, upper, right, lower) in pixels that would
        remove the black bars from a processed image.
        """
        left, upper, right, lower = self.config_data['crop_box'][cam]
        w = right - left
        h = lower - upper
        measurement_box = (0, self.config.top_bar,
                           w, h - self.config.bottom_bar)
        return measurement_box

    def read_run_json(self):
        """Read the configuration used to process the run."""
        inpath = os.path.join(self.input_dir, self.run_json_fname)
        fin = open(inpath, 'r')
        data = json.loads(fin.read())
        fin.close()
        return data

    def stitcher(self, im1, im2):
        """im1 and im2 are ProcessedImage instances"""
        fname = os.path.basename(im1.path)
        output_dir = self.output_dir
        outpath = os.path.join(output_dir,
                               'stitched',
                               fname)

        # the output size is determined by the outer edges of the
        # camera crop regions
        width = self.config.crop['cam2']['left'] \
                - self.config.crop['cam1']['right']
        # and we'll use the full height of the images
        height = self.config.ideal_25 \
                 + self.config.top_bar \
                 + self.config.bottom_bar
        outsize = (int(width * config.ideal_m), height)

        # you can join the images together anywhere you like as
        # long as is in the overlap between the two cameras
        join = self.config.overlap[self.style]
        # where is the join in cam2?
        join_2 = self.real_to_pixel(join, 0, 'cam2')[0]
        # where is the join in cam1?
        join_1 = self.real_to_pixel(join, 0, 'cam1')[0]

        stitched_image = self.stitch(im1.im, im2.im,
                                     join_1=join_1,
                                     join_2=join_2,
                                     outsize=outsize)

        return self.Image(im=stitched_image, path=outpath, run=self)

    @staticmethod
    def stitch(im1, im2, join_1, join_2, outsize):
        """Stitch two images together, where im1 is taken by cam1
        and im2 is taken by cam2.

        im1 and im2 are PIL Image objects.

        join_i is the horizontal position (pixels) at which to crop
        image i. Ideally these positions will correspond to the same
        physical place.

        outsize is the size (w, h) in pixels of the stitched
        image.

        Returns a PIL Image object.
        """
        # crop the images down
        cim1 = im1.crop((join_1, 0, im1.size[0], im1.size[1]))
        cim2 = im2.crop((0, 0, join_2, im2.size[1]))
        # output boxes for the cropped images to go in
        b2 = (0, 0, join_2, cim2.size[1])
        b1 = (join_2, 0, join_2 + cim1.size[0], cim1.size[1])

        out = Image.new(mode='RGB', size=outsize)
        out.paste(cim1, box=b1)
        out.paste(cim2, box=b2)
        return out

    @property
    def stitched_images(self):
        """Iterator of stitched images formed from joining corresponding
        images in camera streams. Each stitched image is an instance
        of a ProcessedImage.
        """
        cam1_images = (im for im in self.images if im.cam == 'cam1')
        cam2_images = (im for im in self.images if im.cam == 'cam2')
        return imap(self.stitcher, cam1_images, cam2_images)

    def real_to_pixel(self, x, y, cam='cam2'):
        """Convert a real measurement (in metres, relative to the lock
        gate) into a pixel measurement (in pixels, relative to the top
        left image corner, given the camera being used (cam).
        """
        config = self.config
        x_ = (config.crop[cam]['left'] - x) * config.ideal_m
        h = config.crop[cam]['upper'] - config.crop[cam]['lower']
        y_ = config.top_bar + (h - y) * config.ideal_m
        return int(x_), int(y_)

    def pixel_to_real(self, x, y, cam='cam2'):
        """Convert a pixel measurement (in image coordinates, relative
        to the top left corner) into a real measurement (in metres,
        relative to the lock gate base), given the camera being used
        (cam).
        """
        config = self.config
        x_ = (config.crop[cam]['left'] - x / config.ideal_m)
        h = config.crop[cam]['upper'] - config.crop[cam]['lower']
        y_ = (config.top_bar - y) / config.ideal_m + h
        return x_, y_

    def pixel_coords(self, cam):
        """Return the pixel coordinate of each point in the
        measurement region.

        Outputs two arrays: the x coordinates and the y coordinates.
        """
        x0, y0, x1, y1 = self.measurement_box(cam=cam)
        x, y = np.indices(((x1 - x0), (y1 - y0)))
        return x.T, y.T

    def real_coords(self, cam):
        """For each pixel in the measurement region, compute
        the real coordinate.

        Outputs two arrays: the x coordinates and the y coordinates.
        """
        x, y = self.pixel_coords(cam)
        y += config.top_bar
        return self.pixel_to_real(x, y, cam=cam)

    def current_interface_X(self, cam):
        """One dimensional X vector for a run."""
        images = (im for im in self.images if im.cam == cam)
        w, h = images.next().im.size
        x = np.arange(w)
        # TODO: convert to real in images?
        return self.pixel_to_real(x, 0, cam=cam)[0]

    def current_interface_Y(self, cam):
        """Grab all interface data from a run"""
        images = (im for im in self.images if im.cam == cam)
        y = np.vstack(im.current_profile for im in images)
        return self.pixel_to_real(0, y, cam=cam)[1]
        # possible alternate? :
        # interfacer = ProcessedImage()
        # lock_interface = interfacer(im, fluid_type='lock')

    def current_interface_T(self, cam):
        """One dimensional time vector for a run."""
        images = (im for im in self.images if im.cam == cam)
        times = np.fromiter((im.time for im in images),
                            dtype=np.float)
        return times

    def mesh_XT_current(self, cam):
        """Grid x and t"""
        x = self.current_interface_X(cam)
        t = self.current_interface_T(cam)
        return np.meshgrid(x, t)

    @lazyprop
    def combine_current(self):
        """Combine the cameras into a single data array in each of
        X, T and Y.

        Chops off the individual camera arrays at the config defined
        overlap and at the shortest sequence time length.
        """
        x1 = self.current_interface_X('cam1')
        x2 = self.current_interface_X('cam2')

        t1 = self.current_interface_T('cam1')
        t2 = self.current_interface_T('cam2')

        join = self.config.overlap[self.style]
        x1v = x1[x1 < join]
        x2v = x2[x2 > join]

        tmax = min(t1.max(), t2.max())
        t1v = t1[t1 < tmax]
        t2v = t2[t2 < tmax]

        xv = np.hstack((x2v, x1v))
        tv = np.unique(np.hstack((t2v, t1v)))

        Y1 = self.current_interface_Y('cam1')
        Y2 = self.current_interface_Y('cam2')
        Y = np.hstack((Y2[:t2v.size, :x2v.size], Y1[:t1v.size, :x1v.size]))

        return xv[::-1], tv, Y[:, ::-1]

    def wave_interface_X(self, cam):
        """One dimensional X vector for a run."""
        images = (im for im in self.images if im.cam == cam)
        w, h = images.next().im.size
        x = np.arange(w)
        return self.pixel_to_real(x, 0, cam=cam)[0]

    def wave_interface_Y(self, cam):
        """Grab all interface data from a run"""
        images = (im for im in self.images if im.cam == cam)
        y = np.vstack(im.wave_profile for im in images)
        return self.pixel_to_real(0, y, cam=cam)[1]
        # possible alternate? :
        # interfacer = ProcessedImage()
        # lock_interface = interfacer(im, fluid_type='lock')

    def wave_interface_T(self, cam):
        """One dimensional time vector for a run."""
        images = (im for im in self.images if im.cam == cam)
        times = np.fromiter((im.time for im in images),
                            dtype=np.float)
        return times

    def mesh_XT_wave(self, cam):
        """Grid x and t"""
        x = self.wave_interface_X(cam)
        t = self.wave_interface_T(cam)
        return np.meshgrid(x, t)

    @lazyprop
    def combine_wave(self):
        """Combine the cameras into a single data array in each of
        X, T and Y.

        Chops off the individual camera arrays at the config defined
        overlap and at the shortest sequence time length.
        """
        x1 = self.wave_interface_X('cam1')
        x2 = self.wave_interface_X('cam2')

        t1 = self.wave_interface_T('cam1')
        t2 = self.wave_interface_T('cam2')

        join = self.config.overlap[self.style]
        x1v = x1[x1 < join]
        x2v = x2[x2 > join]

        tmax = min(t1.max(), t2.max())
        t1v = t1[t1 < tmax]
        t2v = t2[t2 < tmax]

        xv = np.hstack((x2v, x1v))
        tv = np.unique(np.hstack((t2v, t1v)))

        Y1 = self.wave_interface_Y('cam1')
        Y2 = self.wave_interface_Y('cam2')
        Y = np.hstack((Y2[:t2v.size, :x2v.size], Y1[:t1v.size, :x1v.size]))

        return xv[::-1], tv, Y[:, ::-1]

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


class Run(object):
    def __init__(self, index, autoload=True):
        self.index = index
        self.parameters_f = config.paramf
        self.parameters = LabRun.read_parameters(index, self.parameters_f)

        self.processed = ProcessedRun(index)
        self.raw = RawRun(index)

        self.h5name = os.path.join(self.processed.output_dir,
                                   self.index + '.hdf5')

        if autoload:
            self.load()
            self.load_to_memory()

    def extract(self):
        """Extract data from the processed run."""
        waves = self.processed.combine_wave
        current = self.processed.combine_current
        self.waves = Waves(*waves)
        self.current = Current(*current)

    def save(self, overwrite=False):
        """Save data to run-wide hdf5"""
        if overwrite and os.path.exists(self.h5name):
            os.remove(self.h5name)
        elif not overwrite and os.path.exists(self.h5name):
            raise IOError('file already exists ({})'.format(self.h5name))

        h5 = h5py.File(self.h5name)

        waves = h5.create_group('waves')
        current = h5.create_group('current')

        waves.create_dataset('x', data=self.waves.x)
        waves.create_dataset('t', data=self.waves.t)
        waves.create_dataset('z', data=self.waves.z)

        current.create_dataset('x', data=self.current.x)
        current.create_dataset('t', data=self.current.t)
        current.create_dataset('z', data=self.current.z)

        h5.close()

    def load(self):
        """Load data from hdf5."""
        h5 = h5py.File(self.h5name)

        wx = h5['waves']['x']
        wt = h5['waves']['t']
        wz = h5['waves']['z']

        cx = h5['current']['x']
        ct = h5['current']['t']
        cz = h5['current']['z']

        self.waves = Waves(wx, wt, wz)
        self.current = Current(cx, ct, cz)

    def load_to_memory(self):
        """Load the hdf5 data to memory."""
        self.waves.x = self.waves.x[:]
        self.waves.t = self.waves.t[:]
        self.waves.z = self.waves.z[:]

        self.current.x = self.current.x[:]
        self.current.t = self.current.t[:]
        self.current.z = self.current.z[:]


class InterfaceData(object):
    """Container for data from an interface."""
    def __init__(self, x, t, z):
        self.x = x
        self.t = t
        self.z = z

    def __call__(self, x=None, t=None):
        """Return the value of z at the given x and t."""
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        return self._interpolator(x, t)

    @property
    def _interpolator(self):
        if not hasattr(self, '_cached_interpolator'):
            self._cached_interpolator = interp.interp2d(self.x, self.t, self.z)

        return self._cached_interpolator

    @property
    def X(self):
        """Gridded view of x."""
        if not hasattr(self, '_X'):
            self._X, self._T = np.meshgrid(self.x, self.t)
        return self._X

    @property
    def T(self):
        """Gridded view of t."""
        if not hasattr(self, '_T'):
            self._X, self._T = np.meshgrid(self.x, self.t)
        return self._T


class Waves(InterfaceData):
    """Container for wave data."""
    pass


class Current(InterfaceData):
    """Container for current data."""
    pass

# NOTE: scipy.signal._peak_finding._identify_ridge_lines may be
# useful for wave tracking in the signal data, as well as
# scipy.signal.find_peaks_cwt
