import os
import glob
from itertools import imap

import Image
import ImageDraw
# class to deal with images that have been processed from their
# raw state. deals with interface extraction.

# basically takes get_data and puts it in a class

import config
import util
# module for extracting interfaces from image objects
# TODO: make this
import interface

from raw import *


def real_to_pixel(x, y, cam='cam2'):
    """Convert a real measurement (in metres, relative to the lock
    gate) into a pixel measurement (in pixels, relative to the top
    left image corner, given the camera being used (cam).
    """
    x_ = (config.crop[cam]['left'] - x) * config.ideal_m
    h = config.crop[cam]['upper'] - config.crop[cam]['lower']
    y_ = config.top_bar + (h - y) * config.ideal_m
    return int(x_), int(y_)


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

        self.im = Image.open(path)


class ProcessedRun(object):
    """Same init as RawRun. At some point these two will be merged
    into a single Run class.
    """
    def __init__(self, run, path=None):
        """
        Inputs: run - string, a run index, e.g. 'r11_05_24a'
        """
        self.index = run
        self.config = config
        if not path:
            self.path = config.path
        else:
            self.path = path

        # processed input is the output from raw
        self.input_dir = os.path.join(self.path, config.outdir, self.index)

    @property
    def imagepaths(self):
        """Return a list of the full path to all of the images
        in the run.
        """
        rundir = self.input_dir
        # TODO: put these re in config?
        im_re = 'img*jpg'
        cam_re = 'cam[0-9]'
        im_cam_re = cam_re + '/' + im_re
        imagelist = glob.glob(os.path.join(rundir, im_cam_re))
        return sorted(imagelist)

    @property
    def images(self):
        """Return a list of image objects, corresponding to each image
        in the run.
        """
        return (ProcessedImage(p, run=self) for p in self.imagepaths)

    @property
    def stitched_images(self):
        cam1_images = (im for im in self.images if im.cam == 'cam1')
        cam2_images = (im for im in self.images if im.cam == 'cam2')
        return imap(StitchedImage, cam1_images, cam2_images)

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


class StitchedImage(object):
    def __init__(self, im1, im2):
        """Create a Stitched Image, which is the joining of two
        images from different cameras.

        im1 - cam1 image, ProcessedImage object
        im2 - cam2 image, ProcessedImage object

        Alternately, im1 and im2 can be PIL Image objects.
        """
        self.stitched_im = self.stitch(im1.im, im2.im)

        if type(im1) is ProcessedImage and type(im2) is ProcessedImage:
            output_dir = im1.output_dir
            self.fname = os.path.basename(im1.path)
        else:
            output_dir = './'
            self.fname = 'stitched_image.png'

        self.outpath = os.path.join(output_dir, 'stitched', self.fname)

    @staticmethod
    def stitch(im1, im2):
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
        # this would be the leftmost edge of cam1:
        # join = real_to_pixel(c.crop['cam1']['left'], 0, 'cam2')[0]
        # lets pick the centre of the overlap:
        centre = c.crop['cam2']['right'] + \
            (c.crop['cam1']['left'] - c.crop['cam2']['right']) / 2
        # where is the join in cam2?
        join = real_to_pixel(centre, 0, 'cam2')[0]
        # where is the join in cam1?
        join1 = real_to_pixel(centre, 0, 'cam1')[0]

        # crop the images down
        cim1 = im1.crop((join1, 0, im1.size[0], im1.size[1]))
        cim2 = im2.crop((0, 0, join, im2.size[1]))
        # output boxes for the cropped images to go in
        b2 = (0, 0, join, cim2.size[1])
        b1 = (join, 0, join + cim1.size[0], cim1.size[1])

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
