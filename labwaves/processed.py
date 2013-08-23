import os
import glob

import Image
# class to deal with images that have been processed from their
# raw state. deals with interface extraction.

# basically takes get_data and puts it in a class

import config
# module for extracting interfaces from image objects
# TODO: make this
import interface

class ProcessedRun(object):
    """Same init as RawRun. At some point these two will be merged
    into a single Run class.
    """
    def __init__(self, run):
        """
        Inputs: run - string, a run index, e.g. 'r11_05_24a'
        """
        self.index = run
        self.config = config

        # processed input is the output from raw
        self.input_dir = os.path.join(config.path, config.outdir, self.index)

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
        return [Image.open(p) for p in self.imagepaths]

    def interface(self):
        """Grab all interface data from a run"""
        # TODO: multiprocessing
        # TODO: runfiles should point to processed data
        for im in self.images:
            interfaces = interface.interface(im)
            qc_interfaces = [interface.qc(i) for i in interfaces]
            save_interface(qc_interfaces)
