"""Tests for processing module"""
from nose.tools import *

import numpy.testing as npt
import numpy as np
import Image

from labwaves import processing
from labwaves import config

# Testing methods that do image processing is difficult
# because of the necessity of creating images for state


def test_barrel_correct_version():
    """Check that convert is new enough to have barrel
    distortion correction."""
    pass


def test_barrel_correct():
    """Wow this is a ball ache to test. The content of the image
    changes subtley depending on whether the temporary image created
    for convert on the command line is bmp or jpeg.

    If we use bmp and have a bmp reference image, all is well.
    However using a jpg temp and a jpg ref leads to small
    differences.

    Difference appears to be in PIL saving the tmp file. The array
    of this is different from the raw image array when compared.
    Appears to be due to the quantization table for saving a jpeg.
    Not something that can be set with PIL. Hence we have to write
    out the file for it to be comparable.
    """
    raw_image_1 = 'tests/data/raw/r11_07_06c/cam1/img_0001.jpg'
    raw_image_2 = 'tests/data/raw/r11_07_06c/cam2/img_0001.jpg'
    raw_images = [raw_image_1, raw_image_2]

    bc_image_1 = 'tests/data/bc1/r11_07_06c/cam1/img_0001.jpg'
    bc_image_2 = 'tests/data/bc1/r11_07_06c/cam2/img_0001.jpg'
    bc_images = [bc_image_1, bc_image_2]

    for raw_image, bc_image in zip(raw_images, bc_images):
        raw_im = Image.open(raw_image)
        ref_bc_im = Image.open(bc_image)

        coeffs = config.barrel_coeffs['cam1']
        test_bc_im = processing.barrel_correct(raw_im, coeffs)
        # have to write it to disk for images to be comparable
        test_bc_im.save('/tmp/test_bc_im.jpg')
        test_bc_im_on_disk = Image.open('/tmp/test_bc_im.jpg')

        ref_bc_im_array = np.asarray(ref_bc_im)
        test_bc_im_on_disk_array = np.asarray(test_bc_im_on_disk)

        npt.assert_array_equal(ref_bc_im_array, test_bc_im_on_disk_array)
