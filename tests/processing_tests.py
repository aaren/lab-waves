"""Tests for processing module"""
from nose.tools import *

import numpy.testing as npt
import numpy as np
import Image

from labwaves.labwaves import processing
from labwaves.labwaves import config

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

    Do we actually need to test this function???

    A simple way would be to run it once and verify the output, then
    use this as baseline for future tests. This doesn't act as an
    independent test of the function but it does act as a sanity
    check.

    TODO: force the output of barrel_correct to be of type jpg

    Difference appears to be in PIL saving the tmp file. The array
    of this is different from the raw image array when compared.
    Something to do with jpeg compression?? The raw file wasn't
    created by PIL.

    Ughh. Appears to be due to the quantization table for saving a
    jpeg. Not something that can be set with PIL.
    """
    raw_image = 'tests/data/raw/r11_07_06c/cam1/img_0001.jpg'
    bc_image = 'tests/data/bc1/r11_07_06c/cam1/img_0001.jpg'
    raw_im = Image.open(raw_image)
    coeffs = config.barrel_coeffs['cam1']
    ref_bc_im = Image.open(bc_image)
    test_bc_im = processing.barrel_correct(raw_im, coeffs)
    ref_bc_im_array = np.asarray(ref_bc_im)
    test_bc_im_array = np.asarray(test_bc_im)
    npt.assert_array_equal(ref_bc_im_array, test_bc_im_array)
