"""Tests for processing module"""
from nose.tools import *

import numpy.testing as npt
import Image

from labwaves.labwaves import processing

# Testing methods that do image processing is difficult
# because of the necessity of creating images for state

def test_barrel_correct_version():
    """Check that convert is new enough to have barrel
    distortion correction."""
    pass


# def test_barrel_correct():
    # raw_image = 'tests/data/r11_07_06c/raw/img_0001.jpg'
    # bc_image = 'tests/data/r11_07_06c/bc/img_0001.jpg'
    # raw_im = Image.open(raw_image)
    # coeffs = config.barrel_coeffs['cam1']
    # ref_bc_im = Image.open(bc_image)
    # test_bc_im = processing.barrel_correct(raw_im, coeffs)
    # ref_bc_im_array = np.asarray(ref_bc_im)
    # test_bc_im_array = np.asarray(test_bc_im)
    # npt.assert_array_equal(ref_bc_im_array, test_bc_im_array)
