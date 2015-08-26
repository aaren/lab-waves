"""Tests for processing module"""
from nose.tools import *

import numpy.testing as npt
import numpy as np
from PIL import Image

from labwaves import processing
from labwaves import config

# Testing methods that do image processing is difficult
# because of the necessity of creating images for state


def test_barrel_correct_version():
    """Check that convert is new enough to have barrel
    distortion correction.

    Barrel distortion was introduced in v6.4.2-4
    see http://www.imagemagick.org/Usage/distorts/#barrel.
    """
    # Note that LooseVersion and StrictVersion have been deprecated
    # under PEP 386 and will at some point be replaced by
    # NormalizedVersion
    import subprocess
    from distutils.version import LooseVersion

    minimum_version = '6.4.2-4'

    version_cmd = ['convert', '--version']
    version_info = subprocess.check_output(version_cmd)
    version_number = version_info[2]
    assert(LooseVersion(version_number) > LooseVersion(minimum_version))


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

    Another problem is that different versions of convert seem to
    write BMP headers differently. For example, on Debian Wheezy
    with convert --version = 6.7.7-10 we get the error

    > IOError: Unsupported BMP header type (124)

    On CentOS 5, with convert --version = 6.6...something we don't
    get this error.
    """
    raw_image_1 = 'tests/data/synced/r11_07_06c/cam1/img_0001.jpg'
    raw_image_2 = 'tests/data/synced/r11_07_06c/cam2/img_0001.jpg'
    raw_images = [raw_image_1, raw_image_2]

    bc_image_1 = 'tests/data/bc1/r11_07_06c/cam1/img_0001.jpg'
    bc_image_2 = 'tests/data/bc1/r11_07_06c/cam2/img_0001.jpg'
    bc_images = [bc_image_1, bc_image_2]

    for raw_image, bc_image in zip(raw_images, bc_images):
        raw_im = Image.open(raw_image)
        ref_bc_im = Image.open(bc_image)

        coeffs = config.barrel_coeffs['old']['cam1']
        test_bc_im = processing.barrel_correct(raw_im, coeffs)
        # have to write it to disk for images to be comparable
        test_bc_im.save('/tmp/test_bc_im.jpg')
        test_bc_im_on_disk = Image.open('/tmp/test_bc_im.jpg')

        ref_bc_im_array = np.asarray(ref_bc_im)
        test_bc_im_on_disk_array = np.asarray(test_bc_im_on_disk)

        npt.assert_array_equal(ref_bc_im_array, test_bc_im_on_disk_array)


def test_perspective_coefficients():
    """Mapping a set of points onto itself should give the identity
    matrix (in two dimensions).
    """
    x = ((0, 0), (1, 0), (1, 1), (0, 1))
    c = processing.perspective_coefficients(x, x)
    i = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    npt.assert_array_equal(i, c)


def test_perspective_transform():
    """Using the identity matrix to transform an image
    should make no change, except that it will be resampled.

    We use bilinear resampling in this function.
    """
    im = Image.open('tests/data/synced/r11_07_06c/cam1/img_0001.jpg')

    identity = (1, 0, 0, 0, 1, 0, 0, 0)
    tim = processing.perspective_transform(im, identity)
    tima = np.asarray(tim)

    # spoof a zero degree rotate to get the resampling
    rima = np.asarray(im.rotate(angle=0, resample=Image.BILINEAR))

    npt.assert_array_equal(rima, tima)


def test_crop():
    """Bit redundant really. """
    im = Image.open('tests/data/synced/r11_07_06c/cam1/img_0001.jpg')
    box = (0, 0, im.size[0], im.size[1])
    cima = im.crop(box)

    cropped = processing.crop(im, box)
    cropped_a = np.asarray(cropped)

    npt.assert_array_equal(cima, cropped_a)


def test_draw_text():
    """Test with no text, set bars, no font.

    Integration test will more thoroughly test this.
    """
    im = Image.open('tests/data/synced/r11_07_06c/cam1/img_0001.jpg')
    kwargs = {'upper_text':  '',
              'lower_text':  '',
              'upper_bar':   100,
              'lower_bar':   100,
              'font':        None,
              'text_colour': None,
              'bg_colour':   'black'}
    dim = processing.draw_text(im, **kwargs)

    ima = np.asarray(im)
    dima = np.asarray(dim)

    # check centre of image unchanged
    npt.assert_array_equal(ima[101:-100], dima[101:-100])

    # check bars are black
    assert((dima[0: 100] == 0).all())
    assert((dima[-99:] == 0).all())
