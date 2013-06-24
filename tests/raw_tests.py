"""Tests for RawRun class"""
from nose.tools import *

import os
import glob

import Image
import numpy as np
import numpy.testing as npt

from labwaves.runbase import RawRun
from labwaves.runbase import RawImage
from labwaves.runbase import read_parameters
from labwaves.runbase import read_run_data

from labwaves import config

# location of test data files
proc_f = 'tests/data/proc_data'
param_f = 'tests/data/parameters'
test_path = 'tests/data/'

# clear any previous output
for root, dirs, files in os.walk('tests/data/tmp', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
for root, dirs, files in os.walk('tests/data/processed', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))


def assert_image_equal(path1, path2):
    """Assert equality of two images, paths given as arguments."""
    test_im = np.asarray(Image.open(path1))
    ref_im = np.asarray(Image.open(path2))
    npt.assert_array_equal(test_im, ref_im)


class TestRun(object):
    """An idealised test object"""
    def __init__(self):
        """Uses the run r11_07_06c for testing. This run is a
        good example of a perfect run, so be careful using it
        to justify edge cases.
        """
        self.index = 'r11_07_06c'
        self.parameters = {'run_index': 'r11_07_06c',
                           'h_1':             0.25,
                           'rho_0':           1.150,
                           'rho_1':           1.100,
                           'rho_2':           1.000,
                           'alpha':           0.5,
                           'D':               0.4,
                           'sample':          1.0,
                           'perspective':     'old'}
        self.run_data = {'run_index':  'r11_07_06c',
                         'l0x':        2796,
                         'l0y':        1151,
                         'lsx':        2793,
                         'lsy':        716,
                         'j10x':       210,
                         'j10y':       1165,
                         'j1sx':       208,
                         'j1sy':       727,
                         'leakage':    -76,
                         'odd_1':      'n',
                         'j20x':       2728,
                         'j20y':       1086,
                         'j2sx':       2730,
                         'j2sy':       670,
                         'r0x':        1097,
                         'r0y':        1095,
                         'rsx':        1093,
                         'rsy':        683,
                         'odd_2':      'n'}
        self.raw_image = 'tests/data/synced/r11_07_06c/cam1/img_0001.jpg'
        self.bc_image = 'tests/data/bc/r11_07_06c/cam1/img_0001.jpg'
        self.processed_path = 'tests/data/processed_ref/r11_07_06c/cam1/img_0001.jpg'


t = TestRun()


def test_read_parameters():
    params = read_parameters(t.index, param_f)
    assert_equal(type(params), dict)
    assert_dict_equal(params, t.parameters)


def test_read_run_data():
    run_data = read_run_data(t.index, proc_f)
    assert_equal(type(run_data), dict)
    assert_equal(run_data, t.run_data)


r = RawRun(t.index, parameters_f=param_f, run_data_f=proc_f, path=test_path)


def test_RawRun_init():
    assert_equal(r.index, t.index)
    for k in r.parameters:
        assert_equal(r.parameters[k], t.parameters[k])


def test_RawRun_path():
    assert(os.path.samefile(r.path, test_path))


def test_RawRun_get_run_data_when_exists():
    assert_equal(r.run_data, t.run_data)


def test_RawRun_imagepaths():
    """Expect a list of paths of all images in run."""
    p1 = r.imagepaths[0]
    path = 'tests/data/synced/r11_07_06c/cam1/img_0001.jpg'
    assert(os.path.samefile(p1, path))
    assert_equal(len(r.imagepaths), 14)


def test_RawRun_images():
    path = 'tests/data/synced/r11_07_06c/cam1/img_0001.jpg'
    im0 = r.images.next()
    assert(os.path.samefile(im0.path, path))
    assert_equal(im0.cam, 'cam1')


def test_RawRun_bc1():
    path1 = 'tests/data/tmp/bc1/r11_07_06c/cam1/img_0001.jpg'
    path2 = 'tests/data/tmp/bc1/r11_07_06c/cam2/img_0001.jpg'
    r.bc1()
    # just check that there is output for now
    assert(os.path.exists(path1))
    assert(os.path.exists(path2))


def test_RawRun_perspective_reference_old_style():
    """The old style for cam1, with a reference of (0, 0)."""
    ref = (0, 0)
    m = config.ideal_m
    grid = r.perspective_reference(ref, 'old', 'cam1')
    lower_right, upper_right, lower_left, upper_left = grid
    assert_equal((0, 0), lower_right)
    assert_equal((ref[0] - int(1.47 * m), ref[1] - int(0.25 * m)), upper_left)


# runs with different perspective styles
test_runs = {'old':   r,
             'new_1': RawRun('r13_01_08a',
                             parameters_f=param_f,
                             run_data_f=proc_f,
                             path=test_path),
             'new_2': RawRun('r13_01_11e',
                             parameters_f=param_f,
                             run_data_f=proc_f,
                             path=test_path)}


def test_RawRun_perspective_reference():
    """make a new_2 style run and check sanity of perspective
    reference.
    """
    ref = (0, 0)
    m = config.ideal_m

    def grid(run, style, cam):
        return run.perspective_reference(ref, style, cam)

    upper_left_ref = {'old':   (ref[0] - int(1.47 * m),
                                ref[1] - int(0.25 * m)),
                      'new_1': (ref[0] - int(1.606 * m),
                                ref[1] - int(0.25 * m)),
                      'new_2': (ref[0] - int(1.606 * m),
                                ref[1] - int(0.25 * m))}

    for style in test_runs:
        run = test_runs[style]
        lower_right, upper_right, lower_left, upper_left = grid(run, style,
                                                                'cam1')
        assert_equal(upper_left_ref[style], upper_left)


def test_RawRun_styles():
    for style in test_runs:
        assert_equal(test_runs[style].style, style)


def test_RawRun_perspective_coefficients():
    coeff = r.perspective_coefficients
    # reference coefficients for r11_07_06c
    ref_coeff = {'cam1': (2.2314511087582045,
                          0.0096590503446602471,
                          -3414.5423515871221,
                          -0.0052021223956432985,
                          2.2044684499188145,
                          -1355.4500168457409,
                          5.8984779324655127e-06,
                          -1.9885426533893206e-06),
                 'cam2': (2.0197781850715937,
                          0.039832235669042655,
                          -2860.2502250609414,
                          -0.024309498031448946,
                          2.0654835373252758,
                          -1104.7470351277707,
                          -1.1955993103782927e-05,
                          1.8206519773594392e-05)}

    assert_dict_equal(coeff, ref_coeff)


def test_perspective_transform():
    """Do a perspective transform with all types of run and check
    the output. """
    # TODO: write this
    assert(True)


def test_RawRun_process():
    """Integration tests the whole run."""
    for style in test_runs:
        test_runs[style].process()
    # now compare all images with ref
    ref = glob.glob('tests/data/processed_ref/*/*/*')
    outputs = glob.glob('tests/data/processed/*/*/*')
    for ref, out in zip(ref, outputs):
        assert_image_equal(ref, out)

### RawImage class tests
i = RawImage(t.raw_image, r)


def test_RawImage_param_text():
    text = i.param_text
    t_text = ("run {run_index}, "
              "t = {time}s, "
              "h_1 = {h_1}, "
              "rho_0 = {rho_0}, "
              "rho_1 = {rho_1}, "
              "rho_2 = {rho_2}, "
              "alpha = {alpha}, "
              "D = {D}").format(time=0.0, **r.parameters)
    assert_equal(type(text), str)
    assert_equal(text, t_text)


def test_RawImage_time():
    assert_equal(i.time, 0.0)


def test_RawImage_write_out():
    """Integration tests the whole RawImage class."""
    i.write_out()
    # now compare the output with reference
    assert_image_equal(i.outpath, t.processed_path)
