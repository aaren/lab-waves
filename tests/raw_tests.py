"""Tests for RawRun class"""
from nose.tools import *

from labwaves.raw import RawRun
from labwaves.raw import read_parameters
from labwaves.raw import read_run_data

import labwaves.processing

# location of test data files
proc_f = 'tests/data/proc_data'
param_f = 'tests/data/parameters'


class TestRun(object):
    """An idealised test object"""
    def __init__(self):
        """Uses the run r11_07_06c for testing. This run is a
        good example of a perfect run, so be careful using it
        to justify edge cases.
        """
        self.index = 'r11_07_06c'
        self.parameters = {'run_index': 'r11_07_06c',
                           'h_1':       0.25,
                           'rho_0':     1.150,
                           'rho_1':     1.100,
                           'rho_2':     1.000,
                           'alpha':     0.5,
                           'D':         0.4}
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
        self.raw_image = 'tests/data/r11_07_06c/raw/img_0001.jpg'
        self.bc_image = 'tests/data/r11_07_06c/bc/img_0001.jpg'

# initialise test object
t = TestRun()


def test_read_parameters():
    params = read_parameters(t.index, param_f)
    assert_equal(type(params), dict)
    assert_equal(params, t.parameters)


def test_read_run_data():
    run_data = read_run_data(t.index, proc_f)
    assert_equal(type(run_data), dict)
    assert_equal(run_data, t.run_data)


r = RawRun(t.index, parameters_f=param_f, run_data_f=proc_f)


def test_init():
    assert_equal(r.index, t.index)
    for k in r.parameters:
        assert_equal(r.parameters[k], t.parameters[k])


def test_gen_image_text():
    time = 1
    text = r.gen_image_text(time)
    t_text = ("run {run_index}, "
              "t = {time}s: "
              "h_1 = {h_1}, "
              "rho_0 = {rho_0}, "
              "rho_1 = {rho_1}, "
              "rho_2 = {rho_2}, "
              "alpha = {alpha}, "
              "D = {D}").format(time=1, **r.parameters)
    assert_equal(type(text), str)
    assert_equal(text, t_text)
