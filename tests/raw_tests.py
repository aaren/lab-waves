"""Tests for RawRun class"""
from nose.tools import *

from labwaves.raw import RawRun
from labwaves.raw import read_parameters

ex_index = 'r11_07_06c'
param_r11_07_06c = {'run_index': 'r11_07_06c',
                    'h1':  0.25,
                    'rho_0': 1.150,
                    'rho_1': 1.100,
                    'rho_2': 1.000,
                    'alpha': 0.5,
                    'D':   0.4}
proc_f = 'tests/data/proc_data'
parameters = 'tests/data/parameters'

def test_read_parameters():
    params = read_parameters(ex_index, parameters)
    assert_equal(params, param_r11_07_06c)
    assert_equal(type(params), dict)

def test_init():
    r = RawRun(ex_index, parameters_f=parameters, run_data_f=proc_f)
    assert_equal(r.index, ex_index)
    for k in param_r11_07_06c:
        assert_equal(r.parameters[k], param_r11_07_06c[k])
