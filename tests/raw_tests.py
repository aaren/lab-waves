"""Tests for RawRun class"""
from nose.tools import *

from labwaves.raw import RawRun

ex_index = 'r11_7_06c'
proc_f = 'tests/data/proc_data'
parameters = 'tests/data/parameters'

def test_init():
    r = RawRun(ex_index, parameters_f=parameters, run_data_f=proc_f)
    assert_equal(r.index, ex_index)

