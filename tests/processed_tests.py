from __future__ import division

from labwaves.runbase import ProcessedRun

from labwaves import config
# PROCESSED RUN TESTS

pr = ProcessedRun('r11_07_06c')


def test_pixel_to_real_cam1():
    top_bar_real = config.top_bar / config.ideal_m
    x, y = pr.pixel_to_real(0, 0, 'cam1')
    assert(x == config.crop['cam1']['left'])
    assert(y == config.crop['cam1']['upper'] + top_bar_real)


def test_pixel_to_real_cam2():
    top_bar_real = config.top_bar / config.ideal_m
    x, y = pr.pixel_to_real(0, 0, 'cam2')
    assert(x == config.crop['cam2']['left'])
    assert(y == config.crop['cam2']['upper'] + top_bar_real)


def test_real_to_pixel_cam1():
    x, y = pr.real_to_pixel(config.crop['cam1']['left'],
                                 config.crop['cam1']['upper'], 'cam1')
    assert(x == 0)
    assert(y == config.top_bar)

    x, y = pr.real_to_pixel(0, 0, 'cam1')
    assert(x == int(config.crop['cam1']['left'] * config.ideal_m))
