import glob
import os

from gc_turbulence.gc_turbulence.util import *

from labwaves.runbase import RawRun
from labwaves import config


def get_runs():
    runpaths = glob.glob(os.path.join(config.path, 'synced', 'r*'))
    runs = [{'run': os.path.basename(p)} for p in runpaths]
    return runs


@parallel_stub
def barrel_correct(run):
    r = RawRun(run)
    r.bc1()


parallel_process(barrel_correct, get_runs())
