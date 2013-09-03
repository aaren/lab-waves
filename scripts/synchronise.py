import glob
import os

import multiprocessing as mp

from labwaves.runbase import read_parameters

import vid_sync

paramf = 'tests/data/parameters'


def sample(run):
    """Determine sample rate of a run."""
    index = os.path.basename(run)
    try:
        par = read_parameters(index, paramf)
        return par['sample']
    except KeyError:
        return 0


def sync(run):
    wd = os.path.join(raw_data_dir, run)
    c1 = os.path.join(wd, f1)
    c2 = os.path.join(wd, f2)
    out = vid_sync.synchronise(c1, c2, verbose=True)
    return run, out


raw_data_dir = '/home/eeaol/lab/data/flume1/RAW'

f1 = 'cam1.MOV'
f2 = 'cam2.MOV'

runs = glob.glob(os.path.join(raw_data_dir, 'r*'))

runs_25Hz = [r for r in runs if sample(r) == 0.04]

pool = mp.Pool()
times = pool.map(sync, runs_25Hz)

for time in times:
    print time
