#!/apps/enthought-7.2-1/bin/python
from multiprocessing import Process, Pool, Queue
import multiprocessing.pool
import glob
from sys import argv

import proc_im
import join
import get_data
import waves

from config import path

def measure(run):
    """Before the processing in proc_im_main(run) can take
    place, some basic measurements have to be made for each run.
    """
    # Barrel correct the first frame of each camera in a run
    # outputs to dir 'bc1' in the path
    # proc_im.bc1(run)
    # proc_im.barrel_corrections(run)

    proc_im.measure(run)
    # Prompt user for measurements of run data, if they don't
    # exist already.
    # proc_im.get_run_data(run)

def proc_im_base(run):
    # Correct barrel distortion and rotation. Will prompt for run
    # measurements if they do not exist.
    proc_im.std_corrections(run)

def proc_im_main(run):
    """Raw lab images need some massaging to get them into
    a standard format that can be processed later. The pre-
    requisite here is that the images are synchronised and
    called 'img_0001.jpg' onwards, where the first image is
    at t=0 in the run (i.e. as close to the lock release as
    possible).

    Barrel correction is tested for ImageMagick 6.7.3-9
    """

    # Add text and crop the image. Will prompt for run measurements
    # if they do not exist.
    proc_im.std_corrections(run)
    proc_im.text_crop(run)

    # Create some joined up images in presentation and an
    # animated gif.
    # FIXME: Parameter display when there is only one image.
    join.presentation(run)
    join.animate(run)

def basic_data(run):
    """If the run data has been through proc_im_main, or the
    stages of it, it is ready to have the basic data extracted
    from it.

    The basic data consists of a series of lists of interface
    depths for a number of interfaces found in the fluid. These
    are extracted using the threshold module and then saved
    (pickled) to a file as a dictionary.

    This stage is separated out because it has a computationally
    intensive inner loop, where an entire image is thresholded.
    """
    get_data.get_basic_data(run)

def f_basic_data(run):
    """Fast version of basic data (multiprocessing inside the
    thresholding loop).
    """
    get_data.get_basic_data(run, 0)

def data(run):
    """The basic interface depths are further processed to
    pull out maxima / minima using peakdetect. Parallax
    correction is applied here, as well as conversion of the
    data units from pixels in to tank relative numbers (units
    of fluid depth, lock length).

    At this point some 'sanity' images are written out with
    some of the measured data superimposed.

    The data extracted here are saved to another file.
    """
    get_data.main(run)

def wave(run):
    """Before this stage the run data are separated into cam1
    and cam2 streams. This stage combines the data together.

    The coherent structures in the data (waves, front progression)
    are visible to the eye but not to the machine. Structures
    are tracked and output as distinct objects to make higher
    level data analysis easier.
    """
    pass

def plot(run):
    """With the structures pulled out by wave(run), it is
    easy to make some plots.
    """
    waves.main(run)

def all(run):
    """To get some raw, synced, lab data into nice plots in
    a single command.
    """
    proc_im_base(run)
    proc_im_main(run)
    f_basic_data(run)
    data(run)
    plot(run)

def multi(proc, runs):
    # not presently used.
    q = Queue()
    for run in runs:
        q.put(run)

    ps = [Process(target=proc, args=(run,)) for i in range(12)]
    for p in ps:
        p.start()
        print "started %s" % p
    for p in ps:
        p.join()

## see http://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make the daemon attribute false
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def pool(proc, runs):
    p = MyPool()
    p.map(proc, runs)
    p.close()
    p.join()

def loop(proc, runs):
    for run in runs:
        proc(run)

def get_runs(pdir='synced'):
    runpaths = glob.glob(('/').join([path, pdir, 'r*']))
    runs = sorted([runpath.split('/')[-1] for runpath in runpaths])
    # for r in ['r11_7_07e', 'r11_5_25c', 'r11_6_30a', 'r11_6_30b', 'r11_7_08a', 'r11_5_24c']:
        # runs.remove(r)
    return runs

def test():
    print "hello"

if __name__ == '__main__':
    try:
        if argv[2] == 'all':
            runs = get_runs()
        elif 'r11_' in argv[2]:
            runs = argv[2:]
        else:
            print "Supply runs to process. 'all' is valid."
            exit(0)
    except IndexError:
        print "Supply runs to process. 'all' is valid."
        exit(0)

    try:
        process = globals().get(argv[1])
    except IndexError:
        print "Must supply a process to run."
        exit(0)
    except AttributeError:
        print argv[1], "isn't a process."
        exit(0)

    if argv[-1] == 'loop':
        runs.pop(-1)
        print "looping..."
        loop(process, runs)
    elif argv[-1] == 'pool':
        runs.pop(-1)
        print "multiprocessing"
        pool(process, runs)
    elif argv[-1] == 'fast':
        processors = 12
        pool(process, runs)
    elif argv[-1] == 'all':
        pool(process, runs)
    elif 'r11_' in argv[-1] and len(argv) == 3:
        print "straight processing..."
        process(argv[-1])
    elif 'r11_' in argv[-1]:
        print "multiprocessing..."
        pool(process, runs)
    else:
        print "I'm sorry, Aaron. I'm afraid I can't do that."
        exit('goodbye')
