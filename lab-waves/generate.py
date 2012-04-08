import proc_im
import join
import get_data


def proc_im_main(run):
    """Raw lab images need some massaging to get them into
    a standard format that can be processed later. The pre-
    requisite here is that the images are synchronised and
    called 'img_0001.jpg' onwards, where the first image is
    at t=0 in the run (i.e. as close to the lock release as
    possible).

    Barrel correction is tested for ImageMagick 6.7.3-9
    """
    # Barrel correct the first frame of each camera in a run
    # outputs to dir 'bc1' in the path
    proc_im.bc1(run)
    
    # Correct barrel distortion and rotation. Will prompt for run
    # measurements if they do not exist.
    proc_im.std_corrections(run)

    # Add text and crop the image. Will prompt for run measurements
    # if they do not exist.
    proc_im.text_crop(run)

    # Create some joined up images in presentation and an
    # animated gif.
    join.presentation(run, 'processed') 
    join.animate(run, 'processed')

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
    get_data.get_basic_run_data(run)

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

def plot(run):
    """With the structures pulled out by wave(run), it is 
    easy to make some plots.
    """

def all(run):
    """To get some raw, synced, lab data into nice plots in
    a single command.
    """
    proc_im_main(run)    
    basic_data(run)
    data(run)
    wave(run)
    plot(run)

