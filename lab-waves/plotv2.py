from __future__ import division
import matplotlib.pyplot as plt

from get_data import read_data

def plot(run='r11_7_06c', data_storage_file=None):
    # this takes the data container and produces a single plot
    # with the wave and current trajectories plotted.
    # do this in a separate module at some point.
    if run.split('r')[0] == run:
        run = 'r' + run
    if data_storage_file is None:
        data_storage_file = 'data/data_store_' + run

    # read the data to memory
    data = read_data(data_storage_file)
    
    xm = {}
    xn = {}
    f = {}
    t = {}
    # some offsets for changing the meshing of the two camera data
    # streams. i thought these images had been synced but it is
    # possible i made errors at points or one big consistent error.
    off = {}
    off['cam1'] = 0
    off['cam2'] = 2

    plt.figure()
    plt.grid(True)

    plt.xlabel('Distance along tank (lock lengths)')
    plt.xlim(0,13)

    plt.ylabel('Time elapsed (s)')
    T = len(data[run]['cam1'].keys())
    plt.ylim(0,T)

    for cam in ['cam1', 'cam2']:
        cam_data = data[run][cam]
        frames = sorted(cam_data.keys())

        # extract the data
        xm[cam] = [[p[0] for p in cam_data[frame]['max']] for frame in frames]
        xn[cam] = [[p[0] for p in cam_data[frame]['min']] for frame in frames]
        f[cam] = [[p[0] for p in cam_data[frame]['front']] for frame in frames]
        t[cam] = range(off[cam], len(cam_data.keys()) + off[cam])

        # sort out empty cells
        for i in xn[cam]:
            if len(i) == 0:
                i.append(None)
        for i in xm[cam]:
            if len(i) == 0:
                i.append(None)

        plt.plot(f[cam], t[cam], 'ko')
        for i in t['cam1']:
            plt.plot([xm[cam][i]], [t[cam][i]], 'ro')
    # TODO plot a single wave (e.g. the first one).
    # this seems difficult.
    # assume waves follow a straight line in x,t. at a particular time
    # there are only n waves detected. beginning with 1 and adding one
    # at some later time.
    # so in the list of maxima, the first entry corresponds to the first
    # wave until the first wave drops off the end of the fov. if we take
    # the first few points from the wave and extrapolate a straight line,
    # then put some variability on that line, we can create a region in
    # which points for that wave are allowed to fall. then if points don't
    # fall in this, they belong to another wave or are noise.
    # 
    # ideally, create a series of lists, one for each wave, consisting of 
    # tuples designating points in (x,t).

    # filename = 'plots/' + run + '.png'
    # plt.savefig(filename, orientation='landscape')
    plt.show() 
