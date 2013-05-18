import os
import glob
from util import get_parameters
from util import pull_col
from config import path

cutf = '/home/eeaol/code/lab-waves/scripts/simple_amplitudes'
runs = pull_col(0, cutf, ',')[1:-1]
print runs

sync_dir = path + '/synced/'

def cut(run):
    info = get_parameters(run, cutf, ',')
    cutno = int(info['cut'])
    rundir = sync_dir + run
    for cam in ['cam1', 'cam2']:
        images = glob.glob('/'.join([rundir, cam, '/img*jpg']))
        top_image = sorted(images)[-1].split('/')[-1]
        topno = int(top_image.split('_')[-1].split('.')[0])
        for i in range(cutno, topno + 1):
            image = '/'.join([rundir, cam, 'img_%04d.jpg' % i])
            if os.path.exists(image):
                print "rm %s" % image
                os.remove(image)
            else:
                pass

for run in runs:
    cut(run)

