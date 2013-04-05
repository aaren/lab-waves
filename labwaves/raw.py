from __future__ import division

import glob
import os

import Image
import ImageFont
import numpy as np
import matplotlib.pyplot as plt

import config

import processing

import aolcore


def read_parameters(run, paramf):
    """Read in data from the parameters file, which has
    the headers and data types as in the list data.
    'S10' is string, 'f4' is float.

    Reading in gives a numpy recarray. Convert to a dictionary
    and return this.
    """
    data = [('run_index',  'S10'),
            ('h_1',        'f8'),
            ('rho_0',      'f8'),
            ('rho_1',      'f8'),
            ('rho_2',      'f8'),
            ('alpha',      'f8'),
            ('D',          'f8')]
    names, types = zip(*data)
    p = np.genfromtxt(paramf, dtype=types, names=names)
    index = np.where(p['run_index'] == run)
    params = p[index]
    if len(params) == 0:
        # run is not in parameters file
        print("{run} is not in {paramf}".format(run=run, paramf=paramf))
        raise KeyError
    else:
        # convert to dictionary
        names = params.dtype.names
        p_dict = dict(zip(names, params[0]))
        return p_dict


def read_run_data(run, paramf):
    """Read in data from proc file, which has headers and data
    types as in the list data. 'S10' is string, 'i4' is integer.

    Read in as numpy recarray. Convert this to a dictionary for output.
    """
    data = [('run_index',  'S10'),
            ('l0x',        'i4'),
            ('l0y',        'i4'),
            ('lsx',        'i4'),
            ('lsy',        'i4'),
            ('j10x',       'i4'),
            ('j10y',       'i4'),
            ('j1sx',       'i4'),
            ('j1sy',       'i4'),
            ('leakage',    'i4'),
            ('odd_1',      'S10'),
            ('j20x',       'i4'),
            ('j20y',       'i4'),
            ('j2sx',       'i4'),
            ('j2sy',       'i4'),
            ('r0x',        'i4'),
            ('r0y',        'i4'),
            ('rsx',        'i4'),
            ('rsy',        'i4'),
            ('odd_2',      'S10')]
    names, dtypes = zip(*data)
    rd = np.genfromtxt(paramf, skip_header=1,
                       dtype=dtypes,
                       delimiter=',',
                       names=names)
    index = np.where(rd['run_index'] == run)
    rdp = rd[index]
    # convert to dictionary
    rdp_dict = dict(zip(rdp.dtype.names, rdp[0]))
    return rdp_dict


class RawRun(object):
    def __init__(self, run, parameters_f=None, run_data_f=None):
        """

        Inputs: run - string, a run index, e.g. 'r11_05_24a'
                parameters_f - optional, a file containing run parameters
                run_data_f - optional, a file containing run_data
        """
        self.index = run
        if not parameters_f:
            self.parameters = read_parameters(run, config.paramf)
        else:
            self.parameters = read_parameters(run, parameters_f)
        if not run_data_f:
            self.run_data = self.get_run_data()
        else:
            self.run_data = self.get_run_data(procf=run_data_f)

    def get_run_data(self, procf=config.procf):
        proc_runs = aolcore.pull_col(0, procf, ',')
        try:
            proc_runs.index(self.index)
            # print "%s is in proc_data" % run
        except ValueError:
            print "%s is not in the procf (%s)" % (self.index, procf)
            print "get the proc_data for this run now? (y/n)"
            A = raw_input('> ')
            if A == 'y':
                self.measure(self.index)
                self.get_run_data(self.index)
            elif A == 'n':
                return 0
            else:
                print "y or n!"
                self.get_run_data(self.index)

        run_data = read_run_data(self.index, procf)

        return run_data

    def measure(self, procf=config.procf):
        #TODO: doc
        # TODO: make sense in class
        proc = []
        proc.append(self.index)
        for camera in ['cam1', 'cam2']:
            plt.figure(figsize=(16, 12))
            # TODO: fix this path
            simg1 = '{path}/barrel_corr/{run}/{cam}/img_0001.jpg'
            img1 = simg1.format(path=config.path, run=self.index, cam=camera)
            try:
                im = Image.open(img1)
            except IOError:
                print "Bad image for %s %s" % (self.index, camera)
                break
            plt.imshow(im)
            plt.xlim(2500, 3000)
            plt.ylim(750, 1500)
            plt.draw()
            print "Select lock base and surface"
            pt1 = plt.ginput(3, 0)
            if camera == 'cam1':
                plt.xlim(0, 500)
            elif camera == 'cam2':
                plt.xlim(750, 1250)
            plt.draw()
            print "Select join base and surface"
            pt2 = plt.ginput(3, 0)

            pts = pt1[0:2] + pt2[0:2]
            for x, y in pts:
                proc.append(int(x))
                proc.append(im.size[1] - int(y))

            plt.xlim(0, 3000)
            plt.draw()
            if camera == 'cam1':
                print "What is the extent of lock leakage?"
                leak = plt.ginput(2, 0)[0][0]
                proc.append(int(pt1[0][0] - leak))
            print "Weird (y/n)"
            proc.append(raw_input('> '))
            plt.close()

        proc = [str(e) for e in proc]
        entry = ','.join(proc) + '\n'
        f = open(procf, 'a')
        f.write(entry)
        f.close()

    def bc1(self):
        """Just barrel correct the first image of a run
        and save it to the folder 'bc1' under path.
        """
        # TODO: fix synced
        indir = '%s/synced/%s' % (config.path, self.index)
        outdir = 'bc1'
        for camera in ['cam1', 'cam2']:
            dirs = '/'.join([config.path, outdir, self.index, camera])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
                print "made " + dirs
            else:
                # print "using " + dirs
                pass
            image1 = '%s/%s/img_0001.jpg' % (indir, camera)
            im1 = Image.open(image1)
            coeffs = config.barrel_coeffs[camera]
            bim1 = processing.barrel_correct(im1, coeffs)
            outdir = '{path}/{out}/{run}/{cam}/'
            outf = outdir.format(path=config.path,
                                 out=outdir,
                                 run=self.index,
                                 cam=camera) + 'img_0001.jpg'
            bim1.save(outf)

    @property
    def runfiles(self):
        """Build a list of the files from a run, along with the camera
        that they come from.

        Returns a list of tuples
        e.g. [('cam1', 'path/to/image1'), ('cam1', 'path/to/image2'), ...]
        """
        impaths = self.imagepaths()
        cams = [self.icam(impath) for impath in impaths]
        return zip(cams, impaths)

    def iframe(impath):
        """From an image filename, e.g. img_0001.jpg, get just the
        0001 bit and return it.

        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        frame = impath.split('_')[-1].split('.')[0].split('_')[-1]
        return frame

    def icam(impath):
        """Given a path to an image, extract the corresponding camera.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        cam = impath.split('/')[-2]
        return cam

    def irun(impath):
        """Given a path to an image, extract the corresponding run.
        Expects impath to be of form 'path/to/run/cam/img_0001.jpg'
        """
        run = impath.split('/')[-3]
        return run

    def gen_time(self, image):
        # XXX: stub
        # 'time' that a frame represents
        # TODO: update this with 25hz camera in mind. need some time
        # generating function
        run, cam, frame = image.split('/')[-3:]
        time = int(frame.split('.')[0]) - 1
        return time

    @property
    def imagepaths(self):
        """Return a list of the full path to all of the images
        in the run.
        """
        # TODO: put synced in config or something
        # TODO: composition of path names??
        # TODO: put these re in config?
        rundir = config.path + 'synced' + self.index
        im_re = 'img*jpg'
        cam_re = 'cam*'
        im_cam_re = cam_re + '/' + im_re
        imagelist = glob.glob(rundir + im_cam_re)
        return imagelist

    ### Applicators
    # Only a function of the run, generate all context based on this.
    # These do actual, persistent IO
    # Should go into a class really
    # multiprocessing starts to come in here too
    def barrel_correct(self):
        for camera, image in self.runfiles:
            coeffs = config.barrel_coeffs[camera]
            im = Image.open(image)
            out = processing.barrel_correct(im, coeffs)
            out.save('blah')

    def perspective_coefficients(self):
        """Generate the cam1 and cam2 perspective transform coefficients
        for a given run.

        Inputs: run - string, the run index

        Outputs: dictionary of the camera coefficients
                d.keys() = ['cam1', 'cam2']
                d['cam1'] = (a, b, c, d, e, f, g, h)
        """
        run_data = self.run_data

        lock_0 = int(run_data['l0x']), int(run_data['l0y'])
        lock_surf = int(run_data['lsx']), int(run_data['lsy'])
        join1_0 = int(run_data['j10x']), int(run_data['j10y'])
        join1_surf = int(run_data['j1sx']), int(run_data['j1sy'])

        join2_0 = int(run_data['j20x']), int(run_data['j20y'])
        join2_surf = int(run_data['j2sx']), int(run_data['j2sy'])
        ruler_0 = int(run_data['r0x']), int(run_data['r0y'])
        ruler_surf = int(run_data['rsx']), int(run_data['rsy'])
        # need some standard vertical lines in both cameras.
        # cam1: use lock gate and tank join
        # cam2: tank join and ruler at 2.5m
        # (checked to be vertical, extrapolate to surface)
        # so for each camera, 4 locations (8 numbers) need
        # to be recorded.

        x1 = (lock_0, lock_surf, join1_0, join1_surf)
        X1 = (lock_0,
              (lock_0[0], lock_0[1] - config.ideal_25),
              (lock_0[0] - config.ideal_base_1, lock_0[1]),
              (lock_0[0] - config.ideal_base_1, lock_0[1] - config.ideal_25))

        x2 = (join2_0, join2_surf, ruler_0, ruler_surf)
        X2 = (join2_0,
              (join2_0[0], join2_0[1] - config.ideal_25),
              (join2_0[0] - config.ideal_base_2, join2_0[1]),
              (join2_0[0] - config.ideal_base_2, join2_0[1] - config.ideal_25))

        cam1_coeff = tuple(self.perspective_coefficients(x1, X1))
        if run_data['j20x'] == '0':
            cam2_coeff = 0
        else:
            cam2_coeff = tuple(self.perspective_coefficients(x2, X2))

        return {'cam1': cam1_coeff, 'cam2': cam2_coeff}

    def perspective_transform(self):
        # TODO: flesh out
        coeffs = self.perspective_coefficients()
        for camera, image in self.runfiles:
            im = Image.open(image)
            trans = processing.perspective_transform(im, coeffs[camera])
            trans.save('blah')

    def crop_text(self):
        run_data = self.run_data

        for image, cam in self.runfiles:
            cim = processing.crop(image, run_data, cam)

            time = self.gen_time(image)
            param_text = self.gen_image_text(time)

            kwargs = {'upper_text': param_text,
                      'lower_text': config.author_text,
                      'upper_bar': config.top_bar,
                      'lower_bar': config.bottom_bar,
                      'font': ImageFont.truetype(config.font, 40),
                      'text_colour': 'white',
                      'bg_colour': 'black'}
            dcim = processing.draw_text(cim, **kwargs)

            dcim.save('blah')

    ### argument generators ###
    # No side effects, but rely on external context (config, run_data)
    def crop_box(self, cam):
        """Calculate box needed to crop an image to standard
        orientation.

        Inputs: run_data - a dictionary of measurements
                cam - string, the camera used for the image, e.g. 'cam1'

        Output: a tuple of four integers defining the box
                (left, upper, right, lower)
        """
        run_data = self.run_data
        odd = {'cam1': run_data['odd_1'], 'cam2': run_data['odd_2']}
        if odd[cam] == '999':
            return

        # define the box to crop the image to relative to the
        # invariant point in the projection transform (se).
        l0x = run_data['l0x']
        l0y = run_data['l0y']
        j20x = run_data['j20x']
        j20y = run_data['j20y']
        ref = {'cam1': (l0x, l0y), 'cam2': (j20x, j20y)}
        left = ref[cam][0] + config.crop[cam][0]
        right = ref[cam][0] + config.crop[cam][1]
        upper = ref[cam][1] - config.ideal_25 + config.crop[cam][2]
        lower = ref[cam][1] + config.crop[cam][3]
        return (left, upper, right, lower)

    def gen_image_text(self, time):
        """Create text string that lists the parameters used
        for the run an image came from.

        Inputs: time - a number giving the time the image
                       was taken at (seconds)

        Returns: a string.
        """
        parameters = self.parameters
        param = ("run {run_index}, "
                 "t = {time}s: "
                 "h_1 = {h_1}, "
                 "rho_0 = {rho_0}, "
                 "rho_1 = {rho_1}, "
                 "rho_2 = {rho_2}, "
                 "alpha = {alpha}, "
                 "D = {D}")
        param_text = param.format(time=time, **parameters)
        return param_text



# this class is at the run level.
# the other level that we could conceive of using is the image level
# say by extending the PIL Image class by composition.
# then we would do things like
#
# LabImage = compose(Image)

# im = LabImage.open(file)
# im.barrel_correct()
# im.crop_text()

# this isn't consistent with the design of my pure functions so far,
# which operate on a Image object.

# It doesn't really make sense to process images in isolation - the
# context for doing so is determined by the run that they belong to
# anyway. Therefore, stick with this run level class.
