"""Pure functions for pulling interfaces out of images
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import filter as skif

from runbase import real_to_pixel

import config

# NOTE: scipy.signal._peak_finding._identify_ridge_lines may be
# useful for wave tracking in the signal data, as well as
# scipy.signal.find_peaks_cwt


class InterfaceImage(object):
    def __init__(self, labimage):
        self.labimage = labimage
        self.im = self.measurement_region(labimage)
        self.imarray = np.asarray(self.im)
        self.pixels = np.indices(self.imarray[:, :, 0].shape)

    @staticmethod
    def measurement_region(labimage):
        """Remove the bars from the top and bottom of a lab image."""
        w, h = labimage.im.size
        box = (0, config.top_bar, w, h - config.bottom_bar)
        return labimage.im.crop(box)

    @property
    def channels(self):
        """Return the r, g, b channels of the lab image,
        normalised on the range [0, 1].
        """
        return [c.squeeze() / 255. for c in np.dsplit(self.imarray, 3)]

    @property
    def lock_fluid(self):
        """Return the color space projection that best captures
        the fluid from the lock.

        The green channel picks it out the best as a single channel,
        but is contaminated by background noise.

        This noise is largely neutrally toned though, so we can
        subtract one of the other channels to remove it.
        """
        r, g, b = self.channels
        return r - g

    def plot_channels(self):
        """Convenience function to make a figure with the input
        image and the three colour channels."""
        r, g, b = self.channels

        fig, ax = plt.subplots(nrows=5)

        ax[0].set_title('Original')
        ax[0].imshow(self.imarray)

        ax[1].set_title('red')
        ax[1].imshow(r)

        ax[2].set_title('green')
        ax[2].imshow(g)

        ax[3].set_title('blue')
        ax[3].imshow(b)

        ax[4].set_title('r - g')
        ax[4].imshow(self.lock_fluid)

        fig.tight_layout()

        return fig

    @property
    def pixel_rulers(self):
        """Where are the rulers in pixel space"""
        cam = self.labimage.cam
        real_rulers = config.real_rulers[cam]
        pixel_rulers = [[real_to_pixel(x, 0, cam)[0] for x in r]
                        for r in real_rulers]

        style = self.labimage.run.style
        if style == 'old':
            pass
        elif 'new' in style:
            # only one ruler in cam1
            pixel_rulers = [pixel_rulers[0]]

        return pixel_rulers

    @property
    def ruler_mask(self):
        """Mask for rulers in the image."""
        iy, ix = self.pixels
        # Truth wherever there is a ruler
        mask = reduce(np.logical_or, ((x1 < ix) & (ix < x2)
                                      for x2, x1 in self.pixel_rulers))

        return mask

    @property
    def bottom_mask(self):
        """Mask the bottom 5 pixels of the image."""
        iy, ix = self.pixels
        mask = iy > self.im.size[1] - 5
        return mask

    @property
    def crop_mask(self):
        """Mask where the image has been cropped, i.e. there is black
        background."""
        return np.all(self.imarray < 0.01, axis=-1)

    @property
    def lock_mask(self):
        """Combination of masks that masks out irrelevant features
        for the lock fluid detection."""
        return reduce(np.logical_or, (self.ruler_mask,
                                      self.bottom_mask,
                                      self.crop_mask,
                                      self.lock_fluid < 0.2))

    @property
    def lock_interface(self):
        """Pull out the lock interface."""
        # TODO: make it only select the upper bound
        mask = self.lock_mask
        i = self.canny_interface(self.lock_fluid < 0.5, smooth=0, mask=~mask)
        return i

    @staticmethod
    def canny_interface(array, sigma=5, mask=None):
        """Perform canny edge detection on the given array, using
        gaussian smoothing of sigma, only considering the regions
        defined as True in mask.
        """
        cb = skif.canny(array, sigma=sigma, mask=mask)
        iy, ix = np.where(cb)
        s = ix.argsort()
        return ix[s], iy[s]


def test_interface():
    fig = iim.plot_channels()

    lx, ly = iim.lock_interface

    for ax in fig.axes:
        ax.plot(lx, ly, 'k.')

    plt.show()

from runbase import ProcessedRun

pr = ProcessedRun('r13_01_13i')
iim = InterfaceImage(list(pr.images)[30])
# pr = ProcessedRun('r11_07_06c')
# iim = InterfaceImage(list(pr.images)[6])


if __name__ == '__main__':
    test_interface()
