"""Collection of pure functions that operate on individual image objects

Functions:

- barrel_correct: correct barrel distortion

"""

# necessary IO modules for using external functions
import os
import subprocess

import Image


def barrel_correct(im, coeffs, verbose=False, tmp_in='/tmp/bctmpin', tmp_out='/tmp/bctmpout', tmp_fmt='bmp'):
    """Uses Imagemagick convert to apply a barrel correction to
    an image.

    Inputs: im - a PIL Image object
            coeffs - a list of four coefficients [a, b, c, d] which
                     can be str, float or int, will be converted to
                     str anyway.
            verbose - print output to screen?
            tmp_in - input temp file
            tmp_out - output temp file
            tmp_fmt - temp file format, default is raw bitmap

    Outputs: A corrected PIL Image object.

    Barrel distortion is radial. For a given output pixel a
    distance r_dest from the centre of the image, the corresponding
    source pixel is found at a radius r_src using a set of four
    coefficients:

    a, b, c, d = coeffs
    r_src = r_dest * (a * r_dest ** 3 + b * r_dest ** 2 + c * r_dest + d)

    This function is a bit annoying as it calls an external function
    to do the work, which means doing IO in a supposedly pure
    function.

    The solution is to create a temporary file and re-read it
    to an image object. This won't be as fast as it could be,
    but consistency of these low level functions is more important
    than performance as we can solve the latter by using a bigger
    computer.
    """
    # check image is RGB
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    # create temp files
    tin = tmp_in + '.' + tmp_fmt
    tout = tmp_out + '.' + tmp_fmt
    im.save(tin, tmp_fmt)

    # format coefficients for convert
    scoeffs = ' '.join([str(c) for c in coeffs])

    cmd = ["convert",
           "-verbose",
           "-distort", "Barrel",
           scoeffs,
           tin,
           tout]
    if verbose:
        subprocess.call(cmd)
    elif not verbose:
        subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_im = Image.open(tout)

    # cleanup
    os.remove(tin)
    os.remove(tout)

    return out_im
