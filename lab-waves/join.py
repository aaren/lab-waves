import Image
import os
import glob
import sys

# A collection of tools for joining together processed images
# join joins cam1 and cam2 images together for a single run,
# based on the standard offset at the edge of processed lab
# images.

def simple_join(path, image, gap=0, remove_text=0):
        print("Joining %s ..." %(path))
        cam1 = Image.open('%s/cam1/%s' % (path, image))
        cam2 = Image.open('%s/cam2/%s' % (path, image))

        # images are 2810x690 (+/- 1 on both) with the offset 60 from the end.
        inset = 60
        # get the image dimensions, just in case they vary from above slightly
        w1, h1 = cam1.size
        w2, h2 = cam2.size

        # remove text?
        if remove_text==1:
            cam1 = rem_text(cam1)
        else:
            pass

        # define the crop boxes
        box_1 = (inset, 0, w1, h1)
        box_2 = (0, 0, w2 - inset, h2)

        # crop the images
        crop_cam1 = cam1.crop(box_1)
        crop_cam2 = cam2.crop(box_2)

        # make a new, empty image
        w = w1 + w2 - (2 * inset) + gap
        if h1 > h2:
            h = h1
        else:
            h = h2
        joined_image = Image.new("RGB", (w, h))

        # paste the cropped images into the blank_image
        joined_image.paste(crop_cam2, (0, 0))
        joined_image.paste(crop_cam1, (w2 - inset + gap, 0))

        return joined_image

def rem_text(im):
    """Take an image object and paste over the text areas,
    returning an image object"""
    w, h = im.size

    top_bar = (0, 0, w, 100)
    bottom_bar = (0, h-100, w, h)

    im.paste('black', top_bar)
    im.paste('black', bottom_bar)
    
    return im
    

def join(run, proc_dir):
    path = '%s/%s' % (proc_dir, run)
    # make a new directory for the joined images if it doesn't already exist
    if not os.path.exists('%s/join' % path):
        os.mkdir('%s/join' % path, 0755)
    else:
        print("run %s has been joined, skipping..." % run)
        return 
    for image in glob.glob('%s/cam1/*' % (path)):
        image = image.split('/')[-1]
        joined_image = simple_join(path, image)
        # save the joined image, first creating a new directory called 'join'
        joined_image.save('%s/join/%s' % (path, image))
        print("...saved to %s/join/%s" % (path, image))

def remove_text(run, proc_dir):
    path = '%s/%s' % (proc_dir, run)

    if not os.path.exists('%s/join_notext' % path):
        os.mkdir('%s/join_notext' % path, 0755)
    else:
        print("run %s has been de-texted, skipping..." % run)
        return 
    
    for image in glob.glob('%s/join/*' % (path)):
        im = Image.open(image)
        print("Removing text from %s" % image)
        image = image.split('/')[-1]

        outfile = '%s/join_notext/%s' % (path, image)


        w, h = im.size

        top_bar = (0, 0, w, 100)
        bottom_bar = (0, h-100, w, h)

        im.paste('black', top_bar)
        im.paste('black', bottom_bar)
        
        im.save(outfile)

def remove_borders(run, proc_dir):
    path = '%s/%s' % (proc_dir, run)

    if not os.path.exists('%s/join_noborder' % path):
        os.mkdir('%s/join_noborder' % path, 0755)
    else:
        print("run %s has been de-bordered, skipping..." % run)
        return 
    
    for image in glob.glob('%s/join/*' % (path)):
        im = Image.open(image)
        print("Removing borders from %s" % image)
        image = image.split('/')[-1]
        outfile = '%s/join_noborder/%s' % (path, image)

        w, h = im.size
        
        box = (0, 100, w, h-100)
        cropped = im.crop(box)
        cropped.save(outfile)

def runs(proc_dir):
    runs = [path.split('/')[-1] for path in glob.glob('%s/*' % proc_dir)]
    return runs

def presentation(run, proc_dir):
    """Prepares run for presentation by joining the images together
    with a small black space between to make clear the discontinuity
    in parallax, and with only the cam2 text shown."""

    path = '%s/%s' % (proc_dir, run)
    # make a new directory for the joined images if it doesn't already exist
    if not os.path.exists('%s/presentation' % path):
        os.mkdir('%s/presentation' % path, 0755)
    else:
        print("run %s has been joined for presentation, skipping..." % run)
        return

    for image in glob.glob('%s/cam1/*' % (path)):
        image = image.split('/')[-1]
        joined_image = simple_join(path, image, 50, 1)
        # save the joined image, first creating a new directory called 'join'
        joined_image.save('%s/presentation/%s' % (path, image))
        print("...saved to %s/presentation/%s" % (path, image))

def animate(run, proc_dir):
    path = '%s/%s' % (proc_dir, run)
    # make a new directory for the joined images if it doesn't already exist
    if not os.path.exists('%s/animation' % path):
        os.mkdir('%s/animation' % path, 0755)
    else:
        print("run %s has been animated, skipping..." % run)
        return
    if not os.path.exists('%s/presentation' % path):
        print("no source for the animation (i.e. no presentation images), skipping...")
    else:
        pass
    files = glob.glob('%s/presentation/img*jpg')
    images = [Image.open(file) for file in files] 
    outfile = run + '.gif'
    images2gif.writeGif(outfile, images, duration=0.5)

