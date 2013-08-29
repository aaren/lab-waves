You have done some experiments in the lab, the output consisting of
a series of images that look something like this:

![RAW-lab-image.png] TODO: make raw image, link to image

These images could be from a camera shooting stills at a given rate,
or be stills from a video. Whatever, you've got a series of images
and you want to extract data from them.

We're going to turn these images into something like this:

![PROC-lab-image.png] TODO: make processed image, link to image

and then distill the essence of the image, the borders between the
coloured fluids, into two one-dimensional vectors that look like
this:

![INTERFACE-lab-image.png] TODO: make interface image, link to image

### Usage ###

TODO: how to use the program

#### Command line

#### API

Standardising the images from a give run is easy:

```python
from labwaves import RawRun

run = 'r11_07_06c'
r = RawRun(run)
r.process()
```

To do the steps explicitly:

```python
# barrel correct first image
r.bc1()
# obtain run data from first image, unless data already exists
r.get_run_data()
# barrel correct all images in run
r.barrel_correct()
# perspective correct all images in run
r.perspective_transform()
# crop and add border info
r.crop_text()
```

We can do different things to a standardised run. For example
**stitching images together**:

```python
from labwaves import ProcessedRun

r = ProcessedRun('r11_07_06c')

# an iterator of stitched images from the run
si = r.stitched_images
```

We can write out images to disk:

```python
r.write_out(r.stitched_images)
```

`r.stitched_images` could be any sequence of images objects that
have a `write_out()` method. This allows us to make new images that
derive from the Processed Images easily.


### Data organisation ###

Each run consists of a series of images and an entry in a
`parameters` file that contains details of the run (fluid density,
depths)

### Code organisation ###

We can split the work into three sections, *processing*,
*interfacing* and *analysis*. 

There are two basic units of data: the *image* and the *run*.
A run consists of a series of images.

The basic functions take an image as input and output a new image
without changing anything, i.e. they are pure functions.

These are applied by functions that actually do the IO for an
individual image and for a whole run.

**TODO: the applicator of the functions would be a class??**
Basically, yes. I've just added a stub for a RawRun class that would
contain the applicator functions in processing.py

There is already the Run class for analysis.

generate.py is essentially a class, or would go into one very
easily, and this does interfacing and raw.

So we already have 3 classes sketched out, just need to formalise
this.

Pure functions go in a module and are imported to be used in the
class.

Applicator functions that do image IO are the class methods.

What about the functions that generate arguments for the pure
functions?? Where should these go? They have no side effects.

They produce context for a given run / image. Class methods? 
Contain in seperate context class that we then compose to make the
IO class?

**/TODO**

We parallelise using [multiprocessing][]. This can operate on two
levels - processing a set of images in parallel and processing
multiple runs in parallel.

[multiprocessing]: http://docs.python.org/2/library/multiprocessing.html

explain multiple progress bars, link to github demo

#### Processing ####

Raw image --> processed image

Takes a set of raw lab images and outputs a set of processed lab
images. A processed lab image is standardised so that pixel
measurements in the image correspond to absolute physical
measurements.

Stages:

- Optical correction
    - barrel distortion
    - perspective

- Orientation
    - aided user measurement
    - application

- Cropping / bordering


#### Interfacing ####

Processed image -> interfaces
    
Takes a set of processed lab images and outputs a set of interfaces.
This is done by thresholding the image using
[scikits-image][skimage].

[skimage]: TODO: skimage link


### Refactoring ###

New files are being made to replace the functionality of old ones.
The general pattern is to take a module and make it into a class,
making some core pure functions in the process and putting them
into a new module.

proc\_im (old) --> raw.RawRun (class), processing (pure functions)

get\_data, threshold --> processed.ProcessedRun, interface

generate --> command.CommandRun +

This is an intermediate step on the way to creating a single WaveRun
class that consists of the above, and a series of modules that
contain pure functions that do stuff.

+ The top level generate module needs a bit of thought. It exists
  for a number of reasons:

    1) Allow multiprocessing

    2) Provide command line interface

    3) Organise all the module level stuff so we don't have to
       think about it.

Multiprocessing is imperative. Must allow this to happen.

The command line interface can be easily separated out and cleaned
up by using argparse.

Point 3 exists precisely because the implementation of the other
modules is so messy. It is essentially putting a class like
structure in place so that it is easier to see how to do different
things. Once the modules are cleaned up and absorbed into a single
class, the need for the generate 'class' is lost.


#### Analysis ####

Separate raw processing and analysis. The above should extract the
interfaces from a run and perhaps identify distinct wave objects -
this gives us the raw data. Analysis of this data should occur
separately I think.
