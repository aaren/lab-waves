# This is a basic spec for using labwaves to process some lab images

from labwaves import RawRun

run = 'r11_07_06c'
r = RawRun(run)

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

# TODO: explicit write to disk for run processing?
# could keep run images in memory to allow fast chaining
# of barrel correct, perspective trans, crop text
# and then only write when we do r.write_all()

## At this stage, the run has been standardised and we are
## ready to have the interfaces extracted

r = ProcessedRun(run)
# extract interfaces from images
r.interface()

## now the data has been extracted and we can try and identify individual waves
r = Run(run)
# pull out the waves - requires user input
r.identify()
