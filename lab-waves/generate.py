import proc_im

# barrel correct the first frame of each camera in a run
# outputs to dir 'bc1' in the path
def proc_im_main(run):
    proc_im.bc1(run)
    proc_im.std_corrections(run)
    proc_im.text_crop(run)

