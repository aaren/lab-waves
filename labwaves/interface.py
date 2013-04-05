"""Pure functions for pulling interfaces out of images
"""


def interface(image):
    """Pull out the interfaces from a image object.

    Like get_data.get_basic_frame_data.

    Inputs: image - a PIL Image object

    Returns - a container (TODO: list, dict?) of
              numpy arrays that detail the interfaces
    """
    # TODO: write interfacing function
    pass


def qc(interface, ftype):
    """Quality control an interface.

    This means rejecting silly points and smoothing if necessary.

    Like get_data.get_frame_data

    Inputs: interface - a numpy array
            ftype - string, the type of interface to qc
                    (waves or gravity current?)

    Returns - a quality controlled interface
    """
    # TODO: write quality control
    return interface
