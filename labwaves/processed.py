# class to deal with images that have been processed from their
# raw state. deals with interface extraction.

# basically takes get_data and puts it in a class

# module for extracting interfaces from image objects
# TODO: make this
import interface

class ProcessedRun(object):
    """Same init as RawRun. At some point these two will be merged
    into a single Run class.
    """
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

    def interface(self):
        """Grab all interface data from a run"""
        # TODO: multiprocessing
        # TODO: runfiles should point to processed data
        for camera, image in self.runfiles:
            im = Image.open(image)
            interfaces = interface.interface(im)
            qc_interfaces = [interface.qc(i) for i in interfaces]
            save_interface(qc_interfaces)




