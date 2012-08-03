import functions as f
from aolcore import read_data, get_parameters
from config import data_dir, data_storage, paramf

class Run(object):

    def __init__(self, run):
        self.index = run
        self.params = get_parameters(run, paramf)
        self.h1 = float(self.params['h_1/H'])
        self.h2 = 1 - self.h1
        self.D = float(self.params['D/H'])
        self.r0 = float(self.params['rho_0'])
        self.r1 = float(self.params['rho_1'])
        self.r2 = float(self.params['rho_2'])
        self.a = float(self.params['alpha'])

        self.c2l = f.two_layer_linear_longwave(self.h1, self.h2, \
                                                self.r1, self.r2)
        self.gce = f.gc_empirical(self.D / 2, self.r0, self.r1)
        self.gct = f.gc_theoretical(self.D / 2, self.r0, self.r1)

        self.data_file = data_storage + self.index
        self.simple_data = data_dir + 'simple/simple_' + self.index

    def load(self):
        # Explicitly load the run data
        self.data = read_data(self.data_file)[self.index]

    def load_basic(self):
        self.basic = read_data(data_dir + 'basic/basic_%s' % run)
