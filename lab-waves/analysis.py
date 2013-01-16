from __future__ import division

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

import functions as f
from aolcore import read_data, get_parameters, read_simple
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
    def load_simple(self):
        self.simple = read_data(self.simple_data)
        return self.simple

# Getting subsets of all of the runs is important for dealing with
# results. How to?  make a list of the run indices indices =
# pull_col(0, paramf) make a list of run objects runs = [Run(index)
# for index in indices] select sub groupings, e.g.  partial_runs =
# [r for r in runs if r.D == 0.4]

def main(run):
    r = Run(run)
    data = read_simple(run)

    F = sorted(data['front'], key=lambda p: p.t)
    H = sorted(data['head'], key=lambda p: p.t)

    # point = namedtuple(arg, 'x, z, t, u')

    # calculate the front velocity
    # difference
    Fx = [f.x for f in F]
    Fu = list(np.diff(Fx))
    Fu.insert(0, 0)
    Ft = [f.t for f in F]
    Hz = [h.z for h in H]
    Ht = [h.t for h in H]

    # test plot
    plt.plot(Ft, Fu)
    plt.plot(Ft, Fu, 'ko')
    plt.plot(Ft, Fx, 'ro')
    plt.plot(Ht, Hz, 'yo')

    plt.figure()
    X = [ufloat((i, 0.04)) for i in Fx]
    H = [ufloat((i, 0.02)) for i in Hz]
    U = list(np.diff(X))
    U.insert(0, ufloat((0,0.01)))
    Theta = [u * h ** 2 / x for u, h, x in zip(U, H, X)]
    theta = [i.nominal_value for i in Theta]
    etheta = [i.std_dev() for i in Theta]
    plt.plot(Ft, theta)
    plt.plot(Ft, theta, 'ko')
    plt.errorbar(Ft, theta, yerr=etheta)
    plt.xscale('log')
    plt.xlim(1, 40)
    plt.yscale('log')
    plt.ylim(0.00005, 0.1)
    plt.xlabel('time from lock releae (s)')
    plt.ylabel(r'$\Theta$')
    plt.show()

    # smooth

    # back interpolate

if __name__ == '__main__':
    main('r11_7_06c')

class Analysis(Run):
    # maybe use this instead of the main loop??
    pass
