from __future__ import division

from sys import argv
from collections import namedtuple
import json

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

from config import data_dir

from run import Run


class Analysis(object):

    def __init__(self, run):
        self.__dict__ = Run(run).__dict__
        self.simple = self.read_simple()

    def read_simple(self, args='x, z, t'):
        """Reads in a JSON file that is in the format

        {keys: [[x,z,t], [x,z,t], ...]}

        and returns a data structure in the format

        {keys: [key(x=a, z=b, t=c), ...]}

        i.e. a dict of lists of namedtuples. If the
        namedtuple attributes are different from
        'x, z, t', these can be given as an argument
        but you must know a priori what these are as
        the JSON container doesn't store these.
        """
        if hasattr(self, 'simple'):
            return self.simple
        else:
            print "reading data!"
            dataf = data_dir + 'simple/simple_%s.json' % self.index
            fin = open(dataf, 'r')
            idata = json.loads(fin.read())
            fin.close()
            ndata = {}
            for k in idata:
                point = namedtuple(k, args)
                ndata[k] = [point(*p) for p in idata[k]]
            self.simple = ndata
        return ndata

    def front(self):
        data = self.read_simple()
        F = sorted(data['front'], key=lambda p: p.t)
        return F

    def head(self):
        data = self.read_simple()
        H = sorted(data['head'], key=lambda p: p.t)
        return H

    @property
    def Fx(self):
        return [ufloat((f.x, 0.04)) for f in self.front()]

    @property
    def Fu(self):
        # calculate the front velocity
        # TODO: smooth and reinterpolate so there isn't a half
        # second offset.
        Fu = list(np.diff(self.Fx / np.diff(self.Ft)))
        # here this is offset by 1 and there is one less value

        # really it is offset by 0.5

        tin = self.Ft
        tout = [np.mean(tin[i: i + 2]) for i, v in list(enumerate(tin))[:-1]]
        Fu = np.interp(tin, tout, Fu)

        # we know that front velocity is 0 at 0
        # Fu.insert(0, ufloat((0, 0.01)))

        return Fu

    @property
    def Ft(self):
        return [f.t for f in self.front()]

    @property
    def Ht(self):
        return [h.t for h in self.head()]

    @property
    def Hz(self):
        return [ufloat((h.z, 0.02)) for h in self.head()]

    def test_plot(self):
        # test plot of simple things
        Fu = [i.nominal_value for i in self.Fu]
        Fx = [i.nominal_value for i in self.Fx]
        Fue = [i.std_dev() for i in self.Fu]
        Fxe = [i.std_dev() for i in self.Fx]
        Ft = self.Ft
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.errorbar(Ft, Fx, yerr=Fxe, fmt='r.', label=r'$x_n$')

        plt.title(r"$\alpha = %s$ $\rho_0 = %s$ $\rho_1 = %s$ $\rho_2 = %s$ $h_1 = %s$"
                  % (self.a, self.r0, self.r1, self.r2, self.h1))
        plt.xlabel("Time from lock release (s)")

        ax2 = ax1.twinx()
        ax2.errorbar(Ft, Fu, yerr=Fue)
        ax2.plot(Ft, Fu, 'ko', label=r'$u_n$')
        Ht = self.Ht
        Hz = [i.nominal_value for i in self.Hz]
        ax2.plot(Ht, Hz, 'yo', label=r'$h_n$')

        plt.savefig('figs/%s_basic.png' % self.index)

    def theta(self):
        Fx = self.Fx
        Hz = self.Hz
        U = self.Fu
        Theta = [u * h ** 2 / x for u, h, x in zip(U, Hz, Fx)]
        return Theta

    def ungarish_theta(self):
        # Ungarish theta plot
        plt.figure()
        plt.title(r"$\alpha = %s$ $\rho_0 = %s$ $\rho_1 = %s$ $\rho_2 = %s$ $h_1 = %s$"
                  % (self.a, self.r0, self.r1, self.r2, self.h1))
        theta = [i.nominal_value for i in self.theta()]
        etheta = [i.std_dev() for i in self.theta()]
        plt.plot(self.Ft, theta, 'ko')
        plt.errorbar(self.Ft, theta, yerr=etheta)
        plt.xscale('log')
        plt.xlim(1, 40)
        plt.yscale('log')
        plt.ylim(0.00005, 0.1)
        plt.xticks([1, 5, 10, 15, 20, 25, 30, 35])
        plt.xlabel('time from lock releae (s)')
        plt.ylabel(r'$\Theta$')

        plt.savefig('figs/%s_theta.png' % self.index)

    def dynamic_reynolds(self):
        # Dynamic reynolds number plot
        U = self.Fu
        H = self.Hz
        plt.figure()
        Re = [u * h / 0.000001 for u, h in zip(U, H)]
        nRe = [i.nominal_value for i in Re]
        eRe = [i.std_dev() for i in Re]
        plt.plot(self.Ft, nRe, 'b-', label='dynamic')
        plt.plot(self.Ft, nRe, 'ko')
        plt.errorbar(self.Ft, nRe, yerr=eRe)

        # simplistic reynolds number
        sRe = [0.2 * 0.4 * 10E6 for t in self.Ft]
        plt.plot(self.Ft(), sRe, 'k-', label='simple')

        # Ungarish full ratio of inertial to viscous
        Rf = [th * 0.2 * 0.4 * 0.4 * 1E6 for th in self.theta()]
        R = [i.nominal_value for i in Rf]
        plt.plot(self.Ft, R, 'r-', label='ungarish box')

        plt.title(r"$\alpha = %s$ $\rho_0 = %s$ $\rho_1 = %s$ $\rho_2 = %s$ $h_1 = %s$"
                  % (self.a, self.r0, self.r1, self.r2, self.h1))
        plt.xscale('log')
        plt.xlim(1, 40)
        plt.yscale('log')
        plt.ylim(1, 1E7)
        plt.xlabel('time from lock releae (s)')
        plt.ylabel(r'$\Re$')
        plt.legend()

        plt.savefig('figs/%s_Recomp.png' % self.index)

    def full_ungarish(self):
        plt.figure()
        U = self.Fu
        H = self.Hz
        Re = [u * h / 0.000001 for u, h in zip(U, H)]
        RT = [r * t for r, t in zip(Re, self.theta())]
        nRT = [i.nominal_value for i in RT]
        eRT = [i.std_dev() for i in RT]
        # plt.plot(Ft, nRT)
        plt.plot(self.Ft, nRT, 'ko')
        plt.errorbar(self.Ft, nRT, yerr=eRT)
        plt.xscale('log')
        plt.xlim(1, 40)
        plt.yscale('log')
        plt.ylim(1, 1000)
        plt.xlabel('time from lock releae (s)')
        plt.ylabel(r'$\Re \theta$')


def main(run):
    r = Analysis(run)
    print r.index, r.a, r.r0, r.r1, r.r2
    r.test_plot()
    r.ungarish_theta()
    # continuous error bars
    # surely this needs continuous data??
    r.dynamic_reynolds()
    # r.full_ungarish()
    # plt.show()


if __name__ == '__main__':
    if len(argv) > 1:
        main(argv[1])
    else:
        main('r11_7_06c')

# Getting subsets of all of the runs is important for dealing with
# results. How to?  make a list of the run indices indices =
# pull_col(0, paramf) make a list of run objects runs = [Run(index)
# for index in indices] select sub groupings, e.g.  partial_runs =
# [r for r in runs if r.D == 0.4]
