# I want to produce a load of plots of theoretical wave speeds.
from __future__ import division

# Two layer linear waves
g = 9.81
H = 0.25
pi = 3.141
rho0 = 1.150
rho1 = 1.100
rho2 = 1.000
drho = rho1 - rho2
h1 = 0.25
h2 = 0.75
h0 = 0.4
g_ = g * drho / rho1
hg = h0 / 2
a = h1

class TwoLayer(object):
    def __init__(self):
        pass
    # general
    def linear(rho0, rho1, h0, h1, k):
        pass
        
    # H = lambda --> coth -> 1
    @staticmethod
    def equiv(rho1=rho1, rho2=rho2):
        c = (g * H * (rho1 - rho2) / (2 * pi * (rho1 + rho2)))**.5
        return c/H

    # long wave limit
    @staticmethod
    def linear_longwave():
        c = (g * H * drho * (rho1 / h1 + rho2 / h2)**-1)**.5
        # c units are SI
        # return units are non dimensional
        return c/H

    @staticmethod
    def linear_shallow():
        return (g_ * h1 * H)**.5 / H

class GCHomo(object):
    @staticmethod
    def empirical():
        Fr = 0.5 * a**-(1/3)
        c = Fr * (g_ * hg * H)**.5
        return c/H
    @staticmethod
    def theoretical():
        Fr = ((2 - a) * (1 - a) / (1 + a))**.5
        c = Fr * (g_ * hg * H)**.5
        return c/H
        
def plot():
    t = np.linspace(0, 30, 31)
    
