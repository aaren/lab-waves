# I want to produce a load of plots of theoretical wave speeds.
from __future__ import division
import numpy as np
# Two layer linear waves
g = 9.81
H = 0.25 # metres
pi = 3.1415

def rg(rho0, rho1):
    """Calculate the reduced gravity between two fluids.
    arguments: rho0 is the higher density, rho1 the lower.
    """
    drho = (rho0 - rho1)
    div = (rho1)
    g_ = g * drho / div
    return g_


def two_layer_linear(h1, h2, rho1, rho2, k):
    drho = rho1 - rho2
    c = (g * H * drho / (k * (rho1 / np.tanh(k*h1) + \
                                rho2 / np.tanh(k*h0)))) ** .5
    return c/H


# long wave limit
def two_layer_linear_longwave(h1, h2, rho1, rho2):
    drho = rho1 - rho2
    c = (g * H * drho * (rho1 / h1 + rho2 / h2)**-1) ** .5
    # c units are SI
    # return units are non dimensional
    return c/H

def shallow_linear(h, rho0, rho1):
    g_ = rg(rho0, rho1)
    return (g_ * h * H)**.5 / H


def gc_empirical(h, rho0, rho1):
    """Empirically observed gravity current speed.
    Arguments: h, g.c. height (fractional!, i.e. actually h/H)
               rho1, ambient density
               rho0, g.c. density
    Returns: Speed in lock lengths
    """
    Fr = 0.5 * h**-(1/3)
    g_ = rg(rho0, rho1)
    c = Fr * (g_ * h * H)**.5
    return c/H

def gc_theoretical(h, rho0, rho1):
    Fr = ((2 - h) * (1 - h) / (1 + h))**.5
    g_ = rg(rho0, rho1)
    c = Fr * (g_ * h * H)**.5
    return c/H
