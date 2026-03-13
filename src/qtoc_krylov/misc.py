import numpy as np

def discrete_gauss_1D(n, m, d):
    """
    Discrete-evaluated bell curve with mean 'm' and standard deviation 'd'.
    """
    x = (n-m)/d
    gn = np.exp(-x**2)
    gn /= np.sum(gn)
    return gn


def gauss_1D(x, m, s):
    n = (2*np.pi*s**2)**0.5
    return np.exp(-((x-m)**2)/(2*s**2))/n


def gauss_2D(x, y, x0, y0, s):
    """
    Equivariant 2D Gaussian distribution.
    """
    return gauss_1D(x, x0, s)*gauss_1D(y, y0, s)


def periodic_gauss_1D(x, m, s):
    """
    Periodic Gaussian distribution with mean position `m` and
    standard deviation `s`, defined on the unit interval.
    """
    if s > 1/(2*np.pi)**0.5:
        raise Exception('A value `s` > 0.4 breaks partial sum approximation.')
    out = np.zeros_like(x)
    for n in range(-2, 3):
        out += np.exp(-(x-m+n)**2/(2*s**2))
    out /= (2*np.pi*s**2)**0.5 # same normalization as usual Gaussian holds
    return out


def periodic_gauss_2D(x, y, x0, y0, s):
    """
    Periodic 2D Gaussian distribution with mean position `m` and
    standard deviation `s`, defined on the unit square.
    """
    return periodic_gauss_1D(x, x0, s)*periodic_gauss_1D(y, y0, s)
