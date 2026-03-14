"""
Stuff related to the harmonic oscillator.
"""
import os
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson
from scipy.special import gammaln
from scipy.integrate import simpson

from qtoc_krylov.utilities.store import store
from qtoc_krylov.utilities.paths import *


def h_harmonic(cutoffs, hbar=1):
    """
    Harmonic oscillator Hamiltonian with unit frequency and mass, written in
    the number representation, with cut-offs at the n-th and m-th
    (minimum and maximum) excitation states.
    """
    n, m = cutoffs
    return hbar*np.diag(np.arange(m-n+1) + 1/2)


def u_harmonic(dt, cutoffs, hbar=1):
    """
    Quantum harmonic oscillator propagator with unit frequency and mass,
    defined on the unit square, and with fixed point at (q, p) = (qe, pe).
    The evolution is propagated forwards in time in a `dt` time step.
    """
    h = h_harmonic(cutoffs, hbar=hbar)
    return expm(-1j*dt*h/hbar)


def harmonic_map(q, p, dt=0.01):
    """
    Classical Harmonic oscillator map with unit frequency and mass defined by
    its stroboscopic motion with period dt.
    """
    q_ = q + dt*p
    p_ = p - dt*q_
    return q_, p_


def harmonic_map_inv(q_, p_, dt=0.01):
    """
    Inverse classical Harmonic oscillator map with unit frequency and mass
    defined by its stroboscopic motion with period dt.
    """
    p = p_ + dt*q_
    q = q_ - dt*p
    return q, p


##
def husimi_operator(q, p, op, cutoffs, hbar=1):
    """
    Evaluate Husimi operator distribution on the plane at the position (q, p).
    The operator `op` is assummed to be written in the number basis.
    """
    qq, pp = np.meshgrid(q, p)
    pois2D = poisson(mu=(qq**2 + pp**2)/(2*hbar))
    theta = np.atan2(pp, qq)

    n, m = cutoffs
    a_psi = np.zeros((p.size, q.size), dtype=complex) # <a|psi>

    x = np.arange(n, m+1)
    n_a = np.zeros((x.size, *a_psi.shape), dtype=complex) # <n|a>
    for i, k in enumerate(x):
        n_a[i] = np.exp(1j*k*theta)*pois2D.pmf(k)**0.5

    a_psi = np.einsum('iqp,ij,jqp->qp', n_a.conj(), op, n_a, optimize=True)
    return np.real(a_psi)/(2*np.pi*hbar)


def husimi_state(q, p, psi, cutoffs, hbar=1):
    """
    Evaluate Husimi distribution on the plane at the position (q, p).
    The state `psi` is assummed to be written in the number basis.
    """
    qq, pp = np.meshgrid(q, p)
    pois2D = poisson(mu=(qq**2 + pp**2)/(2*hbar))
    theta = np.atan2(pp, qq)

    n, m = cutoffs
    a_psi = np.zeros((p.size, q.size), dtype=complex) # <a|psi>
    for i, x in enumerate(np.arange(n, m+1)):
        a_psi += psi[i] * np.exp(-1j*x*theta)*pois2D.pmf(x)**0.5
    return np.abs(a_psi)**2/(2*np.pi*hbar)


##
def cutoffs_for_coherent_state(f, q, p, hbar=1):
    """
    Calculate the minimum and maximum number of excitations needed to
    accurately write the coherent state |a>, with a = (q + ip)/sqrt(2*hbar),
    in terms of the number states |n>, with a fidelity of |<a|a_approx>|^2 = f.

    See: notebook B (orange, beach), pages 43-44 and 60.
    """
    n, m = poisson(mu=(q**2 + p**2)/(2*hbar)).interval(f**0.5)
    return int(n), int(m)


def coherent_state(q, p, cutoffs, hbar=1):
    """
    Coherent state on the plane, localized at (q, p), written in the number
    representation, with cut-offs at the n-th and m-th (minimum and maximum)
    excitation states.

    See: notebook B (orange, beach), page 46.
    """
    n, m = cutoffs
    x = np.arange(n, m+1)
    pois = poisson(mu=(q**2 + p**2)/(2*hbar))
    theta = np.atan2(p, q)
    ket = np.exp(1j*x*theta)*pois.pmf(x)**0.5 # <x|a>
    return ket.reshape((m-n+1, 1))


def _integration_limits(qlims, plims, hbar=1):
    """
    Given the q and p values that hold f's support, translate those values into
    the polar coordinates used for integration.

    Limiting the integration region to a size comparable to the integrand's
    support is necessary for integration not to break. For instance, see the
    last example in:
    http://scipy.github.io/devdocs/reference/generated/scipy.integrate.quad.html
    """
    ## angular limits
    angles = []
    for q in qlims:
        for p in plims:
            angles.append(np.atan2(p, q))
    phil = np.min(angles)
    phir = np.max(angles)
    philims = (phil, phir)

    ## radial limits
    ql, qr = qlims
    pl, pr = plims

    q0 = (ql + qr)/2
    dq = (qr - ql)/2
    p0 = (pl + pr)/2
    dp = (pr - pl)/2

    u0 = ((q0**2 + p0**2)/(2*hbar))**0.5
    du = ((dq**2 + dp**2)/(2*hbar))**0.5

    ul, ur = u0 - du, u0 + du
    if ul < 0:
        ul = 0
        philims = (0, 2*np.pi)
    ulims = (ul, ur)
    return ulims, philims


def cutoffs_for_coherent_ensemble(f, qlims, plims, hbar=1):
    """
    This is a heuristic calculation. The idea is to ensure that both the
    coherent state with minimal energy and the one with maximal energy
    contained in the ensemble have a high fidelity.
    """
    cutoffs_min = cutoffs_for_coherent_state(f, qlims[0], plims[0], hbar=hbar)
    cutoffs_max = cutoffs_for_coherent_state(f, qlims[1], plims[1], hbar=hbar)
    return cutoffs_min[0], cutoffs_max[1]


@store(path=STORE_DIR)
def coherent_ensemble(f, qlims, plims, cutoffs, args=(), hbar=1):
    """
    Density matrix for a classical ensemble of coherent states
    written in the excitation number representation.
    See: notebook B (orange, beach), p. 88.

    The values `qlims` and `plims` represent f's approximate
    support to carry over the integrations below.
    See the `_integration_limits` docstring.
    """
    r2 = (2*hbar)**0.5

    def fpp(u, phi): return f(r2*u*np.cos(phi), r2*u*np.sin(phi), *args)

    u_lims, phi_lims = _integration_limits(qlims, plims, hbar=hbar)

    Nres = 300
    us = np.linspace(*u_lims, Nres)
    phis = np.linspace(*phi_lims, Nres)
    uu, phiphi = np.meshgrid(us, phis)

    # evaluate functions
    ff = fpp(uu, phiphi)
    uu_squared = uu**2
    uu_log = np.log(uu)

    def logbadpart(k, l):
        """
        Calculate this in a logarithm in order to avoid overflows.
        """
        out = -uu_squared + (k+l+1)*uu_log - 0.5*(gammaln(k+1) + gammaln(l+1))
        return out

    def expphi(k, l):
        return np.exp(1j*phiphi*(k-l))

    global calculate_element # avoid AttributeError

    def calculate_element(indices):
        k, l = indices
        out = (2*hbar)*ff*np.exp(logbadpart(k, l))*expphi(k, l)
        out = simpson(simpson(out, x=us), x=phis)
        return (k, l), out

    n, m = cutoffs
    out = np.zeros((m-n+1, m-n+1), dtype=complex)
    kl = np.arange(n, m+1)
    offdiag = np.asarray(list(combinations(kl, 2)))
    diag = np.asarray(list(zip(kl, kl)))

    with Pool(None) as p:
        pmap_offdiag = p.map(calculate_element, offdiag)
        pmap_diag = p.map(calculate_element, diag)

    for (k, l), val in pmap_offdiag:
        out[k-n, l-n] = val
        out[l-n, k-n] = val.conjugate()

    for (k, _), val in pmap_diag:
        out[k-n, k-n] = val

    trace = np.real(np.linalg.trace(out))
    print(f'Trace: {trace}')
    if abs(1 - trace) > 1e-4:
        print('WARNING: trace is not 1. Integration limits may be wrong, or cutoff too aggressive')
    return out
