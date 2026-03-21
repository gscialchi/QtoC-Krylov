from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.integrate import simpson

from qtoc_krylov.utilities.store import store
from qtoc_krylov.utilities.paths import *


PI = np.pi
DPI = 2*np.pi


### Classical maps
def c_harper(k, q, p):
    """ Classical Harper map on the unit torus. """
    q_ = q - k*np.sin(DPI*p)
    p_ = p + k*np.sin(DPI*q_)
    return np.mod(q_, 1), np.mod(p_, 1)


def c_harper_inv(k, q_, p_):
    """ Inverse classical Harper map on the unit torus. """
    p = p_ - k*np.sin(DPI*q_)
    q = q_ + k*np.sin(DPI*p)
    return np.mod(q, 1), np.mod(p, 1)


### Quantum maps
"""
Globally set the Floquet angles explicitly.
For discussion, see /tests/floquet_angles.py
"""
QBAR, PBAR = (0.5, 0)


#  @store(path=STORE_DIR)
def q_harper(k, N, qbar=QBAR, pbar=PBAR):
    """
    Unitary for the quantum Harper map. `qbar` and `pbar` define
    the Floquet angles, such that
        |q+1> = exp(2i pi pbar)|q> and |p+1> = exp(-2i pi qbar)|p>.
    """
    qn = np.arange(0, N) + qbar # n + qbar
    pn = np.arange(0, N) + pbar # m + pbar
    # NOTE: usually qn = (n+qbar)/N but here I use a different definition

    qq, pp = np.meshgrid(qn, pn)

    ff = np.exp(1j * DPI/N * qq * pp) / N**0.5

    v_nm = np.exp(-1j*N*k*np.cos(DPI/N * qq)) * ff
    v_mn = np.exp(-1j*N*k*np.cos(DPI/N * pp)) * ff.conj()

    u = v_nm.T @ v_mn
    return u


## Husimi distribution on the unit torus
def jacobi_theta(q, p, N, qn):
    """
    Approximated controlled Jacobi Theta function by selecting a good cutoff
    for the sum. It is accurate to within an atol of at least 1e-9/N
    for all q, p, qn in the range [0, 1].

    See: /docs/docs.pdf Section 1.
    """
    dq = q - qn
    piN = PI*N
    ord0 = np.exp(-piN*dq**2)
    ord1 = np.exp(-piN*(dq - 1)**2 + 2j*piN*p) + np.exp(-piN*(dq + 1)**2 - 2j*piN*p)
    ord2 = np.exp(-piN*(dq - 2)**2 + 4j*piN*p) + np.exp(-piN*(dq + 2)**2 - 4j*piN*p)
    return ord0 + ord1 + ord2


def husimi_torus_state(q, p, psi, N, qbar=QBAR):
    """
    Husimi distribution on the unit torus for the state `psi`.
    """
    qn = (np.arange(0, N) + qbar)/N

    qq, pp = np.meshgrid(q, p)
    zz = (qq + 1j*pp)/2**0.5

    z_psi = np.zeros((p.size, q.size), dtype=complex) # <z|psi>
    for m, qm in enumerate(qn):
        exp_m = np.exp(-1j*PI*N*pp*(qq - 2*qm))
        theta_m = jacobi_theta(qq, pp, N, qm)

        zz_m = (exp_m * theta_m).conj()
        z_psi += zz_m * psi[m]

    out = (2*N)**0.5 * np.abs(z_psi)**2
    return out


def husimi_torus_operator(q, p, rho, N, qbar=QBAR):
    """
    Husimi distribution on the unit torus for a hermitic operator
    (density matrix) `rho`.
    """
    qn = (np.arange(0, N) + qbar)/N

    qq, pp = np.meshgrid(q, p)
    zz = (qq + 1j*pp)/2**0.5

    z_qn = np.zeros((N, p.size, q.size), dtype=complex) # <z|qn>
    for m, qm in enumerate(qn):
        exp_m = np.exp(-1j*PI*N*pp*(qq - 2*qm))
        theta_m = jacobi_theta(qq, pp, N, qm)
        z_qn[m] = (exp_m * theta_m).conj()

    z_rho_z = np.einsum('iqp,ij,jqp->qp', z_qn, rho, z_qn.conj(),
                        optimize=True)

    out = (2*N)**0.5 * z_rho_z
    return np.real(out) # rho is assumed to be hermitic


##
def coherent_state_torus(q, p, N, qbar=QBAR):
    """
    Coherent state on the torus.
    """
    qn = (np.arange(0, N) + qbar)/N

    ket = np.zeros(N, dtype=complex) # |z>
    norm = 0
    for m, qm in enumerate(qn):
        exp_m = np.exp(-1j*PI*N*p*(q - 2*qm))
        theta_m = jacobi_theta(q, p, N, qm)
        ket[m] = exp_m * theta_m
        norm += np.abs(theta_m)**2
    norm = norm**0.5
    ket /= norm
    return ket.reshape((N, 1))


#  @store(path=STORE_DIR)
def coherent_ensemble_torus(f, N, qbar=QBAR, args=()):
    """
    Density matrix for a classical ensemble of coherent states on the unit
    torus written in the position representation.

    See: /docs/docs.pdf Section 2.2.
    """
    qlims = [0, 1]
    plims = [0, 1]
    qn = (np.arange(0, N) + qbar)/N

    # spatial resolution for integration, which is defined by the uncertainty
    # principle on the torus
    Nres = 3*int(np.ceil((4*np.pi*N)**0.5) + 1) # times 3 for good measure
    qs = np.linspace(*qlims, Nres)
    ps = np.linspace(*plims, Nres)
    qq, pp = np.meshgrid(qs, ps)

    # calculate Jacobi theta stuff
    thetas = np.zeros((N, Nres, Nres), dtype=complex)
    for m, qm in enumerate(qn):
        thetas[m] = jacobi_theta(qq, pp, N, qm)
    theta_sum = np.sum(np.abs(thetas)**2, axis=0)

    # evaluate function
    ff = f(qq, pp, *args)

    global calculate_element # avoid AttributeError

    def calculate_element(indices):
        n, m = indices
        out = ff*np.exp(1j*DPI*N*pp*(qn[n]-qn[m]))*thetas[n]*thetas[m].conj()
        out /= theta_sum
        out = simpson(simpson(out, x=qs), x=ps)
        return (n, m), out

    out = np.zeros((N, N), dtype=complex)
    nm = np.arange(0, N)
    offdiag = np.asarray(list(combinations(nm, 2)))
    diag = np.asarray(list(zip(nm, nm)))

    with Pool(None) as p:
        pmap_offdiag = p.map(calculate_element, offdiag)
        pmap_diag = p.map(calculate_element, diag)

    for (n, m), val in pmap_offdiag:
        out[n, m] = val
        out[m, n] = val.conjugate()

    for (n, _), val in pmap_diag:
        out[n, n] = val

    trace = np.real(np.linalg.trace(out))
    print(f'Trace: {trace}')
    if abs(1 - trace) > 1e-4:
        print('WARNING: trace is not 1. Integration limits may be wrong or point density too low.')
    return out


def pure_coherent_torus(q, p, N, qbar=QBAR):
    """
    Pure coherent-state density matrix on the torus.
    """
    ket = coherent_state_torus(q, p, N, qbar)
    return np.outer(ket, ket.conj())
