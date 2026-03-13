import os
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from scipy.integrate import simpson

from utilities import store

STORE_PATH = os.path.dirname(os.path.realpath(__file__))
STORE_PATH += f'/store/{os.path.basename(__file__)[:-3]}/'

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


##
def evolve_map(map, n_steps,
               points=None,
               n_evos=None, qlim=[0, 1], plim=[0, 1]):
    """
    Concurrently run `n_evos`x`n_evos` evolutions from the map with
    initial conditions uniformly chosen for `n_steps` iterations of the map.
    """
    if points is None:
        # initial conditions uniformly distributed
        q0 = np.linspace(*qlim, n_evos)
        p0 = np.linspace(*plim, n_evos)
        qq, pp = np.meshgrid(q0, p0)

        q_out = np.zeros((n_steps, n_evos**2))
        p_out = q_out.copy()
    else:
        qq, pp = points

        q_out = np.zeros((n_steps, qq.size))
        p_out = q_out.copy()

    # set of initial conditions
    q_out[0,:] = qq.flatten()
    p_out[0,:] = pp.flatten()

    for n in range(1, n_steps):
        q_out[n,:], p_out[n,:] = map(q=q_out[n-1,:], p=p_out[n-1,:])
    return q_out, p_out


def evolve_distribution(map, q, p, psi, n_steps):
    """
    Evolve a distribution `psi` on phase space according to a
    measure-preserving map of the form
        q_ = q + f(p)
        p_ = p + g(q_)
    """
    qq, pp = np.meshgrid(q, p)
    psi0 = psi(qq, pp)
    out = np.zeros((n_steps, *psi0.shape))

    out[0] = psi0
    for n in range(1, n_steps):
        qq_, pp_ = map(qq, pp)
        qq, pp = qq_, pp_
        out[n] = psi(qq, pp)
    return out


### Quantum maps
@store(path=STORE_PATH)
def q_harper(k, N, qbar=0.5, pbar=0.5):
    """
    Unitary for the quantum Harper map. `qbar` and `pbar` define
    the Floquet angles, such that
        |q+1> = exp(2i pi pbar)|q> and |p+1> = exp(-2i pi qbar)|p>.
    """
    qn = np.arange(0, N) + qbar # n + qbar
    pn = np.arange(0, N) + pbar # m + pbar

    qq, pp = np.meshgrid(qn, pn)

    ff = np.exp(1j * DPI/N * qq * pp) / N**0.5

    v_nm = np.exp(-1j*N*k*np.cos(DPI/N * qq)) * ff
    v_mn = np.exp(-1j*N*k*np.cos(DPI/N * pp)) * ff.conj()

    u = v_nm.T @ v_mn
    return u


## Husimi distribution on the unit torus
def jacobi_theta(q, p, N, qn):
    r"""
    (Controlled, see ../test/torus_husimi_overflows.py)
    Approximated Jacobi Theta function by selecting a good cutoff
    for the sum. It is accurate to within an atol of at least 1e-9/N
    for all q, p, qn in the range [0, 1]. See ../test/approx_jacobi_theta.py.

    Compared to the actual Jacobi Theta, this is equal to
        \exp(-N \pi (q - qn)^2) * jacobi_theta(q, p, N, qn).
    """
    dq = q - qn
    piN = PI*N
    ord0 = np.exp(-piN*dq**2)
    ord1 = np.exp(-piN*(dq - 1)**2 + 2j*piN*p) + np.exp(-piN*(dq + 1)**2 - 2j*piN*p)
    ord2 = np.exp(-piN*(dq - 2)**2 + 4j*piN*p) + np.exp(-piN*(dq + 2)**2 - 4j*piN*p)
    return ord0 + ord1 + ord2


def husimi_torus_state(q, p, psi, qbar, N):
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


def husimi_torus_operator(q, p, rho, qbar, N):
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
def coherent_state_torus(q, p, qbar, N):
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
    return ket


@store(path=STORE_PATH)
def coherent_ensemble_torus(f, N, qbar, args=()):
    """
    Density matrix for a classical ensemble of coherent states on the unit
    torus written in the position representation.

    See: notebook B (orange, beach), pages 92-93.
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
