import numpy as np


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


def evolve_distribution(map, qlim, plim, N, rho, n_steps):
    """
    Evolve a distribution `rho` on phase space according to a
    measure-preserving map.
    """
    q = np.linspace(*qlim, N)
    p = np.linspace(*plim, N)
    qq, pp = np.meshgrid(q, p)

    rho0 = rho(qq, pp)
    out = np.zeros((n_steps, N, N))

    out[0] = rho0
    for n in range(1, n_steps):
        qq_, pp_ = map(qq, pp)
        qq, pp = qq_, pp_
        out[n] = rho(qq, pp)
    return out


def evolve_state(u, psi, n_steps):
    """ Evolve a ket with the unitary for `n_steps` time steps. """
    out = [psi]
    psi_t = psi
    for n in range(n_steps-1):
        psi_t = u @ psi_t
        out.append(psi_t)
    return np.asarray(out)


def evolve_operator(u, rho, n_steps):
    """ Evolve an operator with a unitary for `n_steps` time steps. """
    out = [rho]
    rho_t = rho
    for n in range(n_steps-1):
        rho_t = u @ rho_t @ u.conj().T
        out.append(rho_t)
    return np.asarray(out)
