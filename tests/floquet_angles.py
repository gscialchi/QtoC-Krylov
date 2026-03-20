"""
I want to understand the relevance of the Floquet angles when quantizing
the torus.

This need arises from the following observation:
the quantum evolution as seen from the Husimi distribution is different
depending on the choice of Floquet angles.
    - For Floquet angles (qbar, pbar) = (0, 0), (0.5, 0) its evolution is
      similar to the classical.
    - For Floquet angles (qbar, pbar) = (0, 0.5), (0.5, 0.5) these evolutions
      differ.
(see compare_classical_quantum)
NOTE: the same happens if instead of evolving density matrices I evolve
coherent states.

This difference does not arise from the Husimi distribution itself:
the discrepancies only depend on pbar but the Husimi only depends on qbar.

Does this have something to do with the classical symmetries?
[M. Saraceno, Classical structures in the quantized baker transformation,
 Annals of Physics 199, 37 (1990).]

Indeed some choices of Floquet angles break symmetries (see test_symmetries)
but they are not a one-to-one correlation with those that make the quantum
and classical evolutions diverge.

CONCLUSION:
    Turns out that the discussion about the choice of Floquet angles is highly
    non trivial (discussion with Saraceno). For now, I will be content with a
    choice such that the evolution follows the classical motion, i.e.,
    I could simply set (qbar, pbar) = (0, 0).

    Even so, based on the symmetry observations below, I will set
    (qbar, pbar) = (0.5, 0), since in this case the classical evolution is well
    represented, AND the R and T symmetries are respected.
"""
from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from qtoc_krylov.harper import *
from qtoc_krylov.evolve import *
from qtoc_krylov.misc import *


def compare_classical_quantum():
    """
    Check the time-evolved Husimi distribution for different choices
    of Floquet angles.

    OBSERVATIONS:
        - Classical and quantum evolutions diverge when pbar != 0 for any qbar,
          if the evolution touches the edges of the unit square.
    """
    k = 0.05
    q0, p0 = (0, 0.25)
    s = 0.025
    n_steps = 10

    ## Classical
    N_res = 100
    qlim, plim = [0, 1], [0, 1]

    c_map = partial(c_harper_inv, k) # classical map

    f = partial(periodic_gauss_2D, x0=q0, y0=p0, s=s) # initial classical

    evo_cl = evolve_distribution(c_map, qlim, plim, N_res, f, n_steps)[-1]
    # only care about last one

    ## Quantum
    N_qu = 2**8 # dimension of quantum Hilbert space

    floquet_angles = [0, 0.5] # actual Floquet angles are 2pi times this
    indices = list(product(*[range(len(floquet_angles))]*2))
    q = np.linspace(*qlim, N_res); p = np.linspace(*plim, N_res) # for Husimi

    hus_evos_qu = np.zeros((len(floquet_angles), len(floquet_angles),
                            N_res, N_res))
    for i, j in indices:
        qbar, pbar = floquet_angles[i], floquet_angles[j]

        q_map = q_harper(k, N_qu, qbar=qbar, pbar=pbar) # quantum map

        rho = coherent_ensemble_torus(periodic_gauss_2D, N_qu, qbar=qbar,
                                      args=(q0, p0, s))
        evo_qu = evolve_operator(q_map, rho, n_steps)[-1]
        # only care about last one

        hus_evo_qu = husimi_torus_operator(q, p, evo_qu, N_qu, qbar=qbar)
        hus_evos_qu[i, j, :, :] = hus_evo_qu

    ## Plot
    extent = [0, 1, 0, 1]
    fig, axes = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(6, 6))

    for i, j in indices:
        hus = hus_evos_qu[i, j]
        qbar = floquet_angles[i]
        pbar = floquet_angles[j]

        axes[i, j].imshow(hus, origin='lower', extent=extent,
                          cmap='magma', vmin=0, vmax=None)
        axes[i, j].set_title(r'$(\bar{q}, \bar{p}) = '+f'({qbar},{pbar})' + r'$',
                       size=12)
    fig.suptitle(r'Husimi', size=12)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(evo_cl, origin='lower', extent=extent,
              cmap='magma', vmin=0, vmax=None)
    fig.suptitle(r'Classical', size=12)
    fig.tight_layout()
    plt.show()


def test_symmetries():
    """
    See if particular choices of Floquet angles break symmetries of the
    maps.

    OBSERVATIONS:
        R-Symmetry doesn't break when:
            (qbar, pbar) = (0.5, 0), (0.5, 0.5)

        T-Symmetry doesn't break when:
            (qbar, pbar) = (0, 0.5), (0.5, 0), (0, 0), (0.5, 0.5) (any)

    This is not a one-to-one correlation to what happens in the evolution.
    """
    def commutes(x, y):
        comm = np.abs(x@y - y@x)
        return np.all(np.isclose(comm, 0)), np.max(comm)

    k = 0.05
    N_qu = 2**8 # dimension of quantum Hilbert space

    R = np.flip(np.eye(N_qu), axis=0) # R-symmetry operator
    qq, pp = np.meshgrid(np.arange(0, N_qu),
                         np.arange(0, N_qu)) # for T-symmetry op

    floquet_angles = [0, 0.5] # actual Floquet angles are 2pi times this
    indices = list(product(*[range(len(floquet_angles))]*2))

    for i, j in indices:
        qbar, pbar = floquet_angles[i], floquet_angles[j]

        F = np.exp(-1j * DPI/N_qu * (qq+qbar) * (pp+pbar)) / N_qu**0.5
        T = F.T @ F # antiunitary time reversal (T-symmetry operator)

        q_map = q_harper(k, N_qu, qbar=qbar, pbar=pbar) # quantum map

        ###
        comms, delta = commutes(R, q_map)
        print(f'------- ({qbar}, {pbar}) -------')
        text = f'[R, U] = 0: {comms} (dif = {delta})'
        if comms:
            text += ' <---'
        print(text)

        comms, delta = commutes(T, q_map)
        text = f'[T, U] = 0: {comms} (dif = {delta})'
        if comms:
            text += ' <---'
        print(text)


if __name__ == "__main__":
    compare_classical_quantum()
    test_symmetries()
    ...
