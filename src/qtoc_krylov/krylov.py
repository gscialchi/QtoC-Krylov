import numpy as np


## inner products
def l2_prod(f, g, x, y):
    """
    L2 inner product of the distributions f and g.
    """
    return np.trapezoid(np.trapezoid(f*g, x=x), x=y)


def l2_norm(f, x, y):
    """
    L2 norm of the distribution f.
    """
    return l2_prod(f, f, x, y)**0.5


def frobenius_prod(op1, op2):
    prod = np.trace(op1.conj().T @ op2)
    return prod


def operator_prod(op1, op2, hbar=1):
    return frobenius_prod(op1, op2)/(2*np.pi*hbar)


def operator_norm(op, hbar=1):
    return np.sqrt(np.real(operator_prod(op, op, hbar)))


def ket_prod(ket1, ket2):
    return (ket1.conj().T @ ket2)[0][0]


## Arnoldi, Lanczos & Gram-Schmidt
def arnoldi_FO_operator(U: np.ndarray,
                        e0: np.ndarray,
                        stop: int = None,
                        prod = frobenius_prod,
                        ):
    """
    Full re-orthonormalization Arnoldi algorithm for operators.
    """
    if isinstance(U, Qobj):
        U = U.full()
    if isinstance(e0, Qobj):
        e0 = e0.full()

    stop = stop or np.inf
    print(f'stop {stop}.')

    def L(op): return U @ op @ U.conj().T
    expected_stop = U.shape[0]**2 - U.shape[0] + 1

    def norm(e): return np.sqrt(np.real(prod(e, e)))

    e_arr = [e0 / norm(e0)]
    n = 1

    bn = np.inf
    while bn > 1e-7:
        An = L(e_arr[n - 1])

        for _ in range(2):
            for en in e_arr:
                An -= en * prod(en, An)

        bn = norm(An)
        e_arr.append(1 / bn * An)

        if n >= stop -1:
            break
        n += 1
    print(f'stopped at n={n}. Max. expected stop={expected_stop}.')

    e_arr = np.asarray(e_arr)
    return e_arr


def lanczos_FO_state(H: np.ndarray,
                     e0: np.ndarray,
                     stop: int = None,
                     prod = ket_prod,
                     ):
    """
    Full re-orthonormalization Lanczos algorithm for states.
    """
    stop = stop or H.shape[0]
    print(f'stop {stop}.')

    def L(psi): return H @ psi

    def norm(e): return np.sqrt(np.real(prod(e, e)))

    e_arr = [e0 / norm(e0)]
    n = 1

    bn = np.inf
    while bn > 1e-7:
        An = L(e_arr[n - 1])

        for _ in range(2):
            for en in e_arr:
                An -= en * prod(en, An)

        bn = norm(An)
        Kn = 1 / bn * An
        an = prod(Kn, L(Kn))

        e_arr.append(Kn)
        if n >= stop -1: # ensures that there's always 'stop' elements in basis
            break
        n += 1
    print(f'stopped at n={n}.')

    e_arr = np.array(e_arr)
    return  e_arr


def gram_schmidt_ft(ft, qlim, plim, N, stop=100):
    """
    Applies the Gram-Schmidt method to the set of evolved states {ft}.
    This yields the same results as the Arnoldi iteration.
    See ../test/arnoldi_vs_gram_schmidt.py.
    """
    q = np.linspace(*qlim, N)
    p = np.linspace(*plim, N)

    out = np.zeros((stop+1, *ft[0].shape))
    out[0] = ft[0].copy()/l2_norm(ft[0], q, p)

    n = 1
    bn = np.inf
    while bn > 1e-7:
        u = ft[n].copy()
        for _ in range(2):
            for v in out[:n]:
                u -= l2_prod(u, v, q, p) * v

        bn = l2_norm(u, q, p)
        out[n] = u/bn

        if n >= stop - 1:
            break
        n += 1
    print(f'stopped at n={n}.')
    return out[:n+1] # remove trailing zeros


##
def krylov_wavefunction_ket(ket_evo, krylov):
    """
    Compute <k_n|psi_t> for all t, n.
    """
    return np.einsum('tki,nki->tn', ket_evo.conj(), krylov)


def krylov_wavefunction_operator(rho_evo, krylov, hbar=1):
    """
    Compute (k_n|rho_t) for all t, n.
    """
    # normalize initial state is for Krylov wavefunction
    rho_evo_normed = rho_evo/operator_norm(rho_evo[0], hbar=hbar)

    # the operator product defined above
    out = np.einsum('tki,nik->tn', rho_evo_normed, krylov)
    out /= (2*np.pi*hbar)
    return out


##
def krylov_propagator_wave(wave):
    """
    Compute the matrix representation of the propagator in the Krylov basis
    from the wavefunction.

    See: Notebook B (orange, beach), pages 103-104.1.
    """
    nt, nk = wave.shape
    n = min(nt, nk) - 1 # to make the matrix square

    # beta is simply the transpose of the wavefunction evolution matrix
    beta = wave[:n,:n].T
    alpha = np.linalg.inv(beta)
    return beta[:-1, 1:] @ alpha[:-1, :-1]


def krylov_propagator_kry(u, kry, hbar):
    """
    Compute the matrix representation of the propagator in the Krylov basis
    from the Krylov states.
    """
    #  nk = len(kry)
    #  out = np.zeros((nk, nk))

    #  def L(op): return u @ op @ u.conj().T

    #  for m in range(nk):
        #  km = kry[m]
        #  for n in range(m-1, nk):
            #  kn = kry[n]
            #  out[m, n] = prod(km, L(kn))

    # the computation below is equivalent to the above
    kry_dag = np.transpose(kry.conj(), axes=(0, 2, 1)) # dag every k inside
    out = np.einsum('mab,bi,nij,ja->mn', kry_dag, u, kry, u.conj().T,
                    optimize=True)/(2*np.pi*hbar)
    # this assumes the operator inner product defined before
    return out


##
def krylov_complexity(wave):
    n = np.arange(wave.shape[1])
    return np.einsum('n,tn->t', n, np.abs(wave)**2)
