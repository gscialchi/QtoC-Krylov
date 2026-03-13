# make copies to avoid conflicts
doer_evolve_f = doer_evolve_distribution.copy()
doer_evolve_rho = doer_evolve_operator.copy()
doer_evolve_k = doer_evolve_ket.copy()
doer_wave = doer_wavefunction_cl.copy()

doer_gs = doer_gram_schmidt.copy()
doer_arnoldi = doer_arnoldi_operator.copy()
doer_arnoldi_k = doer_arnoldi_ket.copy()

doer_pure = doer_pure_coherent_torus.copy()
doer_ket = doer_ket_coherent_torus.copy()

doer_u = doer_u_harper.copy()

# global parameters
n_steps = 200
stop = n_steps
k = 0.05
q0, p0 = 0.4, 0.5

##
Ns = np.asarray([2**5, 2**6, 2**7, 2**8, 2**9])[1:]

cks_cl = np.zeros((len(Ns), n_steps))
cks_pure = np.zeros_like(cks_cl)
cks_ket = np.zeros_like(cks_cl)

for i, N_qu in enumerate(Ns):
    h = 1/(2*np.pi*N_qu)
    s = h**0.5 # classical standard deviation matching quantum

    doer_u.set_args(k=k, N=N_qu)

    ## pure state
    doer_pure.set_args(q=q0, p=p0, N=N_qu)

    doer_evolve_rho.set_args(u=doer_u, rho=doer_pure, n_steps=n_steps)
    evo_pure = doer_evolve_rho.doit()

    doer_arnoldi.set_args(U=doer_u, e0=doer_pure,
                          stop=stop,
                          prod=partial(operator_prod, hbar=h))
    kry_pure = doer_arnoldi.doit()

    wave_pure = krylov_wavefunction_operator(evo_pure, kry_pure, hbar=h)
    cks_pure[i, :] = krylov_complexity(wave_pure)

    ## ket
    doer_ket.set_args(q=q0, p=p0, N=N_qu)

    doer_evolve_k.set_args(u=doer_u, psi=doer_ket, n_steps=n_steps)
    evo_ket = doer_evolve_k.doit()

    doer_arnoldi_k.set_args(H=doer_u, e0=doer_ket, stop=stop)
    kry_ket = doer_arnoldi_k.doit()

    wave_ket = krylov_wavefunction_ket(evo_ket, kry_ket)
    cks_ket[i, :] = krylov_complexity(wave_ket)

    ## Classical
    N_cl = 300
    qlim, plim = [0, 1], [0, 1]

    f = partial(periodic_gauss_2D, x0=q0, y0=p0, s=s)
    map = partial(c_harper_inv, k)

    doer_evolve_f.set_args(map=map,
                           qlim=qlim, plim=plim, N=N_cl,
                           rho=f, n_steps=n_steps)
    ft = doer_evolve_f.doit()

    doer_gs.set_args(ft=ft, qlim=qlim, plim=plim, N=N_cl, stop=stop)
    doer_gs.set_fakeargs(map=map, rho=f)
    kry_cl = doer_gs.doit()

    doer_wave.set_args(ft=ft, fk=kry_cl, qlim=qlim, plim=plim, N=N_cl)
    doer_wave.set_fakeargs(map=map, rho=f, stop=stop)
    wave_cl = doer_wave.doit()
    cks_cl[i, :] = krylov_complexity(wave_cl)

## plot complexity
figname = f'manuscript/Harper_ket_pure_cl_complexity_limit'
plot_complexity_ket_pure_cl_limit(cks_cl, cks_ket, cks_pure, Ns,
                                  plot_dif=True, inset_pos=[0.1125, 0.475, 0.5, 0.5],
                                  #  save=True,
                                  savedir=FIG_DIR + figname)
