# make copies to avoid conflicts
doer_evolve_f = doer_evolve_distribution.copy()
doer_gs = doer_gram_schmidt.copy()
doer_wave = doer_wavefunction_cl.copy()
doer_evolve_rho = doer_evolve_operator.copy()
doer_arnoldi = doer_arnoldi_operator.copy()
doer_rho = doer_rho_gauss_torus.copy()
doer_u = doer_u_harper.copy()

# global parameters
n_steps = 100
stop = n_steps
k = 0.45
q0, p0 = 0.4, 0.5
s = 0.025 # classical standard deviation

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
fk = doer_gs.doit()

doer_wave.set_args(ft=ft, fk=fk, qlim=qlim, plim=plim, N=N_cl)
doer_wave.set_fakeargs(map=map, rho=f, stop=stop)
wave_cl = doer_wave.doit()

ck_cl = krylov_complexity(wave_cl)

## Quantum
Ns = np.asarray([2**5, 2**6, 2**7, 2**8])

cks_qu = np.zeros((len(Ns), n_steps))
for i, N_qu in enumerate(Ns):
    print(f'N_qu={N_qu}')

    h = 1/(2*np.pi*N_qu)
    print(f'h real={h}')

    doer_u.set_args(k=k, N=N_qu)
    doer_rho.set_args(N=N_qu, args=(q0, p0, s))

    doer_evolve_rho.set_args(u=doer_u, rho=doer_rho, n_steps=n_steps)
    rho_qu = doer_evolve_rho.doit()

    doer_arnoldi.set_args(U=doer_u, e0=doer_rho, stop=stop,
                          prod=partial(operator_prod, hbar=h))
    kry_qu = doer_arnoldi.doit()

    wave_qu = krylov_wavefunction_operator(rho_qu, kry_qu, hbar=h)
    cks_qu[i, :] = krylov_complexity(wave_qu)

### compare
figname = f'manuscript/Harper_complexity_correspondence'
plot_complexity_correspondence(ck_cl, cks_qu, Ns,
                               nlogyticks=2,
                               #  save=True,
                               #  show=False,
                               savedir=FIG_DIR + figname)
