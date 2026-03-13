doer_evolve_f = doer_evolve_distribution.copy()
doer_gs = doer_gram_schmidt.copy()
doer_wave = doer_wavefunction_cl.copy()
doer_evolve_rho = doer_evolve_operator.copy()
doer_arnoldi = doer_arnoldi_operator.copy()
doer_rho = doer_rho_gauss.copy()
doer_u = doer_u_ho.copy()

# global parameters
n_steps = 100 # number of time-steps
stop = n_steps
q0, p0 = 1, 0
s = 0.1 # classical standard deviation

# fix dt such that in one timestep there is some overlap (if a<~1)
a = 0.5
dt = a * 2*s/(q0**2 + p0**2)**0.5

## Classical
N = 300 # NOTE: make sure is enough to resolve packet!

r = (q0**2+p0**2)**0.5 + 5*s # 5 sigma space between packet and border
qlim = [-r, r]
plim = qlim

f = partial(gauss_2D, x0=q0, y0=p0, s=s)
map = partial(harmonic_map_inv, dt=dt)

doer_evolve_f.set_args(map=map,
                       qlim=qlim, plim=plim, N=N,
                       rho=f, n_steps=n_steps)
ft = doer_evolve_f.doit()

doer_gs.set_args(ft=ft, qlim=qlim, plim=plim, N=N, stop=stop)
doer_gs.set_fakeargs(map=map, rho=f)
fk = doer_gs.doit()

doer_wave.set_args(ft=ft, fk=fk, qlim=qlim, plim=plim, N=N)
doer_wave.set_fakeargs(map=map, rho=f, stop=stop)
wave_cl = doer_wave.doit()

u_cl = krylov_propagator_wave(wave_cl)

## Quantum
hs = 1/np.asarray([2**4, 2**5, 2**6, 2**7])#, 2**8])
fid = 1-1e-9 # fidelity of truncated quantum coherent state

doer_rho.set_args(args=(q0, p0, s))

us_qu = []
for i, h in enumerate(hs):
    nsig = 5 # n sigma padding for integ lims & cutoffs
    qlims = (q0 - nsig*s, q0 + nsig*s)
    plims = (p0 - nsig*s, p0 + nsig*s)
    cutoffs = cutoffs_for_coherent_ensemble(fid, qlims, plims, hbar=h)

    doer_rho.set_args(qlims=qlims, plims=plims, cutoffs=cutoffs, hbar=h)
    doer_u.set_args(dt=dt, cutoffs=cutoffs, hbar=h)

    doer_evolve_rho.set_args(u=doer_u, rho=doer_rho,
                             n_steps=n_steps)
    rho_evo = doer_evolve_rho.doit()

    doer_arnoldi.set_args(U=doer_u, e0=doer_rho,
                          stop=stop,
                          prod=partial(operator_prod, hbar=h))
    rho_kry = doer_arnoldi.doit()

    u_qu = krylov_propagator_kry(doer_u._doit(), rho_kry, hbar=h)
    us_qu.append(u_qu)

###
figname = 'manuscript/HO_sequences_correspondence'
plot_sequences_correspondence(u_cl, us_qu, hs,
                              #  save=True,
                              savedir=FIG_DIR+figname)
