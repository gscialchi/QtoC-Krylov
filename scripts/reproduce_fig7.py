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
N_qu = 2**8
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
ck_pure = krylov_complexity(wave_pure)

## ket
doer_ket.set_args(q=q0, p=p0, N=N_qu)

doer_evolve_k.set_args(u=doer_u, psi=doer_ket, n_steps=n_steps)
evo_ket = doer_evolve_k.doit()

doer_arnoldi_k.set_args(H=doer_u, e0=doer_ket, stop=stop)
kry_ket = doer_arnoldi_k.doit()

wave_ket = krylov_wavefunction_ket(evo_ket, kry_ket)
ck_ket = krylov_complexity(wave_ket)

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

ck_cl = krylov_complexity(wave_cl)

## plot states
which = [0, 17, 55] # which states to show

# points for drawing classical trayectories
# points for drawing classical trayectories
point_map = partial(c_harper, k)
qq0 = 2*np.linspace(*qlim, 2*25); pp0 = 1-qq0
qq0 = np.append(qq0, [0.05/3]); pp0 = np.append(pp0, [0.5])
# ^ an extra point at the hyperbolic point
points = [qq0, pp0]

figname = f'manuscript/Harper_ket_pure_cl_states'
plot_states_ket_pure_cl(kry_cl, kry_ket, kry_pure,
                              qlim, plim, N_cl, N_q=N_qu,
                              points=points, map=point_map,
                              norm_from_k0=True,
                              which=which,
                              #  save=True,
                              savedir=FIG_DIR + figname,
                              saveformat='png')
