# make copies to avoid conflicts
doer_evolve_f = doer_evolve_distribution.copy()
doer_gs = doer_gram_schmidt.copy()
doer_evolve_rho = doer_evolve_operator.copy()
doer_arnoldi = doer_arnoldi_operator.copy()
doer_rho = doer_rho_gauss_torus.copy()
doer_u = doer_u_harper.copy()

# global parameters
n_steps = 100
stop = n_steps
k = 0.05
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

## Quantum
#  hs = 1/np.asarray([2**8, 2**13])
Ns = np.asarray([2**6, 2**11])

krys = []
#  for i, h in enumerate(hs):
for i, N_qu in enumerate(Ns):
    #  N_qu = int(1/(2*np.pi*h)) + 1
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

    krys.append(kry_qu)

### compare
which = [0, 1, 2, 3, 15, 35, 55] # which states to show

# points for drawing classical trayectories
point_map = partial(c_harper, k)
qq0 = 2*np.linspace(*qlim, 2*25); pp0 = 1-qq0
qq0 = np.append(qq0, [0.05/3]); pp0 = np.append(pp0, [0.5])
# ^ an extra point at the hyperbolic point
points = [qq0, pp0]

figname = f'manuscript/Harper_states_correspondence'
plot_states_correspondence(fk, krys, qlim, plim, N_cl, Nqus=Ns,
                           points=points, map=point_map,
                           norm_from_k0=True,
                           which=which,
                           #  save=True,
                           #  show=False,
                           savedir=FIG_DIR + figname,
                           saveformat='png')
