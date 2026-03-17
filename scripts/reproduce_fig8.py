from functools import partial

import numpy as np

from qtoc_krylov.harper import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
from qtoc_krylov.utilities.paths import *
from qtoc_krylov.utilities.doer import Doer

from plotters import plot_states_ket_pure_cl


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=CALC_DIR)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=CALC_DIR)

doer_arnoldi = Doer(arnoldi_FO_operator, path=CALC_DIR)

doer_lanczos = Doer(lanczos_FO_ket, path=CALC_DIR)

doer_ket = Doer(coherent_state_torus)

doer_pure = Doer(pure_coherent_torus)

doer_u = Doer(q_harper)


#### Calculation parameters
n_steps = 60 # number of time-steps
stop = n_steps # when to stop calculating Krylov

k = 0.05 # map parameter
q0, p0 = 0.4, 0.5 # center of initial state in phase space

# number of points to evaluate classical distribution
N_res = 300 # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
qlim, plim = [0, 1], [0, 1] # the whole unit torus

N_qu = 2**8 # quantum dimension


#### Calculate
h = 1/(2*np.pi*N_qu)
s = h**0.5 # classical standard deviation matching quantum

### Quantum
doer_u.set_args(k=k, N=N_qu)

## pure density matrix
doer_pure.set_args(q=q0, p=p0, N=N_qu)

doer_arnoldi.set_args(U=doer_u, e0=doer_pure,
                      stop=stop,
                      prod=partial(operator_prod, hbar=h))
kry_pure = doer_arnoldi.doit()

## ket
doer_ket.set_args(q=q0, p=p0, N=N_qu)

doer_lanczos.set_args(H=doer_u, e0=doer_ket, stop=stop)
kry_ket = doer_lanczos.doit()

### Classical
f = partial(periodic_gauss_2D, x0=q0, y0=p0, s=s) # initial classical distribution
map = partial(c_harper_inv, k) # inverse harper map

doer_evolve_f.set_args(map=map,
                       qlim=qlim, plim=plim, N=N_res,
                       rho=f, n_steps=n_steps)
evo_cl = doer_evolve_f.doit()

doer_gs.set_args(ft=evo_cl, qlim=qlim, plim=plim, N=N_res, stop=stop)
doer_gs.set_fakeargs(map=map, rho=f)
kry_cl = doer_gs.doit() # get Krylov via Gram-Schmidt


#### Plot
which = [0, 17, 55] # which states to show

# points for drawing classical trayectories
point_map = partial(c_harper, k)
qq0 = 2*np.linspace(*qlim, 2*25); pp0 = 1-qq0
qq0 = np.append(qq0, [0.05/3]); pp0 = np.append(pp0, [0.5])
# ^ an extra point at the hyperbolic point
points = [qq0, pp0]

figname = 'Figure_8'
plot_states_ket_pure_cl(kry_cl, kry_ket, kry_pure,
                        qlim, plim, N_res, N_q=N_qu,
                        points=points, map=point_map,
                        norm_from_k0=True,
                        which=which,
                        save=True,
                        savedir=FIG_DIR + figname,
                        saveformat='png') # pdf can be large
