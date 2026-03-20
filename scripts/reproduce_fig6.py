from functools import partial

import numpy as np

from qtoc_krylov.harper import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
from qtoc_krylov.utilities.paths import *
from qtoc_krylov.utilities.doer import Doer

from plotters import plot_states_correspondence


#### Setup Doers for data saving & retrieval
DISABLE_DOER = False
# ^ if True, bypasses Doer functionality altogether. Nothing is loaded or saved

doer_evolve_f = Doer(evolve_distribution, path=CALC_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=CALC_DIR,
               disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=CALC_DIR, disabled=DISABLE_DOER)

doer_rho = Doer(coherent_ensemble_torus, args={'f': periodic_gauss_2D},
                disabled=DISABLE_DOER)

doer_u = Doer(q_harper, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = 60 # number of time-steps
stop = n_steps # when to stop calculating Krylov

k = 0.05 # map parameter
q0, p0 = 0.4, 0.5 # center of initial state in phase space
s = 0.025 # standard deviation for classical distribution

# number of points to evaluate classical distribution
N_res = 300 # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
qlim, plim = [0, 1], [0, 1] # the whole unit torus

f = partial(periodic_gauss_2D, x0=q0, y0=p0, s=s) # initial classical distribution
map = partial(c_harper_inv, k) # inverse harper map

N_qus = np.asarray([2**6, 2**11]) # values of quantum dimension

doer_rho.set_args(args=(q0, p0, s)) # parameters for initial quantum distribution


#### Classical
doer_evolve_f.set_args(map=map,
                       qlim=qlim, plim=plim, N=N_res,
                       rho=f, n_steps=n_steps)
evo_cl = doer_evolve_f.doit() # get evolution

doer_gs.set_args(ft=evo_cl, qlim=qlim, plim=plim, N=N_res, stop=stop)
doer_gs.set_fakeargs(map=map, rho=f)
kry_cl = doer_gs.doit() # get Krylov via Gram-Schmidt


#### Quantum
krys_qu = [] # Krylov states for each N_qu
for i, N_qu in enumerate(N_qus):
    h = 1/(2*np.pi*N_qu) # hbar correspoding to N_qu

    # Apply parameters for given value of N_qu
    doer_rho.set_args(N=N_qu)

    doer_u.set_args(k=k, N=N_qu)

    doer_arnoldi.set_args(U=doer_u, e0=doer_rho, stop=stop,
                          prod=partial(operator_prod, hbar=h))

    # Calculate quantum
    kry_qu = doer_arnoldi.doit() # quantum Krylov basis via Arnoldi

    krys_qu.append(kry_qu)


#### Plot
which = [0, 1, 2, 3, 15, 35, 55] # which states to show

# points for drawing classical trayectories
point_map = partial(c_harper, k)
qq0 = 2*np.linspace(*qlim, 2*25); pp0 = 1-qq0
qq0 = np.append(qq0, [0.05/3]); pp0 = np.append(pp0, [0.5])
# ^ an extra point at the hyperbolic point
points = [qq0, pp0]

figname = 'Figure_6'
# calculation of Husimi distributions is inside plotter
plot_states_correspondence(kry_cl, krys_qu,
                           qlim, plim, N_res,
                           Nqus=N_qus,
                           points=points, map=point_map,
                           norm_from_k0=True,
                           which=which,
                           save=True,
                           #  show=False,
                           savedir=FIG_DIR + figname,
                           saveformat='png') # pdf can be large
