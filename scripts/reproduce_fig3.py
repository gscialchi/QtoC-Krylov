from functools import partial

import numpy as mp

from qtoc_krylov.harmonic import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
from qtoc_krylov.utilities.paths import *
from qtoc_krylov.utilities.doer import Doer

from plotters import plot_states_correspondence


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=CALC_DIR)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=CALC_DIR)

doer_arnoldi = Doer(arnoldi_FO_operator, path=CALC_DIR)

doer_rho = Doer(coherent_ensemble, args={'f': gauss_2D})

doer_u = Doer(u_harmonic)


#### Calculation parameters
n_steps = 60 # number of time-steps
stop = n_steps # when to stop calculating Krylov

q0, p0 = 1, 0 # center of position initial state in phase space
s = 0.1 # standard deviation for classical distribution

# fix dt such that in one timestep there is some overlap (if a<~1)
a = 0.5
dt = a * 2*s/(q0**2 + p0**2)**0.5

# number of points to evaluate classical distribution
N_res = 300 # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
r = (q0**2+p0**2)**0.5 + 5*s # 5 sigma space between packet center and limits
qlim = [-r, r]
plim = qlim

f = partial(gauss_2D, x0=q0, y0=p0, s=s) # initial classical distribution
map = partial(harmonic_map_inv, dt=dt) # inverse harmonic map

hs = 1/np.asarray([2**5, 2**8]) # values of hbar to evaluate at
fid = 1-1e-9 # fidelity of truncated quantum coherent state

# integration limits that will be used to calculate the quantum initial state
nsig = 5 # n sigma padding for integration lims & cutoffs
q_integlims = (q0 - nsig*s, q0 + nsig*s)
p_integlims = (p0 - nsig*s, p0 + nsig*s)

doer_rho.set_args(args=(q0, p0, s)) # set parameters for initial quantum distribution


#### Classical
doer_evolve_f.set_args(map=map,
                       qlim=qlim, plim=plim, N=N_res,
                       rho=f, n_steps=n_steps)
evo_cl = doer_evolve_f.doit() # get evolution

doer_gs.set_args(ft=evo_cl, qlim=qlim, plim=plim, N=N_res, stop=stop)
doer_gs.set_fakeargs(map=map, rho=f)
kry_cl = doer_gs.doit() # get Krylov via Gram-Schmidt


#### Quantum
krys_qu = [] # Krylov states for each hbar
cutoffs_list = [] # keep track of cutoffs used for each hbar
for i, h in enumerate(hs):
    # find where to cutoff energy levels for the given fidelity, hbar and lims
    cutoffs = cutoffs_for_coherent_ensemble(fid, q_integlims, p_integlims,
                                            hbar=h)

    # Apply parameters for given value of hbar
    doer_rho.set_args(qlims=q_integlims, plims=p_integlims, cutoffs=cutoffs,
                      hbar=h)

    doer_u.set_args(dt=dt, cutoffs=cutoffs, hbar=h)

    doer_arnoldi.set_args(U=doer_u, e0=doer_rho,
                          stop=stop,
                          prod=partial(operator_prod, hbar=h))

    # Calculate quantum
    kry_qu = doer_arnoldi.doit() # quantum Krylov basis via Arnoldi

    krys_qu.append(kry_qu)
    cutoffs_list.append(cutoffs)


#### Plot
which = [0, 1, 2, 3, 15, 35, 55] # which states to show

# points for drawing classical trayectories
point_map = partial(harmonic_map, dt=dt)
qq0 = 2*np.linspace(*qlim, 2*15); pp0 = np.zeros_like(qq0)
points = [qq0, pp0]

figname = 'Figure_3'
# calculation of Husimi distributions is inside plotter
plot_states_correspondence(kry_cl, krys_qu,
                           qlim, plim, N_res,
                           hbars=hs,
                           points=points, map=point_map,
                           norm_from_k0=True,
                           which=which, cutoffs=cutoffs_list,
                           save=True,
                           #  show=False,
                           savedir=FIG_DIR + figname,
                           saveformat='png') # pdf can be large
