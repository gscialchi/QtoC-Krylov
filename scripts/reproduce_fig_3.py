from functools import partial
import argparse

import numpy as np

from qtoc_krylov.harmonic import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
import qtoc_krylov.utilities.paths as paths
from qtoc_krylov.utilities.doer import Doer
from qtoc_krylov.utilities.store import store
from qtoc_krylov.utilities.config import get_config

from plotters import plot_states_correspondence


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=paths.CONFIG_DIR+'/defaults.yml')
args = parser.parse_args()

configs = get_config(args.config)

####
DISABLE_DOER = configs['DISABLE_DOER']
DOER_DIR = configs['DOER_DIR']
if DOER_DIR == 'default':
    DOER_DIR = paths.DOER_DIR

DISABLE_STORE = configs['DISABLE_STORE']
STORE_DIR = configs['STORE_DIR']
if STORE_DIR == 'default':
    STORE_DIR = paths.STORE_DIR

#### Wrap store on some costly operators
if not DISABLE_STORE:
    coherent_ensemble = store(path=STORE_DIR)(coherent_ensemble)


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=DOER_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=DOER_DIR,
               disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_rho = Doer(coherent_ensemble, args={'f': gauss_2D}, disabled=DISABLE_DOER)

doer_u = Doer(u_harmonic, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = configs['HO_fig_3_n_steps']
stop = n_steps # when to stop calculating Krylov

q0 = configs['HO_q0']
p0 = configs['HO_p0']
s = configs['HO_s']

# fix dt such that in one timestep there is some overlap (if a<~1)
a = configs['HO_a']
dt = a * 2*s/(q0**2 + p0**2)**0.5

N_res = configs['HO_N_res']

# define limits of phase space where the distribution is evaluated
r = (q0**2+p0**2)**0.5 + configs['HO_N_sigma_res']*s
qlim = [-r, r]
plim = qlim

f = partial(gauss_2D, x0=q0, y0=p0, s=s) # initial classical distribution
map = partial(harmonic_map_inv, dt=dt) # inverse harmonic map

hs = np.asarray(configs['HO_fig_3_hbars']) # values of hbar
fid = configs['HO_fid'] # fidelity of truncated quantum coherent state

# integration limits that will be used to calculate the quantum initial state
nsig = configs['HO_N_sigma_res'] # n sigma padding for integ lims & cutoffs
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
FIG_DIR = configs['FIG_DIR']
if FIG_DIR == 'default':
    FIG_DIR = paths.FIG_DIR

which = configs['HM_fig_3_states'] # which states to show

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
                           save=configs['SAVE_FIGURES'],
                           #  show=False,
                           usetex=configs['FIGURES_USETEX'],
                           savedir=FIG_DIR + figname,
                           saveformat='png', # pdf can be large
                           show=configs['SHOW_FIGURES'])
