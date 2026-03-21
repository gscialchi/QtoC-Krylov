from functools import partial
import argparse

import numpy as np

from qtoc_krylov.harmonic import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
from qtoc_krylov.utilities.paths import *
from qtoc_krylov.utilities.doer import Doer
from qtoc_krylov.utilities.store import store
from qtoc_krylov.utilities.config import get_config

from plotters import (plot_sequences_correspondence,
                      plot_complexity_correspondence)


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=CONFIG_DIR+'/defaults.yml')
args = parser.parse_args()

configs = get_config(args.config)

####
DISABLE_DOER = configs['DISABLE_DOER']
DISABLE_STORE = configs['DISABLE_STORE']

#### Wrap store on some costly operators
if not DISABLE_STORE:
    coherent_ensemble = store(path=STORE_DIR)(coherent_ensemble)


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=DOER_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=DOER_DIR,
               disabled=DISABLE_DOER)

doer_wave = Doer(krylov_wavefunction_cl, ignore_args=['ft', 'fk'],
                 path=DOER_DIR, disabled=DISABLE_DOER)

doer_evolve_rho = Doer(evolve_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_rho = Doer(coherent_ensemble, args={'f': gauss_2D}, disabled=DISABLE_DOER)

doer_u = Doer(u_harmonic, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = configs['HO_figs_1_and_2_n_steps']
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

hs = np.asarray(configs['HO_figs_1_and_2_hbars']) # values of hbar
fid = configs['HO_fid'] # fidelity of truncated quantum coherent state

# integration limits that will be used to calculate the quantum initial state
nsig = configs['HO_N_sigma_res'] # n sigma padding for integ lims & cutoffs
q_integlims = (q0 - nsig*s, q0 + nsig*s)
p_integlims = (p0 - nsig*s, p0 + nsig*s)

doer_rho.set_args(args=(q0, p0, s)) # params for initial quantum distribution


#### Classical
doer_evolve_f.set_args(map=map,
                       qlim=qlim, plim=plim, N=N_res,
                       rho=f, n_steps=n_steps)
evo_cl = doer_evolve_f.doit() # get evolution

doer_gs.set_args(ft=evo_cl, qlim=qlim, plim=plim, N=N_res, stop=stop)
doer_gs.set_fakeargs(map=map, rho=f)
kry_cl = doer_gs.doit() # get Krylov via Gram-Schmidt

doer_wave.set_args(ft=evo_cl, fk=kry_cl, qlim=qlim, plim=plim, N=N_res)
doer_wave.set_fakeargs(map=map, rho=f, stop=stop)
wave_cl = doer_wave.doit() # get Krylov wavefunction

u_cl = krylov_propagator_wave(wave_cl) # propagator in Krylov basis
ck_cl = krylov_complexity(wave_cl) # Krylov complexity


#### Quantum
us_qu = [] # propagators in Krylov basis
cks_qu = np.zeros((len(hs), n_steps)) # Krylov complexities

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

    doer_evolve_rho.set_args(u=doer_u, rho=doer_rho, n_steps=n_steps)

    # Calculate quantum
    evo_qu = doer_evolve_rho.doit() # evolution

    kry_qu = doer_arnoldi.doit() # quantum Krylov basis via Arnoldi

    wave_qu = krylov_wavefunction_operator(evo_qu, kry_qu, hbar=h) # wavefunction

    cks_qu[i, :] = krylov_complexity(wave_qu) # Krylov complexity

    u_qu = krylov_propagator_kry(doer_u._doit(), kry_qu, hbar=h) # propagator
    us_qu.append(u_qu)


#### Plot
figname = 'Figure_1'
plot_sequences_correspondence(u_cl, us_qu, up_to=100,
                              save=configs['SAVE_FIGURES'],
                              savedir=FIG_DIR + figname)

figname = 'Figure_2'
plot_complexity_correspondence(ck_cl, cks_qu, up_to=200,
                               save=configs['SAVE_FIGURES'],
                               savedir=FIG_DIR + figname)
