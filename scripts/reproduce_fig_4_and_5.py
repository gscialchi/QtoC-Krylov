from functools import partial
import argparse

import numpy as np

from qtoc_krylov.harper import *
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
    q_harper = store(path=STORE_DIR)(q_harper)

    coherent_ensemble_torus = store(path=STORE_DIR)(coherent_ensemble_torus)


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=DOER_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=DOER_DIR,
               disabled=DISABLE_DOER)

doer_wave = Doer(krylov_wavefunction_cl, ignore_args=['ft', 'fk'],
                 path=DOER_DIR, disabled=DISABLE_DOER)

doer_evolve_rho = Doer(evolve_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_rho = Doer(coherent_ensemble_torus, args={'f': periodic_gauss_2D},
                disabled=DISABLE_DOER)

doer_u = Doer(q_harper, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = configs['HM_fig_4_and_5_n_steps']
stop = n_steps # when to stop calculating Krylov

k = configs['HM_k']
q0 = configs['HM_q0']
p0 = configs['HM_p0']
s = configs['HM_s']

# number of points to evaluate classical distribution
N_res = configs['HM_N_res'] # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
qlim, plim = [0, 1], [0, 1] # the whole unit torus

f = partial(periodic_gauss_2D, x0=q0, y0=p0, s=s) # initial classical distribution
map = partial(c_harper_inv, k) # inverse harper map

N_qus = np.asarray(configs['HM_fig_4_and_5_N_qus']) # values of quantum dimension

doer_rho.set_args(args=(q0, p0, s)) # parameters for initial quantum distribution


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
cks_qu = np.zeros((len(N_qus), n_steps)) # Krylov complexities

for i, N_qu in enumerate(N_qus):
    h = 1/(2*np.pi*N_qu) # hbar correspoding to N_qu

    # Apply parameters for given value of N_qu
    doer_rho.set_args(N=N_qu)

    doer_u.set_args(k=k, N=N_qu)

    doer_arnoldi.set_args(U=doer_u, e0=doer_rho, stop=stop,
                          prod=partial(operator_prod, hbar=h))

    doer_evolve_rho.set_args(u=doer_u, rho=doer_rho, n_steps=n_steps)

    # Calculate quantum
    evo_qu = doer_evolve_rho.doit() # evolution

    kry_qu = doer_arnoldi.doit() # quantum Krylov basis via Arnoldi

    wave_qu = krylov_wavefunction_operator(evo_qu, kry_qu, hbar=h) # wavefunction

    cks_qu[i, :] = krylov_complexity(wave_qu) # Krylov complexity

    u_qu = krylov_propagator_kry(doer_u._doit(), kry_qu, hbar=h)
    us_qu.append(u_qu)


#### Plot
figname = 'Figure_4'
plot_sequences_correspondence(u_cl, us_qu, up_to=100,
                              usetex=configs['FIGURES_USETEX'],
                              save=configs['SAVE_FIGURES'],
                              savedir=FIG_DIR + figname)

figname = 'Figure_5'
plot_complexity_correspondence(ck_cl, cks_qu, up_to=200,
                               nlogyticks=2,
                               usetex=configs['FIGURES_USETEX'],
                               save=configs['SAVE_FIGURES'],
                               savedir=FIG_DIR + figname)
