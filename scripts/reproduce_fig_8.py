from functools import partial
import argparse

import numpy as np

from qtoc_krylov.harper import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
import qtoc_krylov.utilities.paths as paths
from qtoc_krylov.utilities.doer import Doer
from qtoc_krylov.utilities.store import store
from qtoc_krylov.utilities.config import get_config

from plotters import plot_states_ket_pure_cl


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=CONFIG_DIR+'/defaults.yml')
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
    q_harper = store(path=STORE_DIR)(q_harper)

    coherent_ensemble_torus = store(path=STORE_DIR)(coherent_ensemble_torus)


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=DOER_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=DOER_DIR,
               disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=DOER_DIR, disabled=DISABLE_DOER)

doer_lanczos = Doer(lanczos_FO_ket, path=DOER_DIR, disabled=DISABLE_DOER)

doer_ket = Doer(coherent_state_torus, disabled=DISABLE_DOER)

doer_pure = Doer(pure_coherent_torus, disabled=DISABLE_DOER)

doer_u = Doer(q_harper, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = configs['HM_fig_8_n_steps']
stop = n_steps # when to stop calculating Krylov

k = configs['HM_k']
q0 = configs['HM_q0']
p0 = configs['HM_p0']

# number of points to evaluate classical distribution
N_res = configs['HM_N_res'] # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
qlim, plim = [0, 1], [0, 1] # the whole unit torus

N_qu = configs['HM_fig_8_N_qu'] # quantum dimension


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
FIG_DIR = configs['FIG_DIR']
if FIG_DIR == 'default':
    FIG_DIR = paths.FIG_DIR

which = configs['HM_fig_8_states'] # which states to show

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
                        save=configs['SAVE_FIGURES'],
                        usetex=configs['FIGURES_USETEX'],
                        usephysics=configs['FIGURES_USEPHYSICS'],
                        savedir=FIG_DIR + figname,
                        saveformat='png') # pdf can be large
