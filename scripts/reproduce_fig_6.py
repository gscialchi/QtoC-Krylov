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

from plotters import plot_states_correspondence


# setup parser for script
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=paths.CONFIG_DIR+'defaults.yml')
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

doer_rho = Doer(coherent_ensemble_torus, args={'f': periodic_gauss_2D},
                disabled=DISABLE_DOER)

doer_u = Doer(q_harper, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = configs['HM_fig_6_n_steps']
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

N_qus = np.asarray(configs['HM_fig_6_N_qus']) # values of quantum dimension

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
FIG_DIR = configs['FIG_DIR']
if FIG_DIR == 'default':
    FIG_DIR = paths.FIG_DIR

which = configs['HM_fig_6_states'] # which states to show

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
                           save=configs['SAVE_FIGURES'],
                           #  show=False,
                           usetex=configs['FIGURES_USETEX'],
                           savedir=FIG_DIR + figname,
                           saveformat='png', # pdf can be large
                           show=configs['SHOW_FIGURES'])
