from functools import partial

import numpy as np

from qtoc_krylov.harper import *
from qtoc_krylov.evolve import *
from qtoc_krylov.krylov import *
from qtoc_krylov.misc import *
from qtoc_krylov.utilities.paths import *
from qtoc_krylov.utilities.doer import Doer
from qtoc_krylov.utilities.store import store

from plotters import plot_complexity_ket_pure_cl_limit


####
DISABLE_DOER = False
# ^ if True, bypasses Doer functionality altogether. No calculation will be
# loaded or saved

DISABLE_STORE = False
# ^ if True, disables store functionality. No operator will be loaded or saved.


#### Wrap store on some costly operators
if not DISABLE_STORE:
    q_harper = store(path=STORE_DIR)(q_harper)

    coherent_ensemble_torus = store(path=STORE_DIR)(coherent_ensemble_torus)


#### Setup Doers for data saving & retrieval
doer_evolve_f = Doer(evolve_distribution, path=CALC_DIR, disabled=DISABLE_DOER)

doer_evolve_pure = Doer(evolve_operator, path=CALC_DIR, disabled=DISABLE_DOER)

doer_evolve_ket = Doer(evolve_ket, path=CALC_DIR, disabled=DISABLE_DOER)

doer_gs = Doer(gram_schmidt_ft, ignore_args='ft', path=CALC_DIR,
               disabled=DISABLE_DOER)

doer_wave = Doer(krylov_wavefunction_cl, ignore_args=['ft', 'fk'],
                 path=CALC_DIR, disabled=DISABLE_DOER)

doer_arnoldi = Doer(arnoldi_FO_operator, path=CALC_DIR, disabled=DISABLE_DOER)

doer_lanczos = Doer(lanczos_FO_ket, path=CALC_DIR, disabled=DISABLE_DOER)

doer_ket = Doer(coherent_state_torus, disabled=DISABLE_DOER)

doer_pure = Doer(pure_coherent_torus, disabled=DISABLE_DOER)

doer_u = Doer(q_harper, disabled=DISABLE_DOER)


#### Calculation parameters
n_steps = 200 # number of time-steps
stop = n_steps # when to stop calculating Krylov

k = 0.05 # map parameter
q0, p0 = 0.4, 0.5 # center of initial state in phase space

# number of points to evaluate classical distribution
N_res = 300 # make sure is enough to resolve packet!

# define limits of phase space where the distribution is evaluated
qlim, plim = [0, 1], [0, 1] # the whole unit torus

N_qus = np.asarray([2**6, 2**7, 2**8, 2**9]) # values of quantum dimension


#### Calculate
cks_cl = np.zeros((len(N_qus), n_steps)) # classical Krylov complexities
cks_pure = np.zeros_like(cks_cl) # quantum Krylov complexities (pure density matrix)
cks_ket = np.zeros_like(cks_cl) # quantum Krylov complexities (ket)

for i, N_qu in enumerate(N_qus):
    h = 1/(2*np.pi*N_qu) # hbar correspoding to N_qu
    s = h**0.5 # classical standard deviation matching quantum

    ### Quantum
    doer_u.set_args(k=k, N=N_qu)

    ## pure density matrix
    doer_pure.set_args(q=q0, p=p0, N=N_qu)

    doer_evolve_pure.set_args(u=doer_u, rho=doer_pure, n_steps=n_steps)
    evo_pure = doer_evolve_pure.doit()

    doer_arnoldi.set_args(U=doer_u, e0=doer_pure,
                          stop=stop,
                          prod=partial(operator_prod, hbar=h))
    kry_pure = doer_arnoldi.doit()

    wave_pure = krylov_wavefunction_operator(evo_pure, kry_pure, hbar=h)
    cks_pure[i, :] = krylov_complexity(wave_pure)

    ## ket
    doer_ket.set_args(q=q0, p=p0, N=N_qu)

    doer_evolve_ket.set_args(u=doer_u, psi=doer_ket, n_steps=n_steps)
    evo_ket = doer_evolve_ket.doit()

    doer_lanczos.set_args(H=doer_u, e0=doer_ket, stop=stop)
    kry_ket = doer_lanczos.doit()

    wave_ket = krylov_wavefunction_ket(evo_ket, kry_ket)
    cks_ket[i, :] = krylov_complexity(wave_ket)

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

    doer_wave.set_args(ft=evo_cl, fk=kry_cl, qlim=qlim, plim=plim, N=N_res)
    doer_wave.set_fakeargs(map=map, rho=f, stop=stop)
    wave_cl = doer_wave.doit() # get Krylov wavefunction

    cks_cl[i, :] = krylov_complexity(wave_cl)


#### Plot
figname = 'Figure_7'
plot_complexity_ket_pure_cl_limit(cks_cl, cks_ket, cks_pure, N_qus, up_to=200,
                                  plot_dif=True, inset_pos=[0.1125, 0.475, 0.5, 0.5],
                                  save=True,
                                  savedir=FIG_DIR + figname)
