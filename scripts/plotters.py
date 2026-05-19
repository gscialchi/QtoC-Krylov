import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from qtoc_krylov.evolve import evolve_map
from qtoc_krylov.harmonic import husimi_operator, husimi_state
from qtoc_krylov.harper import husimi_torus_operator, husimi_torus_state


###
def latex_config(usetex=True, usephysics=True):
    if not usetex:
        usephysics = False
    if usetex:
        from matplotlib import rc
        rc('text', usetex=True)
        if usephysics:
            rc('text.latex', preamble=r'\usepackage{physics}')


def savefig(fig, savedir, saveformat, *args, **kwargs):
    """
    Create path if non existent.
    """
    dir_split = savedir.split('/')
    path = '/'.join(dir_split[:-1])+'/'
    if not os.path.isdir(path): # make path before saving
        os.makedirs(path)
    savedir += f'.{saveformat}'
    fig.savefig(savedir, *args, **kwargs)
    print(f'Saved: {savedir}')


def plot_sequences_correspondence(u_cl, us_qu, up_to=None,
                                  figsize=(5, 4), size=14, labelsize=11,
                                  linewidth=1.5, nxticks=4,
                                  usetex=True,
                                  save=False, savedir=None, saveformat='pdf',
                                  show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0)

    xticks = mpl.ticker.MaxNLocator(nxticks)
    for ax in axes:
        ax.tick_params(labelsize=labelsize)
        ax.xaxis.set_major_locator(xticks)

    axes[0].set_ylabel(r'$a_n$', size=size)
    axes[1].set_ylabel(r'$b_n$', size=size)
    axes[2].set_ylabel(r'$c_n$', size=size)
    axes[-1].set_xlabel(r'$n$', size=size)

    cmap = mpl.colormaps['tab20b']

    # quantum
    for i, u in enumerate(us_qu):
        an = np.diag(u, k=0)
        bn = np.append([0], np.diag(u, k=-1)) # add the initial 0
        cn = u[0, :]
        n = np.arange(len(an[:up_to]))

        axes[0].plot(n, an[:up_to], c=cmap(3-i), linewidth=linewidth)
        axes[1].plot(n, bn[:up_to], c=cmap(3+4*3-i), linewidth=linewidth)
        axes[2].plot(n, cn[:up_to], c=cmap(3+4*4-i), linewidth=linewidth)

    # classical
    an = np.diag(u_cl, k=0)
    bn = np.append([0], np.diag(u_cl, k=-1)) # add the initial 0
    cn = u_cl[0, :]
    n = np.arange(len(an[:up_to]))

    axes[0].plot(n, an[:up_to], c='k', linestyle='--', linewidth=linewidth)
    axes[1].plot(n, bn[:up_to], c='k', linestyle='--', linewidth=linewidth)
    axes[2].plot(n, cn[:up_to], c='k', linestyle='--', linewidth=linewidth)

    axes[0].set_ylim(-1*(1+0.05), 1*(1+0.05))
    axes[1].set_ylim(0, 1*(1+0.01))
    axes[2].set_ylim(-1*(1+0.05), 1*(1+0.05))

    #  fig.tight_layout()
    if save:
        savefig(fig, savedir, saveformat, bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_complexity_correspondence(ck_cl, cks_qu, up_to=None,
                                   figsize=(5, 6), size=14, labelsize=11,
                                   linewidth=1.5, nxticks=5, nyticks=4, nlogyticks=3,
                                   usetex=True,
                                   save=False, savedir=None, saveformat='pdf',
                                   show=True):
    latex_config(usetex, False)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.05)

    xticks = mpl.ticker.MaxNLocator(nxticks)
    yticks = mpl.ticker.MaxNLocator(nyticks)
    for ax in axes:
        ax.tick_params(labelsize=labelsize)
        ax.xaxis.set_major_locator(xticks)
    axes[0].yaxis.set_major_locator(yticks)

    yticks_log = mpl.ticker.LogLocator(numticks=nlogyticks)
    axes[1].yaxis.set_major_locator(yticks_log)

    dif_label = 'Relative difference'
    ck_label = r'$C_{\mathcal{K}}(t)$'

    axes[0].set_ylabel(ck_label, size=size)
    axes[1].set_ylabel(dif_label, size=size)
    axes[1].set_xlabel(r'$t$', size=size)

    cmap = mpl.colormaps['tab20b']

    # quantum
    for i, ck_qu in enumerate(cks_qu):
        axes[0].plot(ck_qu[:up_to], color=cmap(3-i), linewidth=linewidth)

        dif = np.abs( (ck_qu - ck_cl)[1:]/ck_cl[1:] )
        dif = np.append([0], dif)[:up_to]
        axes[1].semilogy(dif, color=cmap(3-i))
        axes[1].axhline(y=np.mean(dif), linestyle='dashdot', color=cmap(3-i))

    # classical
    axes[0].plot(ck_cl[:up_to], color='k', linestyle='--', linewidth=linewidth)

    fig.tight_layout()
    if save:
        savefig(fig, savedir, saveformat, bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_states_correspondence(cl, qus, qlim, plim, N, hbars=None, Nqus=None,
                               cutoffs=None, which=None, map=None,
                               points=None, norm_from_k0=False,
                               size=14, labelsize=11,
                               usetex=True,
                               save=False, savedir=None, saveformat='pdf',
                               show=True):
    latex_config(usetex, False)

    q = np.linspace(*qlim, N)
    p = np.linspace(*plim, N)

    if not hbars is None:
        ms = hbars
        mlabel = r'$\hbar'
    elif not Nqus is None:
        ms = Nqus
        mlabel = r'$N'

    if not which is None:
        ncols = len(which)
    else:
        ncols = min([len(cl), len(qus[0])])
    nrows = len(ms) + 1

    scale = 1.875
    fig = plt.figure(figsize=(scale*ncols, scale*nrows))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                     axes_pad=0, cbar_mode='edge', cbar_location='right', cbar_pad=0.05)

    left = [i for i in range(ncols*nrows) if i%ncols == 0]
    right = [i for i in range(ncols*nrows) if (i-ncols+1)%ncols == 0]
    top = [i for i in range(ncols*nrows) if i < ncols]
    bottom = [i for i in range(ncols*nrows) if i >= (nrows -1)*ncols]

    grid[0].set_ylabel(r'Classical', size=size)
    for i, ax in enumerate(grid):
        if (i in left) and (i > 0):
            m = ms[::-1][i%nrows-1]
            pow = int(np.floor(np.log2(m)))
            ax.set_ylabel(mlabel + rf'=2^{{{pow}}}$', size=size)
        if i in top: ax.set_title(rf'$t={which[i]}$', size=size)


    for ax in grid:
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    if not map is None: # plot trajectories of classical map
        qm, pm = evolve_map(map, n_steps=200, points=points,
                            n_evos=10,
                            qlim=qlim, plim=plim)
        for ax in grid:
            ax.plot(qm.flatten(), pm.flatten(), ',', c='dimgrey', markersize=0.5, zorder=0)

    ##
    alpha = 0.75
    extent = [q[0], q[-1], p[0], p[-1]]

    ##
    ims_cbar = []

    for n, k in zip(range(ncols), which):
        cl_op = cl[k]

        if norm_from_k0 and (n == 0):
            kmax = np.max(np.abs(cl_op))
        elif norm_from_k0 and (n != 0):
            ...
        else:
            kmax = np.max(np.abs(cl_op))

        vmin, vmax = -kmax, kmax
        im = grid[n].imshow(cl_op, origin='lower', extent=extent,
                            vmin=vmin, vmax=vmax,
                            cmap='seismic', zorder=1, alpha=alpha)
        if n == 0: ims_cbar.append(im)


    for j in range(len(ms)):
        qu = qus[::-1][j]
        m = ms[::-1][j]
        if not cutoffs is None:
            cutoffs_h = cutoffs[::-1][j]
        for n, k in zip(range(ncols), which):
            if not cutoffs is None:
                hus = husimi_operator(q, p, qu[k], cutoffs_h, hbar=m)
            else:
                qu_k = qu[k]
                N_q = qu_k.shape[0]
                hus = husimi_torus_operator(q, p, qu_k, N_q)

            if norm_from_k0 and (n == 0):
                kmax = np.max(np.abs(hus))
            elif norm_from_k0 and (n != 0):
                ...
            else:
                kmax = np.max(np.abs(hus))

            vmin, vmax = -kmax, kmax
            im = grid[(j+1)*ncols+n].imshow(hus, origin='lower', extent=extent,
                                            vmin=vmin, vmax=vmax,
                                            cmap='seismic', zorder=1,
                                            alpha=alpha)
            if n == 0: ims_cbar.append(im)

    for i, (ax, im) in enumerate(zip(grid.cbar_axes, ims_cbar)):
        cbar = ax.colorbar(im)
        cbar.locator = mpl.ticker.MaxNLocator(3)
        cbar.update_ticks()
        if i == 0: cbar.set_label(r'$\kappa_t(q, p)$', size=size)
        else: cbar.set_label(r'$H_{\kappa_t}(q, p)$', size=size)

    if save:
        savefig(fig, savedir, saveformat, bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_states_ket_pure_cl(cl, ket, pure, qlim, plim, N, hbar=None, N_q=None,
                            cutoffs=None, which=None, map=None,
                            points=None, norm_from_k0=False,
                            size=14, labelsize=11,
                            usetex=True, usephysics=True,
                            save=False, savedir=None, saveformat='pdf',
                            show=True):
    latex_config(usetex, usephysics)

    q = np.linspace(*qlim, N)
    p = np.linspace(*plim, N)

    if not which is None:
        ncols = len(which)
    else:
        ncols = min([len(cl), len(qus[0])])
    nrows = 3

    scale = 1.875
    fig = plt.figure(figsize=(scale*ncols, scale*nrows))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                     axes_pad=0, cbar_mode='edge', cbar_location='right', cbar_pad=0.05)

    left = [i for i in range(ncols*nrows) if i%ncols == 0]
    right = [i for i in range(ncols*nrows) if (i-ncols+1)%ncols == 0]
    top = [i for i in range(ncols*nrows) if i < ncols]
    bottom = [i for i in range(ncols*nrows) if i >= (nrows -1)*ncols]

    grid[0].set_ylabel(r'Classical', size=size)

    if usephysics:
        pure_label = r'$\ketbra{\alpha(q\,p)}$'
        ket_label = r'$\ket{\alpha(q\,p)}$'
    else:
        pure_label = r'Pure'
        ket_label = r'Ket'

    grid[3].set_ylabel(pure_label, size=12)
    grid[6].set_ylabel(ket_label, size=12)

    for ax in grid:
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    if not map is None:
        qm, pm = evolve_map(map, n_steps=200, points=points,
                            n_evos=10,
                            qlim=qlim, plim=plim)
        for ax in grid:
            ax.plot(qm.flatten(), pm.flatten(), ',', c='dimgrey', markersize=0.5, zorder=0)

    ##
    alpha = 0.75
    extent = [q[0], q[-1], p[0], p[-1]]

    ##
    ims_cbar = []

    # classical
    for n, k in zip(range(ncols), which):
        cl_op = cl[k]

        if norm_from_k0 and (n == 0):
            kmax = np.max(np.abs(cl_op))
        elif norm_from_k0 and (n != 0):
            ...
        else:
            kmax = np.max(np.abs(cl_op))

        vmin, vmax = -kmax, kmax
        im = grid[n].imshow(cl_op, origin='lower', extent=extent,
                            vmin=vmin, vmax=vmax,
                            cmap='seismic', zorder=1, alpha=alpha)
        if n == 0: ims_cbar.append(im)
        grid[n].set_title(rf'$t={which[n]}$', size=size)


    for n, k in zip(range(ncols), which):
        # pure
        if not cutoffs is None:
            hus_pure = husimi_operator(q, p, pure[k], cutoffs, hbar=hbar)
        else:
            hus_pure = husimi_torus_operator(q, p, pure[k], N_q)

        if norm_from_k0 and (n == 0):
            kmax_pure = np.max(np.abs(hus_pure))
        elif norm_from_k0 and (n != 0):
            ...
        else:
            kmax_pure = np.max(np.abs(hus_pure))

        vmin_pure, vmax_pure = -kmax_pure, kmax_pure
        im = grid[3+n].imshow(hus_pure, origin='lower', extent=extent,
                              vmin=vmin_pure, vmax=vmax_pure,
                              cmap='seismic', zorder=1, alpha=alpha)
        if n == 0: ims_cbar.append(im)

        # ket
        if not cutoffs is None:
            hus_ket = husimi_state(q, p, ket[k], cutoffs, hbar=hbar)
        else:
            hus_ket = husimi_torus_state(q, p, ket[k], N_q)

        if norm_from_k0 and (n == 0):
            kmax_ket = np.max(np.abs(hus_ket))
        elif norm_from_k0 and (n != 0):
            ...
        else:
            kmax_ket = np.max(np.abs(hus_ket))

        vmin_ket, vmax_ket = -kmax_ket, kmax_ket
        im = grid[6+n].imshow(hus_ket, origin='lower', extent=extent,
                              vmin=vmin_ket, vmax=vmax_ket,
                              cmap='seismic', zorder=1, alpha=alpha)
        if n == 0: ims_cbar.append(im)


    for i, (ax, im) in enumerate(zip(grid.cbar_axes, ims_cbar)):
        cbar = ax.colorbar(im)
        cbar.locator = mpl.ticker.MaxNLocator(3)
        cbar.update_ticks()
        if i == 0: cbar.set_label(r'$\kappa_t(q, p)$', size=size)
        else: cbar.set_label(r'$H_{\kappa_t}(q, p)$', size=size)

    if save:
        savefig(fig, savedir, saveformat, bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_complexity_ket_pure_cl_limit(cl, ket, pure, hs, up_to=None,
                                      plot_dif=False, inset_pos=None,
                                      figsize=(5, 6), size=14, labelsize=11,
                                      linewidth=1.5, nxticks=5, nyticks=4, nlogyticks=3,
                                      usetex=True, usephysics=True,
                                      save=False, savedir=None, saveformat='pdf',
                                      show=True):
    latex_config(usetex, usephysics)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.05)

    xticks = mpl.ticker.MaxNLocator(nxticks)
    yticks = mpl.ticker.MaxNLocator(nyticks)
    for ax in axes:
        ax.tick_params(labelsize=labelsize)
        ax.xaxis.set_major_locator(xticks)
        ax.yaxis.set_major_locator(yticks)

    dif_label = 'Relative difference'
    ck_label = r'$C_{\mathcal{K}}(t)$'

    if usephysics:
        pure_label = r'$\ketbra{\alpha(q\,p)}$'
        ket_label = r'$\ket{\alpha(q\,p)}$'
    else:
        pure_label = r' Pure'
        ket_label = r' Ket'

    axes[0].set_ylabel(ck_label + r' (Classical vs.'+pure_label+')', size=size)
    axes[1].set_ylabel(ck_label + r' (Classical vs.'+ket_label+')', size=size)
    axes[1].set_xlabel(r'$t$', size=size)

    cmap = mpl.colormaps['tab20b']

    for i in range(len(hs)):
        axes[0].plot(cl[i][:up_to], color=cmap(3-i),
                     linestyle='--', linewidth=linewidth)
        axes[1].plot(cl[i][:up_to], color=cmap(3-i),
                     linestyle='--', linewidth=linewidth)

        axes[0].plot(pure[i][:up_to], color=cmap(3-i), linewidth=linewidth)
        axes[1].plot(ket[i][:up_to], color=cmap(3-i), linewidth=linewidth)

    if plot_dif:
        axi = axes[0].inset_axes(inset_pos)
        dif_label = 'Rel. diff.'

        for i in range(len(hs)):
            dif_pure = np.abs( (pure[i] - cl[i])[1:]/cl[i][1:] )
            dif_pure = np.append([0], dif_pure)[:up_to]

            axi.semilogy(dif_pure, color=cmap(3-i))
            axi.axhline(y=np.mean(dif_pure), linestyle='dashdot', color=cmap(3-i))

        axi.tick_params(labelsize=labelsize)
        axi.set_xticks([0, len(cl[0])//2, len(cl[0])])
        axi.set_yticks([1/10**3, 1/10**2 , 1/10, 1],
                       labels=[r'$10^{-3}$', '', '', r'$10^0$'])
        axi.set_ylabel(dif_label, size=size, labelpad=-18)

    fig.tight_layout()
    if save:
        savefig(fig, savedir, saveformat, bbox_inches='tight', dpi=fig.dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
