

# Math
from numpy import pi
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.patheffects as path_effects
import seaborn as sns
import colorsys

# Kwant
import kwant

# Logging
import logging
import colorlog
from colorlog import ColoredFormatter

# import internal modules
from modules.functions import *
from modules.Hamiltonian_Kwant import local_DoS
from modules.colorbar_marker import get_continuous_cmap


#%% Loading data
file_list = ['fig2-data.h5']
data_dict = load_my_data(file_list, 'data')

# Parameters
Nx              = data_dict[file_list[0]]['Parameters']['Nx']
Ny              = data_dict[file_list[0]]['Parameters']['Ny']
r               = data_dict[file_list[0]]['Parameters']['r']
width           = data_dict[file_list[0]]['Parameters']['width']
gamma           = data_dict[file_list[0]]['Parameters']['gamma']
eta            = data_dict[file_list[0]]['Parameters']['eta']
cutx            = data_dict[file_list[0]]['Parameters']['cutx']
cuty            = data_dict[file_list[0]]['Parameters']['cuty']
center_theta    = data_dict[file_list[0]]['Parameters']['center_theta']
sharpness_theta = data_dict[file_list[0]]['Parameters']['sharpness_theta']
crystalline     = data_dict[file_list[0]]['Parameters']['crystalline']
C4symmetry      = data_dict[file_list[0]]['Parameters']['C4symmetry']

# Simulation data
x              = data_dict[file_list[0]]['Simulation']['x']
y              = data_dict[file_list[0]]['Simulation']['y']
site_pos       = data_dict[file_list[0]]['Simulation']['site_pos']
A_indices      = data_dict[file_list[0]]['Simulation']['A_indices']
Hval           = data_dict[file_list[0]]['Simulation']['Hval']
Hvec           = data_dict[file_list[0]]['Simulation']['Hvec']
rhoval         = data_dict[file_list[0]]['Simulation']['rhoval']
rhovec         = data_dict[file_list[0]]['Simulation']['rhovec']
rhoredval      = data_dict[file_list[0]]['Simulation']['rhoredval']
rhoredvec      = data_dict[file_list[0]]['Simulation']['rhoredvec']
theta          = data_dict[file_list[0]]['Simulation']['theta']
Imode_marker   = data_dict[file_list[0]]['Simulation']['Imode']
Ishell_marker  = data_dict[file_list[0]]['Simulation']['Ishell']

# Local density of states of the zero modes
state1 = Hvec[:, int(0.5 * Nx * Ny * 4) - 2]
state2 = Hvec[:, int(0.5 * Nx * Ny * 4) - 1]
state3 = Hvec[:, int(0.5 * Nx * Ny * 4)]
state4 = Hvec[:, int(0.5 * Nx * Ny * 4) + 1]
DoS1 = local_DoS(state1, int(Nx * Ny))
DoS2 = local_DoS(state2, int(Nx * Ny))
DoS3 = local_DoS(state3, int(Nx * Ny))
DoS4 = local_DoS(state4, int(Nx * Ny))
DoS_edge = DoS1 + DoS2 + DoS3 + DoS4
state_red = rhoredvec[:, int(0.5 * len(rhoredval))]
DoS1_redOPDM = local_DoS(state_red, int(len(rhoredval) * 0.25))

# Total value of the mode and shell markers
total_mode = np.sum(Imode_marker)
total_shell = np.sum(Ishell_marker)

# Defining the shell for plotting
def theta_func(x, b, a):
    return 1 - 1 / (1 + np.exp(-a * (x - b)))
x_theta = np.linspace(0, Nx, 1000)
y_theta = np.linspace(0, Ny, 1000)
theta_grid = np.zeros((1000, 1000))
shell_x, shell_y = [], []
for i, x_coord in enumerate(x_theta):
    for j, y_coord in enumerate(y_theta):
        r = np.sqrt(x_coord ** 2 + y_coord ** 2)
        theta_grid[i, j] = theta_func(r, center_theta, sharpness_theta)
        if 0.45 < theta_grid[i, j] < 0.55:
            shell_x.append(x_coord)
            shell_y.append(y_coord)
shell_x.append(-1)
shell_y.append(max(shell_y))
sorted_indices = np.argsort(np.array(shell_x))
shell_x, shell_y = np.array(shell_x)[sorted_indices], np.array(shell_y)[sorted_indices]

#%% Figures

# Style
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize = 20
fontsize_inset = 13
site_size  = 0.3
site_lw    = 0.00
site_color = 'orangered'
hop_color  = 'royalblue'
hop_lw     = 0.1
lead_color = 'royalblue'

# Colormap for the DoS
max_DoS, min_DoS = np.max(DoS2), np.min(DoS2)
palette_DoS = sns.color_palette("mako_r", as_cmap=True)
colors_DoS = palette_DoS(np.linspace(0.1, 1, 100))
colors_DoS[0] = [1, 1, 1, 1]
colormap_DoS = LinearSegmentedColormap.from_list("custom_colormap", colors_DoS)
norm_DoS = Normalize(vmin=min_DoS, vmax=max_DoS)
norm_theta = Normalize(vmin=0, vmax=1)
colorbar_DoS = cm.ScalarMappable(norm=Normalize(vmin=min_DoS, vmax=max_DoS), cmap=colormap_DoS)

# Diverging colormap for the invariants
norm_invariants = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
hex_blues_mako_r = sns.color_palette("mako_r", 6).as_hex()[:4]
hex_white = ['#ffffff']
hex_reds = sns.color_palette("flare_r", 6).as_hex()[2:]
hex_list = hex_reds + hex_white + hex_blues_mako_r
float_list = [0.0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]
colormap_invariants = get_continuous_cmap(hex_list, float_list=float_list)
colorbar_invariants = cm.ScalarMappable(norm=norm_invariants, cmap=colormap_invariants)


# Figure grid
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig1, wspace=0.45, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[1, 1])
fig1.text(0.03, 0.85, '$(a)$', fontsize=20, ha="center")
fig1.text(0.03, 0.4, '$(c)$', fontsize=20, ha="center")
fig1.text(0.51, 0.85, '$(b)$', fontsize=20, ha="center")
fig1.text(0.51, 0.4, '$(d)$', fontsize=20, ha="center")



# Spectrum of the restricted OPDM
ax1.plot(np.arange(len(rhoredval)), rhoredval, marker='o', color=colors_DoS[30], linestyle='None', markersize=2)
ax1.set_xlabel(' $N_\lambda$', fontsize=fontsize, labelpad=-15)
ax1.set_ylabel('$\\lambda$', fontsize=fontsize)
ax1.set_xlim(0, len(rhoredval))
ax1.set_ylim(np.min(rhoredval)-0.05, np.max(rhoredval) + 0.05)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.set(xticks=[0, len(rhoredval)])
ax1.text(20, 0.8, f'$L={Nx}$', fontsize=20)
ax1.text(250, 0.4, f'$\\gamma={gamma}$', fontsize=20)
ax1.text(250, 0.2, f'$\\eta={eta}$', fontsize=20)




# DoS of the zero modes
ax2.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_edge, cmap=colormap_DoS, edgecolor='black', s=30, linewidths=0.5, zorder=2)
ax2.fill_between(shell_x, shell_y, color=colors_DoS[30], alpha=0.4, edgecolor=None, zorder=1)
ax2.fill_between(shell_x, -np.ones(shell_y.shape), color=colors_DoS[30], alpha=0.4, edgecolor=None, zorder=1)
A_region = Polygon([[-1, -1], [Nx/2 - 0.5, -1], [Nx/2 - 0.5, Nx/2 - 0.5], [-1, Nx/2 - 0.5]],
                   closed=True, alpha=1, edgecolor=colors_DoS[50], facecolor='None')
ax2.add_patch(A_region)
ax2.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
ax2.set_ylabel('$y$', fontsize=fontsize, labelpad=-15)
ax2.set_xlim(-1.5, Nx + 0.5)
ax2.set_ylim(-1.5, Ny + 0.5)
ax2.set(xticks=[0, Nx-1], yticks=[0, Ny-1])
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
txt = ax2.text(6, 6, '$\mathcal{A}$', fontsize=20)
txt.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'), path_effects.Normal()])

# DoS for the zero modes: colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colorbar_DoS, cax=cax, orientation='vertical', ticks=[0, max_DoS])
cbar.set_ticklabels(['0.00', f'{max_DoS :.2f} '])
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\\vert \psi (\mathbf{r})\\vert ^2$', labelpad=-20, fontsize=20)


# Mode index
ax3.scatter(site_pos[A_indices, 0], site_pos[A_indices, 1], c=Imode_marker, cmap=colormap_invariants,
            norm=norm_invariants, edgecolor='black', s=130, zorder=2)

ax3.fill_between(shell_x, shell_y, color=colors_DoS[30], alpha=0.2, edgecolor=None, zorder=1)
ax3.fill_between(shell_x, -np.ones(shell_y.shape), color=colors_DoS[30], alpha=0.2, edgecolor=None, zorder=1)
ax3.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
ax3.set_ylabel('$y$', fontsize=fontsize, labelpad=-7)
ax3.set_xlim(-1, Nx/2)
ax3.set_ylim(-1, Ny/2)
ax3.set(xticks=[0, Nx/2 - 1], yticks=[0, Ny/2 - 1])
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax3.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax3.text(6.85, 8.9, f'$\mathcal{{I}}_{{\\rm mode}}={str(total_mode)[:5]}$', fontsize=15, ha="center",
         bbox=dict(facecolor="white", edgecolor="black", alpha=1))

# Mode Index: Colorbar
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colorbar_invariants, cax=cax, orientation='vertical')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\mathcal{I}_{\\rm mode}(\mathbf{r})$', labelpad=-20, fontsize=20)



# Shell Index
ax4.scatter(site_pos[A_indices, 0], site_pos[A_indices, 1], c=Ishell_marker, cmap=colormap_invariants,
            norm=norm_invariants, edgecolor='black', s=130, zorder=2)
ax4.fill_between(shell_x, shell_y, color=colors_DoS[30], alpha=0.2, edgecolor=None, zorder=1)
ax4.fill_between(shell_x, -np.ones(shell_y.shape), color=colors_DoS[30], alpha=0.2, edgecolor=None, zorder=1)
ax4.set_xlabel('$x$', fontsize=fontsize, labelpad=-20)
ax4.set_ylabel('$y$', fontsize=fontsize, labelpad=-7)
ax4.set_xlim(-1, Nx/2)
ax4.set_ylim(-1, Ny/2)
ax4.set(xticks=[0, Nx/2 - 1], yticks=[0, Ny/2 -1])
ax4.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax4.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax4.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax4.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax4.text(7, 8.9, f'$\mathcal{{I}}_{{\\rm shell}}={str(total_shell)[:5]}$', fontsize=15, ha="center",
         bbox=dict(facecolor="white", edgecolor="black", alpha=1))

# Shell invariant: Colorbar
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colorbar_invariants, cax=cax, orientation='vertical')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\mathcal{I}_{\\rm shell}(\mathbf{r})$', labelpad=-10, fontsize=20)




fig1.savefig('fig-2.pdf', format='pdf')
plt.show()
