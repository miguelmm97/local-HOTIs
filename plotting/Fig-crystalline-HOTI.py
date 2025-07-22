# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch

# Kwant
import kwant

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

from modules.functions import *
from modules.Hamiltonian_Kwant import local_DoS
from modules.colorbar_marker import get_continuous_cmap


#%% Loading data
file_list = ['Exp1.h5']
data_dict = load_my_data(file_list, '../data')

# Parameters
Nx              = data_dict[file_list[0]]['Parameters']['Nx']
Ny              = data_dict[file_list[0]]['Parameters']['Ny']
r               = data_dict[file_list[0]]['Parameters']['r']
width           = data_dict[file_list[0]]['Parameters']['width']
gamma           = data_dict[file_list[0]]['Parameters']['gamma']
lamb            = data_dict[file_list[0]]['Parameters']['lamb']
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

# DoS to plot
state1 = Hvec[:, int(0.5 * Nx * Ny * 4) - 2]
state2 = Hvec[:, int(0.5 * Nx * Ny * 4) - 1]
state3 = Hvec[:, int(0.5 * Nx * Ny * 4)]
state4 = Hvec[:, int(0.5 * Nx * Ny * 4) + 1]
DoS1 = local_DoS(state1, int(Nx * Ny))
DoS2 = local_DoS(state2, int(Nx * Ny))
DoS3 = local_DoS(state3, int(Nx * Ny))
DoS4 = local_DoS(state4, int(Nx * Ny))
DoS_edge = DoS1 + DoS2 + DoS3 + DoS4
state_red= rhoredvec[:, int(0.5 * len(rhoredval))]
DoS1_redOPDM = local_DoS(state_red, int(len(rhoredval) * 0.25))

indices_shell = [int(A_indices[i]) for i in range(theta.shape[0]) if 0.05 < theta[i, i] < 0.95]


#%% Figures
# Style sheet
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

# Defining a colormap for the DoS
max_value, min_value = np.max(DoS2), np.min(DoS2)
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0.1, 1, 100))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

norm_DoS = Normalize(vmin=min_value, vmax=max_value)
norm_theta = Normalize(vmin=0, vmax=1)
colormap_DoS = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)
# colormap_theta = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)
# edge_cmap_theta = color_map(norm_theta(np.diag(theta)))


# Mode invariant
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
# hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
hex_list = ['#800026', '#ff416d', '#ff7192', '#ffa0b6', '#ffd0db','#ffffff','#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff','#00008b']
float_list = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]
cmap = get_continuous_cmap(hex_list, float_list=float_list)
colormap_marker = cm.ScalarMappable(norm=divnorm, cmap=cmap)


fig1 = plt.figure(figsize=(14, 4))
gs = GridSpec(1, 3, figure=fig1, wspace=0.3)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])
ax0_inset = ax0.inset_axes([0.65, 0.15, 0.3, 0.3])
ax1_inset = ax1.inset_axes([0.68, 0.15, 0.3, 0.3])
pos0 = ax0.get_position()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
ax0.set_position([pos0.x0 - 0.05, pos0.y0, pos0.width, pos0.height])
ax1.set_position([pos1.x0 + 0.01, pos1.y0, pos1.width, pos1.height])
ax2.set_position([pos2.x0 + 0.05, pos2.y0, pos2.width, pos2.height])


# Energy spectrum
ax0.plot(np.arange(len(Hval)), Hval, marker='o', color='mediumslateblue', linestyle='None', markersize=1)
ax0.set_xlabel('$\\vert \\epsilon \\rangle$', fontsize=fontsize, labelpad=-15)
ax0.set_ylabel('$\epsilon$', fontsize=fontsize, labelpad=-10)
ax0.set_xlim(0, len(Hval))
ax0.set_ylim(np.min(Hval), np.max(Hval))
ax0.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax0.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax0.set(xticks=[0, len(Hval)])

# DoS of the zero modes
ax0_inset.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_edge, cmap=color_map, edgecolor='black', s=10, linewidths=0.1)
ax0_inset.tick_params(which='major', width=0.75, labelsize=fontsize_inset, color='black')
ax0_inset.tick_params(which='major', length=6, labelsize=fontsize_inset, color='black')
ax0_inset.set(xticks=[0, Nx], yticks=[0, Ny])
ax0_inset.set_xlabel('$x$', fontsize=fontsize_inset, labelpad=-15)
ax0_inset.set_ylabel('$y$', fontsize=fontsize_inset, labelpad=-15)
ax0_inset.set_title('$\\vert \psi (r)\\vert ^2$', fontsize=fontsize_inset)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colormap_DoS, cax=cax, orientation='vertical', ticks=[0, np.round(0.5 * max_value, 2), np.round(max_value, 2)])
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)



# OPDM spectrum
ax1.plot(np.arange(len(rhoredval)), rhoredval, marker='o', color='mediumslateblue', linestyle='None', markersize=2)
ax1.set_xlabel('$\\vert \\alpha \\rangle$', fontsize=fontsize, labelpad=-15)
ax1.set_ylabel('$\\lambda_\\alpha$', fontsize=fontsize)
ax1.set_xlim(0, len(rhoredval))
ax1.set_ylim(np.min(rhoredval)-0.05, np.max(rhoredval) + 0.05)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.set(xticks=[0, len(rhoredval)])

ax1_inset.scatter(site_pos[A_indices, 0], site_pos[A_indices, 1], c=np.diag(theta), cmap=color_map, edgecolor='black', s=20, linewidths=0.2)
ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize_inset, color='black')
ax1_inset.tick_params(which='major', length=6, labelsize=fontsize_inset, color='black')
ax1_inset.set(xticks=[0, int(Nx/2) - 1], yticks=[0, int(Ny/2) - 1])
ax1_inset.set_xlabel('$x$', fontsize=fontsize_inset, labelpad=-15)
ax1_inset.set_ylabel('$y$', fontsize=fontsize_inset, labelpad=-10)
ax1_inset.set_title('$\\theta (r)$', fontsize=fontsize_inset)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig1.colorbar(colormap_DoS, cax=cax, orientation='vertical', ticks=[0, np.round(0.5 * max_value, 2), np.round(max_value, 2)])
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1_inset.set_xlim(-1, Nx/2)
ax1_inset.set_ylim(-1, Ny/2)


# Markers
ax2.scatter(site_pos[A_indices, 0], site_pos[A_indices, 1], c=Imode_marker+Ishell_marker, cmap=cmap, norm=divnorm, edgecolor='black', s=130)
ax2.scatter(site_pos[indices_shell, 0], site_pos[indices_shell, 1], facecolors='none', edgecolor='violet', s=130, linewidths=2, alpha=0.7)
ax2.set_xlim(-1, Nx/2)
ax2.set_ylim(-1, Ny/2)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax2.set(xticks=[0, Nx/2 - 1], yticks=[0, Ny/2 -1])
ax2.set_xlabel('$x$', fontsize=fontsize, labelpad=-15)
ax2.set_ylabel('$y$', fontsize=fontsize, labelpad=-7)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax2.tick_params(which='major', length=6, labelsize=fontsize, color='black')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar=fig1.colorbar(colormap_marker, cax=cax, orientation='vertical')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.set_label(label='$\mathcal{I}_{\\rm mode} + \mathcal{I}_{\\rm shell}$', labelpad=5, fontsize=20)



fig1.savefig('fig1.pdf', format='pdf', bbox_inches='tight')
plt.show()
