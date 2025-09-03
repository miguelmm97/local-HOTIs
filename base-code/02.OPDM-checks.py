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


# Kwant
import kwant

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.Hamiltonian_Kwant import spectrum, local_DoS, Hamiltonian_Kwant, reduced_OPDM, OPDM, occupied_zero_energy_DoS
from modules.colorbar_marker import get_continuous_cmap

#%% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

stream_handler.setFormatter(formatter)
loger_main.addHandler(stream_handler)



#%% Variables
gamma             = 0.3
lamb              = 1
width             = 0.0000001
r                 = 1.3
Nx                = 20
Ny                = 20
Nsites            = Nx * Ny
cutx, cuty        = 0.5 * Nx, 0.5 * Ny
center_theta      = cutx / 2
sharpness_theta   = 1 * center_theta
params_dict = {'gamma': gamma, 'lamb': lamb}

# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z

# Thea function for selecting a single corner
def theta_func(x, b, a):
    return 1 - 1 / (1 + np.exp(-a * (x - b)))

#%% Main: System

# Lattice
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.build_lattice()
loger_main.info('Lattice promoted to Kwant successfully.')

# Spectrum of the closed system and OPDM
loger_main.info('Calculating Hamiltonian and spectrum:')
bbh_model = Hamiltonian_Kwant(lattice, params_dict).finalized()
H = bbh_model.hamiltonian_submatrix()
eps, eigenvectors = spectrum(H)

# OPDM and spectrum
S = np.kron(np.eye(Nsites), np.kron(tau_z, sigma_0))
rho = OPDM(eigenvectors, filling=0.5, enforce_chiral_sym=True, S=S)
rho_values, rho_vecs = np.linalg.eigh(rho)
idx = rho_values.argsort()
rho_values = rho_values[idx]
rho_vecs = rho_vecs[:, idx]
loger_main.info(f'Chiral symmetry of H: {np.allclose(S @ H @ S, -H)}')
loger_main.info(f'Chiral symmetry of rho: {np.allclose(rho @ S + S @ rho, S)}')

# Cut of the system around one corner
site_pos = np.array([site.pos for site in bbh_model.id_by_site])
cond1 = site_pos[:, 0] < cutx
cond2 = site_pos[:, 1] < cuty
cond = cond1 * cond2
indices = np.where(cond)[0]
Nred = len(indices)

# Reduce OPDM in the cut
rho_red = reduced_OPDM(rho, indices)
rho_red_values, rho_red_vecs = np.linalg.eigh(rho_red)
idx = rho_red_values.argsort()
rho_red_values, rho_red_vecs = rho_red_values[idx],  rho_red_vecs[:, idx]
C = np.kron(np.eye(Nred), np.kron(tau_z, sigma_0))
loger_main.info(f'Chiral symmetry of rho_reduced: {np.allclose(rho_red @ C + C @ rho_red, C)}')


# Theta operator
theta = np.zeros((Nred, Nred))
for i in range(Nred):
    x, y = lattice.x[indices[i]], lattice.y[indices[i]]
    r = np.sqrt(x ** 2 + y ** 2)
    theta[i, i] = theta_func(r, center_theta, sharpness_theta)
theta_op = np.kron(theta, np.eye(4))

# DoS for the corner states
state1 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 3]
state2 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 2]
state3 = eigenvectors[:, int(0.5 * Nx * Ny * 4)]
state4 = eigenvectors[:, int(0.5 * Nx * Ny * 4) + 3]
DoS1_corner = local_DoS(state1, int(Nx * Ny))
DoS2_corner = local_DoS(state2, int(Nx * Ny))
DoS3_corner = local_DoS(state3, int(Nx * Ny))
DoS4_corner = local_DoS(state4, int(Nx * Ny))
E1 = eps[int(0.5 * Nx * Ny * 4) - 1]
E2 = eps[int(0.5 * Nx * Ny * 4) - 2]
E3 = eps[int(0.5 * Nx * Ny * 4)]
E4 = eps[int(0.5 * Nx * Ny * 4) + 1]
occupied_zero_Dos = occupied_zero_energy_DoS(rho_vecs, H, Nsites)


# Eigenstates of the reduced OPDM
state1 = rho_red_vecs[:, int(0.5 * Nred * 4) - 2]
state2 = rho_red_vecs[:, int(0.5 * Nred * 4) - 1]
state3 = rho_red_vecs[:, int(0.5 * Nred * 4)]
state4 = rho_red_vecs[:, int(0.5 * Nred * 4) + 1]
DoS1_redOPDM = local_DoS(state1, Nred)
DoS2_redOPDM = local_DoS(state2, Nred)
DoS3_redOPDM = local_DoS(state3, Nred)
DoS4_redOPDM = local_DoS(state4, Nred)
lambda1 = rho_red_values[int(0.5 * Nred * 4) - 2]
lambda2 = rho_red_values[int(0.5 * Nred * 4) - 1]
lambda3 = rho_red_values[int(0.5 * Nred * 4) ]
lambda4 = rho_red_values[int(0.5 * Nred * 4) + 1]



#%% Figures
# Style sheet
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize = 20
site_size  = 0.3
site_lw    = 0.00
site_color = 'orangered'
hop_color  = 'royalblue'
hop_lw     = 0.1
lead_color = 'royalblue'

# Defining a colormap for the DoS
max_value, min_value = np.max(DoS1_corner), np.min(DoS1_corner)
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)

fig1 = plt.figure(figsize=(15, 7))
gs = GridSpec(2, 4, figure=fig1, wspace=0.5, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])
ax3 = fig1.add_subplot(gs[0, 3])
ax8 = fig1.add_subplot(gs[1, 0])
ax9 = fig1.add_subplot(gs[1, 1])
ax10 = fig1.add_subplot(gs[1, 2])
ax11 = fig1.add_subplot(gs[1, 3])

# Reduced density matrix "zero modes"
ax0.scatter(site_pos[indices, 0], site_pos[indices, 1], c=DoS1_redOPDM, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value, s=200)
ax0.set_axis_off()
ax0.set_title(f'$\\vert \\rho(\\theta), \\lambda _1= {lambda1 :.4f} \\rangle$', fontsize=fontsize)
ax0.set_xlim(-1, Nx/2 + 0.8)
ax0.set_ylim(-1, Ny/2 + 0.8)
ax1.scatter(site_pos[indices, 0], site_pos[indices, 1], c=DoS2_redOPDM, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value, s=200)
ax1.set_axis_off()
ax1.set_title(f'$\\vert \\rho(\\theta), \\lambda _2= {lambda2 :.4f} \\rangle$', fontsize=fontsize)
ax1.set_xlim(-1, Nx/2 + 0.8)
ax1.set_ylim(-1, Ny/2 + 0.8)
ax2.scatter(site_pos[indices, 0], site_pos[indices, 1], c=DoS3_redOPDM, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value, s=200)
ax2.set_axis_off()
ax2.set_title(f'$\\vert \\rho(\\theta), \\lambda _3= {lambda3 :.4f} \\rangle$', fontsize=fontsize)
ax2.set_xlim(-1, Nx/2 + 0.8)
ax2.set_ylim(-1, Ny/2 + 0.8)
ax3.scatter(site_pos[indices, 0], site_pos[indices, 1], c=DoS4_redOPDM, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value, s=200)
ax3.set_axis_off()
ax3.set_title(f'$\\vert \\rho(\\theta), \\lambda _4= {lambda4 :.4f} \\rangle$', fontsize=fontsize)
ax3.set_xlim(-1, Nx/2 + 0.8)
ax3.set_ylim(-1, Ny/2 + 0.8)

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(colormap, cax=cax, orientation='vertical')


# Spectrum corner modes
ax8.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS1_corner, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax8.set_axis_off()
ax8.set_title(f'$\\vert E_1= {E1 :.4f}\\rangle$', fontsize=fontsize)
ax9.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS2_corner, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax9.set_axis_off()
ax9.set_title(f'$\\vert E_2= {E2 :.4f} \\rangle$', fontsize=fontsize)
ax10.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS3_corner, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax10.set_axis_off()
ax10.set_title(f'$\\vert E_3= {E3 :.4f} \\rangle$', fontsize=fontsize)
ax11.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS4_corner, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax11.set_axis_off()
ax11.set_title(f'$\\vert E_4= {E4 :.4f} \\rangle$', fontsize=fontsize)

divider = make_axes_locatable(ax11)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig1.colorbar(colormap, cax=cax, orientation='vertical')


fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.5, hspace=0.5)
ax0 = fig2.add_subplot(gs[0, 0])
ax0.plot(np.arange(len(rho_red_values)), rho_red_values, marker='o', color='mediumslateblue', linestyle='None', markersize=2)
ax0.set_xlabel('Eigenstates', fontsize=fontsize)
ax0.set_ylabel('$\\rho(\\theta)$', fontsize=fontsize)
ax0.set_title('OPDM', fontsize=fontsize)


fig3 = plt.figure(figsize=(5, 5))
gs = GridSpec(1, 1, figure=fig3, wspace=0.5, hspace=0.5)
ax0 = fig3.add_subplot(gs[0, 0])
ax0.scatter(site_pos[:, 0], site_pos[:, 1], c=occupied_zero_Dos, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax0.set_axis_off()
ax0.set_title(f'Occupied DoS at zero energy', fontsize=fontsize)

plt.show()