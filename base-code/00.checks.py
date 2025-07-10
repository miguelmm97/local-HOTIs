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
from modules.AmorphousLattice_C4 import AmorphousLattice_C4
from modules.Hamiltonian_Kwant import spectrum, local_DoS, Hamiltonian_Kwant, reduced_OPDM
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
gamma             = 0.5
lamb              = 1
width             = 0.0000001
r                 = 1.3
Nx                = 10
Ny                = 10
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
# lattice = AmorphousLattice_C4(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.build_lattice()
# lattice_check.build_lattice()
bbh_model = Hamiltonian_Kwant(lattice, params_dict).finalized()
# print('CHECK')
# bbh_model_check = Hamiltonian_Kwant(lattice_check, params_dict).finalized()
loger_main.info('Lattice promoted to Kwant successfully.')

# Spectrum of the closed system
loger_main.info('Calculating spectrum:')
H = bbh_model.hamiltonian_submatrix()
# print(np.allclose(H, H.T.conj()))

# Hcheck = bbh_model_check.hamiltonian_submatrix()
# print(np.allclose(H, Hcheck))
eps, eigenvectors, rho = spectrum(H)
# eps_check, eigenvectors_check, rho_check = spectrum(Hcheck)
rho_values, rho_vecs = np.linalg.eigh(rho)
idx = rho_values.argsort()
rho_values = rho_values[idx]
rho_vecs = rho_vecs[:, idx]

# f = [2, 3, 1, 0]
# for i in range(Nsites):
#     for j in range(Nsites):
#         iC4 = f[i]
#         jC4 = f[j]
#         print('------', i, j, iC4, jC4, '--------')
#         print(np.allclose(H[iC4 * 4: iC4 * 4 + 4, jC4 * 4: jC4 * 4 + 4], Hcheck[i * 4: i * 4 + 4, j * 4: j * 4 + 4], atol=1e-8))
#         if not np.allclose(H[iC4 * 4: iC4 * 4 + 4, jC4 * 4: jC4 * 4 + 4], Hcheck[i * 4: i * 4 + 4, j * 4: j * 4 + 4], atol=1e-8):
#             print(H[iC4 * 4: iC4 * 4 + 4, jC4 * 4: jC4 * 4 + 4])
#             print(Hcheck[i * 4: i * 4 + 4, j * 4: j * 4 + 4])



# Symmetry checks
S = np.kron(np.eye(Nsites), np.kron(tau_z, sigma_0))
loger_main.info(f'Chiral symmetry of H: {np.allclose(S @ H @ S, -H)}')
loger_main.info(f'Chiral symmetry of rho: {np.allclose(rho @ S + S @ rho, S)}')

# DoS for the zero modes
site_pos = np.array([site.pos for site in bbh_model.id_by_site])
# state1 = eigenvectors[:, int(0.5 * Nx * Ny * 4)]
state2 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 1]
# state3 = eigenvectors[:, int(0.5 * Nx * Ny * 4) + 1]
state4 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 2]
# DoS1 = local_DoS(state1, int(Nx * Ny))
DoS2 = local_DoS(state2, int(Nx * Ny))
# DoS3 = local_DoS(state3, int(Nx * Ny))
DoS4 = local_DoS(state4, int(Nx * Ny))
# DoS_edge = DoS1  + DoS2 + DoS3 + DoS4
DoS_edge = DoS2 + DoS4

#%% Main: Local marker an OPDM

# Cut of the system around one corner
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
band_flat_rho_red = 2 * rho_red - np.eye(int(Nred * 4))

# Theta and chiral symmetry for the reduced OPDM
C = np.kron(np.eye(Nred), np.kron(tau_z, sigma_0))
theta = np.zeros((Nred, Nred))
for i in range(Nred):
    x, y = lattice.x[indices[i]], lattice.y[indices[i]]
    r = np.sqrt(x ** 2 + y ** 2)
    theta[i, i] = theta_func(r, center_theta, sharpness_theta)
theta_op = np.kron(theta, np.eye(4))

# Mode and shell invariants
band_flat_rho_red2 = band_flat_rho_red @ band_flat_rho_red
Imode_op = C @ (band_flat_rho_red - band_flat_rho_red2) @ theta_op
Ishell_op = - 0.5 * C @ band_flat_rho_red @ (band_flat_rho_red @ theta_op - theta_op @ band_flat_rho_red)
Imode_marker = np.real([np.trace(Imode_op[4 * i: 4 * i + 4, 4 * i: 4 * i + 4]) for i in range(Nred)])
Ishell_marker = np.real([np.trace(Ishell_op[4 * i: 4 * i + 4, 4 * i: 4 * i + 4]) for i in range(Nred)])
loger_main.info(f'Imode: {np.sum(Imode_marker)}')
loger_main.info(f'Ishell: {np.sum(Ishell_marker)}')
loger_main.info(f'Ishell = Imode: {np.allclose(np.sum(Imode_marker), np.sum(Ishell_marker))}')


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
max_value, min_value = np.max(DoS2), np.min(DoS2)
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)


# Mode invariant
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
cmap = get_continuous_cmap(hex_list)
colormap_marker = cm.ScalarMappable(norm=divnorm, cmap=cmap)


# fig1 = plt.figure()
# gs = GridSpec(1, 1, figure=fig1)
# ax0 = fig1.add_subplot(gs[0, 0])
# lattice.plot_lattice(ax0)

fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.set_axis_off()
kwant.plot(bbh_model, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           site_edgecolor=None, ax=ax1)
ax1.set_title('Lattice structure')



fig5 = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 3, figure=fig5, wspace=0.5, hspace=0.5)
ax0 = fig5.add_subplot(gs[0, 0])
ax1 = fig5.add_subplot(gs[0, 1])
ax2 = fig5.add_subplot(gs[0, 2])
ax3 = fig5.add_subplot(gs[1, 0])
ax4 = fig5.add_subplot(gs[1, 1])
ax5 = fig5.add_subplot(gs[1, 2])

# Energy spectrum
ax5.plot(np.arange(len(eps)), eps, marker='o', color='mediumslateblue', linestyle='None', markersize=1)
ax5.set_xlabel('Eigenstates', fontsize=fontsize)
ax5.set_ylabel('$\epsilon$', fontsize=fontsize)
ax5.set_title('Energy spectrum', fontsize=fontsize)

# OPDM spectrum
ax4.plot(np.arange(len(rho_red_values)), rho_red_values, marker='o', color='mediumslateblue', linestyle='None', markersize=2)
ax4.set_xlabel('Eigenstates', fontsize=fontsize)
ax4.set_ylabel('$\\rho(\\theta)$', fontsize=fontsize)
ax4.set_title('OPDM', fontsize=fontsize)

# DoS of the zero modes
ax3.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_edge, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax3.set_axis_off()
ax3.set_title('DoS$(E\simeq 0)$', fontsize=fontsize)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig5.colorbar(colormap, cax=cax, orientation='vertical')


# Theta
sc = ax0.scatter(site_pos[indices, 0], site_pos[indices, 1], c=np.diag(theta), cmap=color_map, edgecolor='black', s=200)
ax0.set_axis_off()
ax0.set_title('$\\theta(r)$', fontsize=fontsize)
ax0.set_xlim(-1, Nx/3 + 0.8)
ax0.set_ylim(-1, Ny/3 + 0.8)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig5.colorbar(colormap, cax=cax, orientation='vertical')


sc = ax1.scatter(site_pos[indices, 0], site_pos[indices, 1], c=Imode_marker, cmap=cmap, norm=divnorm, edgecolors='black', s=200)
ax1.set_axis_off()
ax1.set_title('$\mathcal{I}_{mode}=$' + f'${np.sum(Imode_marker) :.2f}$', fontsize=fontsize)
ax1.set_xlim(-1, Nx/3 + 0.8)
ax1.set_ylim(-1, Ny/3 + 0.8)

sc = ax2.scatter(site_pos[indices, 0], site_pos[indices, 1], c=Ishell_marker, cmap=cmap, norm=divnorm, edgecolors='black', s=200)
ax2.set_axis_off()
ax2.set_title('$\mathcal{I}_{shell}=$' + f'${np.sum(Ishell_marker) :.2f}$', fontsize=fontsize)
ax2.set_xlim(-1, Nx/3 + 0.8)
ax2.set_ylim(-1, Ny/3 + 0.8)

# Add colorbar using divider
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig5.colorbar(colormap_marker, cax=cax, orientation='vertical')


plt.show()
