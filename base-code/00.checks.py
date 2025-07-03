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
width             = 0.00001
r                 = 1.3
Nx                = 20
Ny                = 20
Nsites            = Nx * Ny
cutx, cuty        = 0.4 * Nx, 0.4 * Ny
center_theta      = cutx / 4
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
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
bbh_model = Hamiltonian_Kwant(lattice, params_dict).finalized()
loger_main.info('Lattice promoted to Kwant successfully.')

# Spectrum of the closed system
loger_main.info('Calculating spectrum:')
H = bbh_model.hamiltonian_submatrix()
eps, eigenvectors, rho = spectrum(H)
rho_values, rho_vecs = np.linalg.eigh(rho)
idx = rho_values.argsort()
rho_values = rho_values[idx]
rho_vecs = rho_vecs[:, idx]

# Checks
S = np.kron(np.eye(Nsites), np.kron(tau_z, sigma_0))
print(np.allclose(S @ H @ S, -H))
print(np.allclose(rho @ S + S @ rho, S))

# DoS
site_pos = np.array([site.pos for site in bbh_model.id_by_site])
state1 = eigenvectors[:, int(0.5 * Nx * Ny * 4)]
state2 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 1]
state3 = eigenvectors[:, int(0.5 * Nx * Ny * 4) + 1]
state4 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 2]
DoS1 = local_DoS(state1, int(Nx * Ny))
DoS2 = local_DoS(state2, int(Nx * Ny))
DoS3 = local_DoS(state3, int(Nx * Ny))
DoS4 = local_DoS(state4, int(Nx * Ny))
DoS_edge = DoS1  + DoS2 + DoS3 + DoS4


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
Sred = np.kron(np.eye(Nred), np.kron(tau_z, sigma_0))
theta = np.zeros((Nred, Nred))
for i in range(Nred):
    x, y = lattice.x[indices[i]], lattice.y[indices[i]]
    r = np.sqrt(x ** 2 + y ** 2)
    theta[i, i] = theta_func(r, center_theta, sharpness_theta)
theta_full = np.kron(theta, np.eye(4))

# Local marker
band_flat_rho_red2 = band_flat_rho_red @ band_flat_rho_red
operator_red = Sred @ (np.eye(int(Nred * 4)) - band_flat_rho_red2) @ theta_full
invariant_red = np.real([np.trace(operator_red[4 * i: 4 * i + 4, 4 * i: 4 * i + 4]) for i in range(Nred)])
loger_main.info(f'Value of the invariant: {np.sum(invariant_red)}')




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
max_value, min_value = np.max(DoS1), np.min(DoS1)
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)



fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.set_axis_off()
kwant.plot(bbh_model, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           site_edgecolor=None, ax=ax1)
ax1.set_title('Lattice structure')


fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])
ax1.plot(np.arange(len(eps)), eps, marker='o', color='dodgerblue', linestyle='None', markersize=1)
ax1.set_xlabel('Eigenstate', fontsize=fontsize)
ax1.set_ylabel('$\epsilon$', fontsize=fontsize)
# ax1.set_ylim([-0.5, 0.5])
# ax1.set_xlim([600, 1000])


fig3 = plt.figure()
gs = GridSpec(1, 1, figure=fig3, wspace=0.2, hspace=0.3)
ax1 = fig3.add_subplot(gs[0, 0])
ax1.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS_edge, cmap=color_map, edgecolor='black', vmin=min_value, vmax=max_value)
ax1.set_axis_off()
ax1.set_title('DoS at the edge')

cbar_ax = fig3.add_subplot(gs[0, 0])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="5%", pad=3)
cbar = fig3.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()


fig4 = plt.figure()
gs = GridSpec(1, 2, figure=fig4, wspace=0.3, hspace=0.1)
ax1 = fig4.add_subplot(gs[0, 0])
ax2 = fig4.add_subplot(gs[0, 1])
ax1.plot(np.arange(len(rho_values)), rho_values, marker='o', color='dodgerblue', linestyle='None', markersize=2)
ax2.plot(np.arange(len(rho_red_values)), rho_red_values, marker='o', color='dodgerblue', linestyle='None', markersize=2)
ax1.set_xlabel('Eigenvectors', fontsize=fontsize)
ax1.set_ylabel('Eigenvalues', fontsize=fontsize)
ax2.set_xlabel('Eigenvectors', fontsize=fontsize)
ax2.set_ylabel('Eigenvalues', fontsize=fontsize)



fig5 = plt.figure()
gs = GridSpec(1, 1, figure=fig5, wspace=0.2, hspace=0.3)
ax1 = fig5.add_subplot(gs[0, 0])
ax1.scatter(site_pos[:, 0], site_pos[:, 1], facecolor='white', edgecolor='black', linewidth=1)
ax1.scatter(site_pos[indices, 0], site_pos[indices, 1], c=np.diag(theta), cmap=color_map, vmin=0, vmax=1, edgecolor='black')
ax1.set_axis_off()
ax1.set_title('Theta function')

cbar_ax = fig5.add_subplot(gs[0, 0])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="5%", pad=3)
cbar = fig5.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()


divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=get_continuous_cmap(hex_list))


fig5 = plt.figure()
gs = GridSpec(1, 1, figure=fig5, wspace=0.2, hspace=0.3)
ax1 = fig5.add_subplot(gs[0, 0])
ax1.scatter(site_pos[:, 0], site_pos[:, 1], c='white',  facecolor='white', edgecolor='black')
ax1.scatter(site_pos[indices, 0], site_pos[indices, 1], c=invariant_red, cmap=get_continuous_cmap(hex_list), edgecolor='black')
ax1.set_axis_off()
ax1.set_title('Marker')

cbar_ax = fig5.add_subplot(gs[0, 0])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="5%", pad=3)
cbar = fig5.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
# cbar.set_label(label='marker', labelpad=0, fontsize=20)
# cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
plt.show()
