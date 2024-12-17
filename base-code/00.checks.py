# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm


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

gamma = 0.2
lamb = 1
width = 0.15
r = 1.3
Nx = 20
Ny = 20
params_dict = {'gamma': gamma, 'lamb': lamb}
#%% Main

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



# DoS
site_pos = np.array([site.pos for site in bbh_model.id_by_site])
state1 = eigenvectors[:, int(0.5 * Nx * Ny * 4)]
DoS1 = local_DoS(state1, int(Nx * Ny))
state2 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 1]
DoS2 = local_DoS(state2, int(Nx * Ny))
state3 = eigenvectors[:, int(0.5 * Nx * Ny * 4) + 1]
DoS3 = local_DoS(state3, int(Nx * Ny))
state4 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 2]
DoS4 = local_DoS(state4, int(Nx * Ny))
DoS = DoS1 + DoS2 + DoS3 + DoS4


# Reduced OPDM
cond1 = site_pos[:, 0] < (0.5 * Nx)
cond2 = site_pos[:, 1] < (0.5 * Ny)
cond = cond1 * cond2
indices = np.where(cond)[0]
rho_red = reduced_OPDM(rho, indices)
rho_red_values, rho_red_vecs = np.linalg.eigh(rho_red)
idx = rho_red_values.argsort()
rho_red_values = rho_red_values[idx]
rho_red_vecs = rho_red_vecs[:, idx]




#%% Figures
# Style sheet
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize=20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

# Defining a colormap
# Normalisation for the plots
sigmas = 4
# mean_value = np.mean(DoS1)
# std_value = np.std(DoS1)
# max_value, min_value = mean_value + sigmas * std_value, 0
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


fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])
ax1.plot(np.arange(len(eps)), eps, marker='o', color='dodgerblue', linestyle='None', markersize=1)
ax1.set_xlabel('Eigenstate', fontsize=fontsize)
ax1.set_ylabel('$\epsilon$', fontsize=fontsize)
print('fun')



fig3 = plt.figure()
gs = GridSpec(1, 1, figure=fig3, wspace=0.2, hspace=0.3)
ax1 = fig3.add_subplot(gs[0, 0])
ax1.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS,  facecolor='white', edgecolor='black', linewidth=2)
ax1.scatter(site_pos[:, 0], site_pos[:, 1], c=DoS, cmap=color_map, vmin=min_value, vmax=max_value)
ax1.set_axis_off()



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



plt.show()
