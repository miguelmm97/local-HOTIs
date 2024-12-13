# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.Hamiltonian_Kwant import Hamiltonian_Kwant, spectrum

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

gamma = 1.
lamb = 1.
width = 0.00000001
r = 1.3
Nx = 10
Ny = 10
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
eps, _, rho = spectrum(H)


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


fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])


kwant.plot(bbh_model, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           site_edgecolor=None, ax=ax1)



fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])


ax1.plot(np.arange(len(eps)), eps, marker='o', color='dodgerblue')
ax1.set_xlabel('Eigenstate')
ax1.set_ylabel('$\\eps/gamma')

plt.show()
