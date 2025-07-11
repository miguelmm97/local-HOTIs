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



#%% Main
width             = 0.1
r                 = 1.3
Nx                = 10
Ny                = 10

# Lattice
loger_main.info('Generating fully amorphous lattice...')
lattice1 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice2 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice3 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice4 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice5 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice1.build_lattice(C4symmetry=True)

# Differently rotated lattices
R90 = np.array([[0, -1], [1, 0]])
R180 = np.array([[-1, 0], [0, -1]])
R270 = np.array([[0, 1], [-1, 0]])
R45 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
centering = np.array([0.5 * (Nx - 1), 0.5 * (Ny - 1)])
coords1 = lattice1.coords - np.array([centering for i in range(Nx * Ny)]).T
coords2 = np.array([R90  @ coords1[:, i] + centering  for i in range(Nx * Ny)]).T
coords3 = np.array([R180 @ coords1[:, i] + centering for i in range(Nx * Ny)]).T
coords4 = np.array([R270 @ coords1[:, i] + centering for i in range(Nx * Ny)]).T
coords5 = np.array([R45  @ coords1[:, i] + centering for i in range(Nx * Ny)]).T
lattice2.set_configuration(coords2[0, :], coords2[1, :])
lattice3.set_configuration(coords3[0, :], coords3[1, :])
lattice4.set_configuration(coords4[0, :], coords4[1, :])
lattice5.set_configuration(coords5[0, :], coords5[1, :])
lattice2.build_lattice()
lattice3.build_lattice()
lattice4.build_lattice()
lattice5.build_lattice()

loger_main.info('Lattice promoted to Kwant successfully.')

fig1 = plt.figure(figsize=(17, 3))
gs = GridSpec(1, 5, figure=fig1)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])
ax3 = fig1.add_subplot(gs[0, 3])
ax4 = fig1.add_subplot(gs[0, 4])
lattice1.plot_lattice(ax0)
lattice2.plot_lattice(ax1)
lattice3.plot_lattice(ax2)
lattice4.plot_lattice(ax3)
lattice5.plot_lattice(ax4)
# ax0.set_axis_off()
# ax1.set_axis_off()
# ax2.set_axis_off()
# ax3.set_axis_off()
ax0.set_title('$R(0)$')
ax1.set_title('$R(\\pi/2)$')
ax2.set_title('$R(\\pi)$')
ax3.set_title('$R(3\\pi/2)$')
ax4.set_title('$R(\\pi/4)$')
plt.show()