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
from modules.Hamiltonian_Kwant import spectrum, local_DoS, Hamiltonian_Kwant, OPDM, reduced_OPDM
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
width             = 0.1
r                 = 1.3
Nx                = 20
Ny                = 20
Nsites            = Nx * Ny
cutx, cuty        = 0.5 * Nx, 0.5 * Ny
center_theta      = 1 * cutx / 2
sharpness_theta   = 1 * center_theta
params_dict = {'gamma': gamma, 'lamb': lamb}
crystalline = False
C4symmetry = True

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
lattice.build_lattice(crystalline=crystalline, C4symmetry=C4symmetry)
loger_main.info('Lattice promoted to Kwant successfully.')

# Spectrum of the closed system
loger_main.info('Calculating Hamiltonian and spectrum:')
bbh_model = Hamiltonian_Kwant(lattice, params_dict).finalized()
H = bbh_model.hamiltonian_submatrix()
eps, eigenvectors = spectrum(H)

# Chiral symmetry and OPDM
S = np.kron(np.eye(Nsites), np.kron(tau_z, sigma_0))
rho = OPDM(eigenvectors, filling=0.5, enforce_chiral_sym=True, S=S)
rho_values, rho_vecs = np.linalg.eigh(rho)
idx = rho_values.argsort()
rho_values = rho_values[idx]
rho_vecs = rho_vecs[:, idx]
loger_main.info(f'Chiral symmetry of H: {np.allclose(S @ H @ S, -H)}')
loger_main.info(f'Chiral symmetry of rho: {np.allclose(rho @ S + S @ rho, S)}')

# DoS for the zero modes
site_pos = np.array([site.pos for site in bbh_model.id_by_site])
state1 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 3]
state2 = eigenvectors[:, int(0.5 * Nx * Ny * 4) - 1]
state3 = eigenvectors[:, int(0.5 * Nx * Ny * 4)]
state4 = eigenvectors[:, int(0.5 * Nx * Ny * 4) + 2]
DoS1 = local_DoS(state1, int(Nx * Ny))
DoS2 = local_DoS(state2, int(Nx * Ny))
DoS3 = local_DoS(state3, int(Nx * Ny))
DoS4 = local_DoS(state4, int(Nx * Ny))
DoS_edge = DoS4 #+ DoS2 + DoS3 + DoS4

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

# Theta and chiral symmetry for the reduced OPDM
C = np.kron(np.eye(Nred), np.kron(tau_z, sigma_0))
theta = np.zeros((Nred, Nred))
for i in range(Nred):
    x, y = lattice.x[indices[i]], lattice.y[indices[i]]
    r = np.sqrt(x ** 2 + y ** 2)
    theta[i, i] = theta_func(r, center_theta, sharpness_theta)
theta_op = np.kron(theta, np.eye(4))
loger_main.info(f'Chiral symmetry of rho_reduced: {np.allclose(rho_red @ C + C @ rho_red, C)}')

# Mode and shell invariants
rho_red2 = rho_red @ rho_red
Imode_op = 4 * C @ (rho_red - rho_red2)  @ theta_op
Ishell_op = - 2 * C @ rho_red @ (rho_red @ theta_op - theta_op @ rho_red)
Imode_marker = np.real([np.trace(Imode_op[4 * i: 4 * i + 4, 4 * i: 4 * i + 4]) for i in range(Nred)])
Ishell_marker = np.real([np.trace(Ishell_op[4 * i: 4 * i + 4, 4 * i: 4 * i + 4]) for i in range(Nred)])
loger_main.info(f'Imode: {np.sum(Imode_marker)}')
loger_main.info(f'Ishell: {np.sum(Ishell_marker)}')
loger_main.info(f'Ishell = Imode: {np.allclose(np.sum(Imode_marker), np.sum(Ishell_marker))}')


# %% Saving data
data_dir = '../data'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'x', lattice.x)
    store_my_data(simulation, 'y', lattice.y)
    store_my_data(simulation, 'site_pos', site_pos)
    store_my_data(simulation, 'A_indices', indices)
    store_my_data(simulation, 'Hval', eps)
    store_my_data(simulation, 'Hvec', eigenvectors)
    store_my_data(simulation, 'rhoval', rho_values)
    store_my_data(simulation, 'rhovec', rho_vecs)
    store_my_data(simulation, 'rhoredval', rho_red_values)
    store_my_data(simulation,'rhoredvec', rho_red_vecs)
    store_my_data(simulation, 'theta', theta)
    store_my_data(simulation, 'Imode', Imode_marker)
    store_my_data(simulation, 'Ishell', Ishell_marker)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'width', width)
    store_my_data(parameters, 'Nx', Nx)
    store_my_data(parameters, 'Ny', Ny)
    store_my_data(parameters, 'r', r)
    store_my_data(parameters, 'gamma', gamma)
    store_my_data(parameters, 'lamb', lamb)
    store_my_data(parameters, 'cutx', cutx)
    store_my_data(parameters, 'cuty', cuty)
    store_my_data(parameters, 'center_theta', center_theta)
    store_my_data(parameters, 'sharpness_theta', sharpness_theta)
    store_my_data(parameters, 'crystalline', crystalline)
    store_my_data(parameters, 'C4symmetry', C4symmetry)

loger_main.info('Data saved correctly')
