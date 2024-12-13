# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad

# Kwant
import kwant
import tinyarray as ta

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# %% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.INFO)

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
loger_kwant.addHandler(stream_handler)

# %% Module

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z



def displacement2D_kwant(site0, site1):
    x1, y1 = site0.pos[0], site0.pos[1]
    x2, y2 = site1.pos[0], site1.pos[1]

    v = np.zeros((2,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)

    # Norm of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:                                    # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                             # Hopping in y
        else:
            phi = 3 * pi / 2                         # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])             # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])    # 3rd and 4th quadrants

    return r, phi

def hopping(gamma, lamb, d, phi, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    normal_hopp = gamma * (np.kron(sigma_x, tau_0) + np.kron(sigma_y, tau_y))
    hopp_x = 0.5 * lamb * np.abs(np.cos(phi)) * (np.kron(sigma_x, tau_0) - 1j * np.kron(sigma_y, tau_z))
    hopp_y = - 0.5 * lamb * np.abs(np.sin(phi)) * (np.kron(sigma_y, tau_y) + 1j * np.kron(sigma_y, tau_x))
    return f_cutoff * (normal_hopp + hopp_x + hopp_y)

def spectrum(H, Nsp=None):

    if Nsp is None:
        Nsp = int(len(H) / 2)

    # Spectrum
    loger_kwant.info('Calculating eigenstates...')
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]

    # OPDM
    loger_kwant.info('Calculating OPDM...')
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))

    return energy, eigenstates, rho


class FullyAmorphousWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, lattice, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        loger_kwant.trace('Initialising cross section as a SiteFamily...')
        self.norbs = norbs
        self.coords = np.array([lattice.x, lattice.y]).T
        self.Nsites = lattice.Nsites
        self.Nx = lattice.Nx
        self.Ny = lattice.Ny
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.coords[tag, :][0, :]

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

def Hamiltonian_Kwant(lattice_tree, param_dict):

    # Load parameters into the builder namespace
    try:
        gamma  = param_dict['gamma']
        lamb   = param_dict['lamb']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily from the amorphous lattice
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='bbh_model')

    # Hopping and onsite functions
    def onsite_potential(site):
        if lattice_tree.K_hopp < 1e-12:
            return np.zeros((4, 4))
        else:
            index = site.tag[0]
            return np.kron(sigma_0, tau_0) * lattice_tree.disorder[index, index]

    def hopp(site1, site0):
        index0, index1 = site0.tag[0], site1.tag[0]
        index_neigh = lattice_tree.neighbours[index0].index(index1)
        d, phi= displacement2D_kwant(site1, site0)
        if lattice_tree.K_hopp < 1e-12:
            return hopping(gamma, lamb, d, phi, lattice_tree.r)
        else:
            return hopping(gamma, lamb, d, phi, lattice_tree.r)  + \
                np.kron(sigma_0, tau_0) * lattice_tree.disorder[index0, index_neigh]

    # Initialise kwant system
    loger_kwant.trace('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i) for i in range(latt.Nsites))] = onsite_potential

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in lattice_tree.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')
            syst[(latt(n), latt(i))] = hopp

    return syst