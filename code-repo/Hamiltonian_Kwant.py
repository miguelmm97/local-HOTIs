"""
This file contains the functions and classes used in order to promote to kwant systems the lattices created as
instances of the classes AmorphousLattice_2d.
Certain sections of this code have been adapted from https://doi.org/10.5281/zenodo.4382483.

The full repository for the project is public in https://github.com/miguelmm97/local-HOTIs.git
For any questions, typos/errors or further data please write to mfmm@kth.se or miguelmartinezmiquel@gmail.com.
"""

# Math
from numpy import pi
import numpy as np
from scipy.integrate import quad
from qutip import Qobj
import scipy.linalg as la

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

#%% Module

# Pauli matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


def displacement2D_kwant(site0, site1):
    """
    Input:
    site1: kwant.builder.Site -> Site towards we hopp
    site0: kwant.builder.Site -> Site from which we hopp

    Output:
    r -> float: distance between the two sites
    phi -> float: azimuthal angle between sites
    """

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

def hopping(eta, d, phi, cutoff_dist):
    """
    Input:
    eta: float -> hopping amplitude as described in the main text
    d: float -> distance between the two sites connected by the hopping term
    phi: float -> azimuthal angle between the two sites connected by the hopping term
    cutoff_dist: float -> cutoff distance above which the hopping vanishes

    Output:
    np.ndarray (4x4): Hopping amplitude between the two sites
    """
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    hopp_x = 0.5 * eta * np.abs(np.cos(phi)) * (np.kron(sigma_x, tau_0) - 1j * np.kron(sigma_y, tau_z))
    hopp_y = - 0.5 * eta * np.abs(np.sin(phi)) * (np.kron(sigma_y, tau_y) + 1j * np.kron(sigma_y, tau_x))
    return f_cutoff * (hopp_x + hopp_y)

def spectrum(H):
    """
    Input:
    H: np.ndarray -> Hamiltonian matrix

    Output:
    energy: np.ndarray -> Hamiltonian eigenvalues
    eigenstates: np.ndarray -> Hamiltonian eigenvectors
    """

    loger_kwant.info('Calculating eigenstates...')
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]
    return energy, eigenstates

def OPDM(eigenvectors, filling=0.5, enforce_chiral_sym=False, S=None):
    """
    Input:
    eigenvectors: np.ndarray -> Hamiltonian eigenvectors
    filling: float -> Filling factor
    enforce_chiral_sym: bool -> enforces the OPDM to be chiral symmetric
    S: np.ndarray -> chiral symmetry (of the full state)

    Output:
    rho: np.ndarray -> One-particle density matrix
    """

    loger_kwant.info('Calculating OPDM...')
    dim = eigenvectors.shape[0]
    Nsp = int(dim * filling)
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[:, 0: Nsp] = eigenvectors[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))

    if enforce_chiral_sym:
        if S is not None:
            return 0.5 * (rho + np.eye(rho.shape[0]) - S @ rho @ S)
        else:
            raise ValueError('Need to specify S in order to enforce chiral symmetry.')
    else:
        return rho

def reduced_OPDM(rho, site_indices):
    """
    Input:
    rho: np.ndarray -> one-particle density matrix
    site_indices: np.ndarray -> site indices of the region A where the OPDM is restricted

    Output:
    rho_red: np.ndarray -> restricted one-particle density matrix in region A
    """

    Nred = len(site_indices) * 4
    rho_red = np.zeros((Nred, Nred), dtype=np.complex128)
    for i, site1 in enumerate(site_indices):
        for j, site2 in enumerate(site_indices):
            block1 = site1 * 4
            block2 = site2 * 4
            rho_red[i * 4: i * 4 + 4, j * 4: j * 4 + 4] = rho[block1: block1 + 4, block2: block2 + 4]

    return rho_red

def local_DoS(state, Nsites):
    """
    Input:
    state: np.ndarray -> state to calculate the local density of states from
    Nsites: int -> Number of system sites

    Output:
    local_DoS: np.ndarray -> Local density of states
    """

    local_DoS = np.zeros((Nsites, ), dtype=np.complex128)
    for i in range(Nsites):
        psi_i = state[i * 4: i * 4 + 4]
        local_DoS[i] = psi_i.T.conj() @ psi_i

    if np.sum(np.imag(local_DoS)) < 1e-10:
        local_DoS = np.real(local_DoS)
    else:
        raise TypeError('DoS is complex.')

    return local_DoS


class FullyAmorphousWire_ScatteringRegion(kwant.builder.SiteFamily):
    """
    Class to build a kwant.builder.SiteFamily for an amorphous system
    """
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
    """
    Input:
    lattice_tree: AmorphousLattice_2d -> Lattice structure of the system
    param_dict:dict -> dictionary of parameters for the Hamiltonian

    Output:
    syst -> kwant.builder.Builder: BBH model kwant system
    """

    # Load parameters into the builder namespace
    try:
        gamma  = param_dict['gamma']
        eta   = param_dict['eta']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily from the amorphous lattice
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='bbh_model')

    # Hopping and onsite functions
    def onsite_potential(site):
        return gamma * (np.kron(sigma_x, tau_0) - np.kron(sigma_y, tau_y))

    def hopp(site1, site0):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(eta, d, phi, lattice_tree.r)

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


