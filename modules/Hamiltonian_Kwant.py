# %% modules setup

# Math and plotting
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

# %% Module

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


def displacement2D(x1, y1, x2, y2):

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

def hopping(lamb, d, phi, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    hopp_x = 0.5 * lamb * np.abs(np.cos(phi)) * (np.kron(sigma_x, tau_0) - 1j * np.kron(sigma_y, tau_z))
    hopp_y = - 0.5 * lamb * np.abs(np.sin(phi)) * (np.kron(sigma_y, tau_y) + 1j * np.kron(sigma_y, tau_x))
    return f_cutoff * (hopp_x + hopp_y)

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
    # loger_kwant.info('Calculating OPDM...')
    # U = np.zeros((len(H), len(H)), dtype=np.complex128)
    # U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    # rho = U @ np.conj(np.transpose(U))

    return energy, eigenstates

def OPDM(eigenvectors, filling=0.5):
    # OPDM
    loger_kwant.info('Calculating OPDM...')
    dim = eigenvectors.shape[0]
    Nsp = int(dim * filling)
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[:, 0: Nsp] = eigenvectors[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))
    return rho

def chiral_OPDM(eigenvalues, eigenvectors, S, filling=0.5, dim_zero_subspace=4):

    # Select the subspace of zero modes
    dim = eigenvectors.shape[0]
    index0 = int(0.5 * dim_zero_subspace)
    zero_modes = eigenvectors[:, int(0.5 * dim) - index0: int(0.5 * dim) + index0]
    H0 = zero_modes @ np.diag(eigenvalues[int(0.5 * dim) - 2: int(0.5 * dim) + 2]) @ zero_modes.T.conj()
    H0_2 = H0 @ H0

    # Simultaneous diagonalization
    loger_kwant.info('Diagonalizing H^2 and S in the zero subspace')
    H0_2q, Sq = Qobj(H0_2), Qobj(S)
    print((H0_2q * Sq - Sq * H0_2q).norm() / (H0_2q * Sq).norm())
    _, U = simdiag([H0_2q, Sq], tol=1e-8)
    U = U.full()
    chiral_zero_modes = U @ zero_modes
    eigenvectors[:, int(0.5 * dim) - 2: int(0.5 * dim) + 2] = chiral_zero_modes

    # OPDM
    loger_kwant.info('Calculating OPDM...')
    Nsp = int(dim * filling)
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[:, 0: Nsp] = eigenvectors[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))
    return rho

def reduced_OPDM(rho, site_indices, Nsp=None):

    Nred = len(site_indices) * 4
    rho_red = np.zeros((Nred, Nred), dtype=np.complex128)
    for i, site1 in enumerate(site_indices):
        for j, site2 in enumerate(site_indices):
            block1 = site1 * 4
            block2 = site2 * 4
            rho_red[i * 4: i * 4 + 4, j * 4: j * 4 + 4] = rho[block1: block1 + 4, block2: block2 + 4]

    return rho_red

def local_DoS(state, Nsites):

    local_DoS = np.zeros((Nsites, ), dtype=np.complex128)
    for i in range(Nsites):
        psi_i = state[i * 4: i * 4 + 4]
        local_DoS[i] = psi_i.T.conj() @ psi_i

    if np.sum(np.imag(local_DoS)) < 1e-10:
        local_DoS = np.real(local_DoS)
    else:
        raise ValueError('DoS is complex.')

    return local_DoS

def bbh_hamiltonian(lattice, param_dict):

    # Load parameters into the builder namespace
    try:
        gamma  = param_dict['gamma']
        lamb   = param_dict['lamb']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Definitions
    Nstates = int(lattice.Nx * lattice.Ny * 4)
    H = np.zeros((Nstates, Nstates), dtype=np.complex128)

    for i in range(lattice.Nsites):
        # Onsite
        i_block = i * 4
        H[i_block: i_block + 4, i_block: i_block + 4] = gamma * (np.kron(sigma_x, tau_0) + np.kron(sigma_y, tau_y))

        # Hopping
        for n in lattice.neighbours[i]:
            n_block = n * 4
            d, phi = displacement2D(lattice.x[i], lattice.y[i],  lattice.x[n], lattice.y[n])
            print(i, n, i_block, n_block)
            H[i_block: i_block + 4, n_block: n_block + 4] = hopping(lamb, d, phi, lattice.r)

    return H


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
        return gamma * (np.kron(sigma_x, tau_0) + np.kron(sigma_y, tau_y))

    def hopp(site1, site0):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(lamb, d, phi, lattice_tree.r)

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


# Simultaneous diagonalization functions (fork from qutip
def simdiag(ops, tol=1e-14, evals=True):
    """Simulateous diagonalization of communting Hermitian matrices..

    Parameters
    ----------
    ops : list/array
        ``list`` or ``array`` of qobjs representing commuting Hermitian
        operators.

    Returns
    --------
    eigs : tuple
        Tuple of arrays representing eigvecs and eigvals of quantum objects
        corresponding to simultaneous eigenvectors and eigenvalues for each
        operator.

    """
    start_flag = 0
    if not any(ops):
        raise ValueError('Need at least one input operator.')
    if not isinstance(ops, (list, np.ndarray)):
        ops = np.array([ops])
    num_ops = len(ops)
    for jj in range(num_ops):
        A = ops[jj]
        shape = A.shape
        if shape[0] != shape[1]:
            raise TypeError('Matricies must be square.')
        if start_flag == 0:
            s = shape[0]
        if s != shape[0]:
            raise TypeError('All matrices. must be the same shape')
        if not A.isherm:
            raise TypeError('Matricies must be Hermitian')
        for kk in range(jj):
            B = ops[kk]
            if (A * B - B * A).norm() / (A * B).norm() > tol:
                raise TypeError('Matricies must commute.')

    A = ops[0]
    eigvals, eigvecs = la.eig(A.full())
    zipped = zip(-eigvals, range(len(eigvals)))
    zipped.sort()
    ds, perm = zip(*zipped)
    ds = -np.real(np.array(ds))
    perm = np.array(perm)
    eigvecs_array = np.array(
        [np.zeros((A.shape[0], 1), dtype=complex) for k in range(A.shape[0])])

    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        eigvecs_array[kk][:, 0] = eigvecs[:, perm[kk]]
    k = 0
    rng = np.arange(len(eigvals))
    while k < len(ds):
        # find degenerate eigenvalues, get indicies of degenerate eigvals
        inds = np.array(abs(ds - ds[k]) < max(tol, tol * abs(ds[k])))
        inds = rng[inds]
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            eigvecs_array[inds] = degen(
                tol, eigvecs_array[inds],
                np.array([ops[kk] for kk in range(1, num_ops)]))
        k = max(inds) + 1
    eigvals_out = np.zeros((num_ops, len(ds)), dtype=float)
    kets_out = np.array([Qobj(eigvecs_array[j] / la.norm(eigvecs_array[j]),
                              dims=[ops[0].dims[0], [1]],
                              shape=[ops[0].shape[0], 1])
                         for j in range(len(ds))])
    if not evals:
        return kets_out
    else:
        for kk in range(num_ops):
            for j in range(len(ds)):
                eigvals_out[kk, j] = np.real(np.dot(
                    eigvecs_array[j].conj().T,
                    ops[kk].data * eigvecs_array[j]))
        return eigvals_out, kets_out

def degen(tol, in_vecs, ops):
    """
    Private function that finds eigen vals and vecs for degenerate matrices..
    """
    n = len(ops)
    if n == 0:
        return in_vecs
    A = ops[0]
    vecs = np.column_stack(in_vecs)
    eigvals, eigvecs = la.eig(np.dot(vecs.conj().T, A.data.dot(vecs)))
    zipped = zip(-eigvals, range(len(eigvals)))
    zipped.sort()
    ds, perm = zip(*zipped)
    ds = -np.real(np.array(ds))
    perm = np.array(perm)
    vecsperm = np.zeros(eigvecs.shape, dtype=complex)
    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        vecsperm[:, kk] = eigvecs[:, perm[kk]]
    vecs_new = np.dot(vecs, vecsperm)
    vecs_out = np.array(
        [np.zeros((A.shape[0], 1), dtype=complex) for k in range(len(ds))])
    for kk in range(len(perm)):  # matrix with sorted eigvecs in columns
        vecs_out[kk][:, 0] = vecs_new[:, kk]
    k = 0
    rng = np.arange(len(ds))
    while k < len(ds):
        inds = np.array(abs(ds - ds[k]) < max(
            tol, tol * abs(ds[k])))  # get indicies of degenerate eigvals
        inds = rng[inds]
        if len(inds) > 1:  # if at least 2 eigvals are degenerate
            vecs_out[inds] = degen(tol, vecs_out[inds],
                                   np.array([ops[jj] for jj in range(1, n)]))
        k = max(inds) + 1
    return vecs_out