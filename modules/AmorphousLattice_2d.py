#%% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
loger_amorphous = logging.getLogger('amorphous')
loger_amorphous.setLevel(logging.INFO)

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
loger_amorphous.addHandler(stream_handler)

#%% Module

# Functions for creating the lattice
def gaussian_point_set_2D(x, y, width):
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    return x, y

@dataclass
class AmorphousLattice_2d:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Class fields set upon instantiation
    Nx:  int                                          # Number of lattice sites along x direction
    Ny:  int                                          # Number of lattice sites along y direction
    w:   float                                        # Width of the Gaussian distribution
    r:   float                                        # Cutoff distance to consider neighbours

    # Class fields that can be set externally
    x: np.ndarray         = None                      # x position of the sites
    y: np.ndarray         = None                      # y position of the sites
    K_onsite: float       = None                      # Strength of the onsite disorder distribution
    K_hopp:   float       = None                      # Strength of the hopping disorder distribution
    disorder: np.ndarray  = None                      # Disorder matrix

    # Class fields that can only be set internally
    Nsites: int = field(init=False)                         # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)              # Neighbours list for each site
    neighbours_projection: np.ndarray = field(init=False)   # Neighbours list for each site on the 2d projection


    # Methods for building the lattice

    def build_lattice(self, n_tries=0, restrict_connectivity=False):

        if n_tries > 100:
            loger_amorphous.error('Loop. Parameters might not allow an acceptable configuration.')
        if self.w  < 1e-10:
            loger_amorphous.error('The amorphicity cannot be strictly 0')
            exit()

        # Restricting to only connected lattice configurations
        if restrict_connectivity:
            try:
                self.generate_configuration(restrict_connectivity=True)
                loger_amorphous.trace('Configuration accepted!')
            except Exception as error:
                loger_amorphous.warning(f'{error}')
                try:
                    self.erase_configuration()
                    self.erase_disorder()
                    self.build_lattice(n_tries=n_tries + 1, restrict_connectivity=True)
                except RecursionError:
                    loger_amorphous.error('Recursion error. Infinite loop. Terminating...')
                    exit()
        else:
            # Accepting literally anything
            self.generate_configuration()

    def generate_configuration(self, restrict_connectivity=False):
        loger_amorphous.trace('Generating lattice and neighbour tree...')

        # Positions of x and y coordinates on the amorphous lattice
        self.Nsites = int(self.Nx * self.Ny)
        if self.x is None and self.y is None:
            list_sites = np.arange(0, self.Nsites)
            x_crystal, y_crystal = list_sites % self.Nx, list_sites // self.Nx
            self.x, self.y = gaussian_point_set_2D(x_crystal, y_crystal, self.w)
        coords = np.array([self.x, self.y])

        # Neighbour tree and accepting/discarding the configuration
        self.neighbours = KDTree(coords.T).query_ball_point(coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)
            if restrict_connectivity and len(self.neighbours[i]) < 2:
                raise ValueError('Connectivity of the lattice too low. Trying a different configuration...')

    def generate_disorder(self, K_onsite=0., K_hopp=0.):

        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite, self.K_hopp = K_onsite, K_hopp

        # Generate a matrix with diagonal onsite disorder and symmetric (hermitian) hopping disorder
        aux_diag = np.random.uniform(-self.K_onsite, self.K_onsite, self.Nsites)
        aux_matrix = np.random.uniform(-self.K_hopp, self.K_hopp, (self.Nsites, self.Nsites))
        disorder_matrix = np.tril(aux_matrix, k=-1)
        disorder_matrix = disorder_matrix + disorder_matrix.T
        disorder_matrix = disorder_matrix + np.diag(aux_diag)
        self.disorder = disorder_matrix

    def plot_lattice(self, ax):

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours[site]:
                ax.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'royalblue', linewidth=1, alpha=0.2)
                ax.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

        # Lattice sites
        ax.scatter(self.x, self.y, color='deepskyblue', s=50)

    # Setters and erasers
    def set_configuration(self, x, y):
        self.x, self.y = x, y

    def set_disorder(self, disorder):
        self.disorder = disorder

    def erase_configuration(self):
        self.x, self.y = None, None

    def erase_disorder(self):
        self.disorder= None

