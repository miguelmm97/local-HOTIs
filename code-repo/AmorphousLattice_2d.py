"""
This file contains the functions and classes used to create a 2d crystalline/amorphous lattice.

The full repository for the project is public in https://github.com/miguelmm97/Amorphous-nanowires.git
For any questions, typos/errors or further data please write to mfmm@kth.se or miguelmartinezmiquel@gmail.com.
"""

# Math
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
from scipy.integrate import quad

# Plotting
import matplotlib.pyplot as plt

# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Logging
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
    """
    Input:
    x: np.ndarray -> x positions of the crystalline sites
    y: np.ndarray -> y positions of the crystalline sites
    width: float -> standard deviation of the Gaussian point set

    Output:
    x: np.ndarray->  x positions of the amorphous structure
    y: np.ndarray->  y positions of the amorphous structure
    """
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    return x, y

@dataclass
class AmorphousLattice_2d:

    # Class fields set upon instantiation
    Nx:  int                                        # Number of lattice sites along x direction
    Ny:  int                                        # Number of lattice sites along y direction
    w:   float                                      # Width of the Gaussian distribution
    r:   float                                      # Cutoff distance to consider neighbours

    # Class fields that can be set externally
    x: np.ndarray   = None                          # x position of the sites
    y: np.ndarray   = None                          # y position of the sites
    coords: np.ndarray = None                       # Coordinates of the lattice sites
    K_onsite: float = None                          # Strength of the onsite disorder distribution
    onsite_disorder: np.ndarray = None              # Disorder array for only the onsite case

    # Class fields that can only be set internally
    Nsites: int = field(init=False)                 # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)      # Neighbours list for each site
    area: float = field(init=False)                 # Area of the wire's cross-section


    # Methods for building the lattice
    def build_lattice(self, crystalline=False, C4symmetry=False):
        if self.w < 1e-10 and not crystalline:
            loger_amorphous.error('The amorphicity cannot be strictly 0')
            exit()
        self.generate_configuration(crystalline=crystalline)
        if C4symmetry:
            self.generate_C4symmetric_configuration()
        self.generate_neighbour_tree()

    def generate_neighbour_tree(self):
        self.neighbours = KDTree(self.coords.T).query_ball_point(self.coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)

    def generate_configuration(self, crystalline=False):
        loger_amorphous.trace('Generating lattice and neighbour tree...')

        # Positions of x and y coordinates in the amorphous structure
        self.Nsites = int(self.Nx * self.Ny)
        if self.x is None and self.y is None:
            list_sites = np.arange(0, self.Nsites)
            x_crystal = list_sites % self.Nx
            y_crystal = list_sites // self.Nx
            if crystalline:
                self.x, self.y = x_crystal, y_crystal
            else:
                self.x, self.y = gaussian_point_set_2D(x_crystal, y_crystal, self.w)
        self.coords = np.array([self.x, self.y])

        # Set up preliminary disorder
        self.K_onsite = 0.

    def generate_C4symmetric_configuration(self):
        if self.Nx % 2 != 0:
            loger_amorphous.error('Number of sites must be even for C4 symmetry')
            exit()
        elif self.Ny % 2 != 0:
            loger_amorphous.error('Number of sites must be even for C4 symmetry')
            exit()

        # C4 rotation matrices
        R90 = np.array([[0, -1], [1, 0]])
        R180 = np.array([[-1, 0], [0, -1]])
        R270 = np.array([[0, 1], [-1, 0]])

        # C4 mapping in a square lattice
        list_sites = np.arange(0, self.Nsites)
        x_crystal, y_crystal = list_sites % self.Nx, list_sites // self.Nx
        condx = x_crystal > (self.Nx - 1) / 2
        condy = y_crystal > (self.Ny - 1) / 2
        cond = condx * condy
        centering = np.array([0.5 * (self.Nx - 1), 0.5 * (self.Ny - 1)])
        gen_sites = np.where(cond)[0]
        gen_coords = [np.array([x_crystal[i], y_crystal[i]]) - centering for i in gen_sites]
        coords_tl = [R90 @ gen_coords[i]  + centering for i in range(len(gen_coords))]
        coords_bl = [R180 @ gen_coords[i] + centering for i in range(len(gen_coords))]
        coords_br = [R270 @ gen_coords[i] + centering for i in range(len(gen_coords))]
        sites_tl = [int(coords_tl[i][0] + self.Nx * coords_tl[i][1]) for i in range(len(coords_tl))]
        sites_bl = [int(coords_bl[i][0] + self.Nx * coords_bl[i][1]) for i in range(len(coords_bl))]
        sites_br = [int(coords_br[i][0] + self.Nx * coords_br[i][1]) for i in range(len(coords_br))]

        # Changing coordinates of the amorphous sites to fulfill C4 symmetry
        am_gen_coords = [np.array([self.x[i], self.y[i]]) - centering for i in gen_sites]
        am_coords_tl = [R90 @  am_gen_coords[i] + centering for i in range(len(am_gen_coords))]
        am_coords_bl = [R180 @ am_gen_coords[i] + centering for i in range(len(am_gen_coords))]
        am_coords_br = [R270 @ am_gen_coords[i] + centering for i in range(len(am_gen_coords))]
        self.x[sites_tl], self.y[sites_tl] = np.array(am_coords_tl)[:, 0], np.array(am_coords_tl)[:, 1]
        self.x[sites_bl], self.y[sites_bl] = np.array(am_coords_bl)[:, 0], np.array(am_coords_bl)[:, 1]
        self.x[sites_br], self.y[sites_br] = np.array(am_coords_br)[:, 0], np.array(am_coords_br)[:, 1]
        self.coords = np.array([self.x, self.y])

    def generate_onsite_disorder(self, K_onsite):
        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite = K_onsite
        self.onsite_disorder = np.random.uniform(-self.K_onsite, self.K_onsite, self.Nsites)

    # Setters and erasers
    def set_configuration(self, x, y):
        self.x, self.y = x, y

    def set_disorder(self, onsite_disorder, K_onsite):
        self.K_onsite = K_onsite
        self.onsite_disorder = onsite_disorder

    def erase_configuration(self):
        self.x, self.y = None, None

    def erase_disorder(self):
        self.onsite_disorder = None
