import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

"""
Get all relevant data from the .txt files for 2D data
"""

alpha_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(3))
rho_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(7))
pressure_pressureSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range(9,50))
pressure_suctionSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (50, 113))

print(alpha_2d)
