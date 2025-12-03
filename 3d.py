import numpy as np
import math as m
import matplotlib.pyplot as plt

from constants import const


# ----- Loading Data -----
exp_data_3d = np.genfromtxt('raw_Group8_3d.txt', skip_header=2)
exp_alpha = exp_data_3d[:,2] # deg
exp_drag_F = exp_data_3d[:,6] # N
exp_lift_F = exp_data_3d[:,7] # N
exp_rho = exp_data_3d[:, 10] # kg/m^3


xflr_data_3d = np.genfromtxt('T1-19.2ms.csv', delimiter=",", skip_header=7)
xflr_alpha = xflr_data_3d[:,0] # deg
xflr_Cl = xflr_data_3d[:,2] # [-]
xflr_Cd = xflr_data_3d[:,5] # [-]
xflr_Cdi = xflr_data_3d[:, 3] # [-]

print(xflr_data_3d)

# ----- Calculations -----
exp_q = 1/2 * exp_rho * const['velocity']**2 
exp_Cl = exp_lift_F / (exp_q * const['surface_area_3D'])
exp_Cd = exp_drag_F / (exp_q * const['surface_area_3D'])

A = const['span_3D']/const['chord']
#print(A)


# Finding tau
key = (exp_alpha > 0) & (exp_alpha < 10)
a3D, intercept = np.polyfit(exp_alpha[key], exp_Cl[key], 1)  # [1/deg], [-]
tau = (a3D*180/m.pi) / (2*m.pi) # replace with proper value when known
#print("Lift slope:", a3D*180/m.pi, "and tau:", tau)
exp_Cdi = exp_Cl**2 / (m.pi * A * tau)


# Preparing for plotting
linear_alpha = np.linspace(np.min(exp_alpha), np.max(exp_alpha), 100)
linear_cl = linear_alpha * a3D + intercept


# ----- PLotting -----
# Plotting lift curve
# plt.title('Lift curve')
plt.plot(exp_alpha[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_alpha, xflr_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()

# Plotting lift polar
# plt.title('Lift polar')
plt.plot(exp_Cd[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_Cd, xflr_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()

# Plotting CLalpha best fit line
# plt.title('Lift polar')
plt.plot(exp_alpha[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(linear_alpha, linear_cl, linestyle='dotted', linewidth=1, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'Linear region line of best fit'))
plt.show()

# Plotting induced drag
# plt.title('Lift polar')
plt.plot(exp_Cdi[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_Cdi, xflr_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}_{\text{i}}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()