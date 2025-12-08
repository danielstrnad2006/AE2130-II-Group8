import numpy as np
import math as m
import matplotlib.pyplot as plt

from constants import const
from experimental_2d import test_cases
from xflr_2d import plot_cp_at_alpha, xflr_alpha_2d, xflr_Cd_2d, xflr_Cl_2d


# ----- Loading Data -----
exp_data_3d = np.genfromtxt('raw_Group8_3d.txt', skip_header=2)
exp_alpha = exp_data_3d[:,2] # deg
exp_drag_F = exp_data_3d[:,6] # N
exp_lift_F = exp_data_3d[:,7] # N
exp_rho = exp_data_3d[:, 10] # kg/m^3
exp_delta_pb = exp_data_3d[:, 3]


xflr_data_3d = np.genfromtxt('T1-19.2ms.csv', delimiter=",", skip_header=7)
xflr_alpha = xflr_data_3d[:,0] # deg
xflr_Cl = xflr_data_3d[:,2] # [-]
xflr_Cd = xflr_data_3d[:,5] # [-]
xflr_Cdi = xflr_data_3d[:, 3] # [-]

exp_2D_alpha = np.array([test_cases[i].alpha for i in test_cases])
exp_2D_Cl = np.array([test_cases[i].c_lift for i in test_cases])

# ----- Calculations -----
#exp_q = 1/2 * exp_rho * const['velocity']**2 
exp_q = 0.211804 + 1.928442 * (exp_delta_pb) + 1.879374e-4 * (exp_delta_pb)**2
exp_Cl = exp_lift_F / (exp_q * const['surface_area_3D'])
exp_Cd = exp_drag_F / (exp_q * const['surface_area_3D'])

A = const['span_3D']/const['chord']
print(A)
print(A)

# Finding slope of 2d
key = (exp_2D_alpha > 0) & (exp_2D_alpha < 10)
a2D, intercept_2d = np.polyfit(exp_2D_alpha[key], exp_2D_Cl[key], 1)  # [1/deg], [-]
a2D = a2D*180/m.pi
#a2D += 0.5
print(a2D)

# Finding slope of 3d
key = (exp_alpha > 0) & (exp_alpha < 10)
a3D, intercept_3d = np.polyfit(exp_alpha[key], exp_Cl[key], 1)  # [1/deg], [-]
a3D = a3D*180/m.pi
print(a3D)

# Finding tau
tau = (m.pi * A / a2D) * (a2D / a3D - 1) - 1
#print("Lift slope:", a3D*180/m.pi, "and tau:", tau)
e = 1/(1+tau)
exp_Cdi = exp_Cl**2 / (m.pi * A * e)
print(tau)


# Preparing for plotting
linear_alpha_2D = np.linspace(np.min(exp_alpha), np.max(exp_alpha), 100)
linear_cl_2D = linear_alpha_2D * a2D * m.pi/180 + intercept_2d

linear_alpha_3D = np.linspace(np.min(exp_alpha), np.max(exp_alpha), 100)
linear_cl_3D = linear_alpha_3D * a3D * m.pi/180 + intercept_3d


# ----- PLotting -----
# Plotting 2D vs 3D data
plt.plot(exp_2D_alpha, exp_2D_Cl, marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(exp_alpha, exp_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{l}}$ or C$_{\text{L}}$ [-]')
plt.legend(('2D wing experimental', '3D wing experimental'))
plt.show()


# Plotting lift curve
plt.plot(exp_alpha[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_alpha, xflr_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()

# Plotting lift polar
plt.plot(exp_Cd[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_Cd, xflr_Cl, marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()

# Plotting Clalpha best fit line
plt.plot(exp_2D_alpha, exp_2D_Cl, marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(linear_alpha_2D, linear_cl_2D, linestyle='dotted', linewidth=1, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{l}}$ [-]')
plt.legend(('Experiment', 'Linear region line of best fit'))
plt.show()

# Plotting CLalpha best fit line
plt.plot(exp_alpha[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(linear_alpha_3D, linear_cl_3D, linestyle='dotted', linewidth=1, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'Linear region line of best fit'))
plt.show()

# Plotting induced drag
plt.plot(exp_Cdi[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(exp_Cd[:27], exp_Cl[:27], marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}}$ or C$_{\text{D}_{\text{i}}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Induced Drag', 'Total Drag'))
plt.show()

# Plotting induced drag comparison
plt.plot(exp_Cdi[:27], exp_Cl[:27], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_Cdi[:-1], xflr_Cl[:-1], marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'C$_{\text{D}_{\text{i}}}$ [-]')
plt.ylabel(r'C$_{\text{L}}$ [-]')
plt.legend(('Experiment', 'XFLR5'))
plt.show()

# Plotting 2d xflr vs experimental
plt.plot(exp_2D_alpha[:29], exp_2D_Cl[:29], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.plot(xflr_alpha_2d[:-1], xflr_Cl_2d[:-1], marker='^', markersize=5,markerfacecolor='lightblue', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
plt.grid()
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'C$_{\text{l}}$ [-]')
plt.legend(('Experiment', 'XFOIL'))
plt.show()

# Plotting pressure coefficient distribution at a given AOA
AOA = float(input("AOA for pressure coefficients comparison [deg]: "))
plot_cp_at_alpha(AOA)
x_axis = np.linspace(0, 1, 100)
if AOA <= 8:
    i = (AOA + 5) * 2 + 1
else:
    i = (AOA - 8) * 2 + 14
plt.plot(x_axis, test_cases[i].cp_normal_pressureSide_distribution(x_axis), label="Experiment", color='black')
plt.plot(x_axis, test_cases[i].cp_normal_suctionSide_distribution(x_axis), color='black')
plt.xlabel("Position along chord [-]")
plt.ylabel(r"Pressure coefficient c$_{\text{p}}$ [-]")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True, axis="both")
plt.axhline(y=0, color='k', linewidth=0.5)
plt.show()