import numpy as np
import matplotlib.pyplot as plt

xflr_data_2d = np.genfromtxt('XFLR5_data_2D_Re=2e5.csv', delimiter=",", skip_header=5)
xflr_alpha = xflr_data_2d[:,0] # deg
xflr_Cl = xflr_data_2d[:,2] # [-]
xflr_Cd = xflr_data_2d[:,1] # [-]
xflr_Cm = xflr_data_2d[:,3] # [-]


def plot_cp_at_alpha(alpha: float):
    n = int(alpha*2 + 10)
    header = int(7 + 186*n)
    footer = int(7258 - header - 181)
    print(header, footer, 7258-footer, 7258-footer-header)
    data = np.genfromtxt(r'XFLR5_fulldata_2D_Re=2e5.csv', delimiter = ',', skip_header= header, skip_footer = footer, usecols=(0, 1))
    print(data[:, 0])
    return len(data[:, 0])


# ----- PLotting -----
# Plotting lift curve
# plt.title('Lift curve')
# plt.plot(xflr_alpha[:-1], xflr_Cl[:-1], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
# plt.grid()
# plt.xlabel(r'$\alpha$ [deg]')
# plt.ylabel(r'C$_{\text{l}}$ [-]')
# plt.legend(('XFoil'))
# plt.show()

print(plot_cp_at_alpha(-5))
print(plot_cp_at_alpha(-4.5))