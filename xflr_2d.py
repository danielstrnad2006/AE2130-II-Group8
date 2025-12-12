import numpy as np
import matplotlib.pyplot as plt

xflr_data_2d = np.genfromtxt('XFLR5_data_2D_Re=2e5.csv', delimiter=",", skip_header=5)
xflr_alpha_2d = xflr_data_2d[:,0] # deg
xflr_Cl_2d = xflr_data_2d[:,2] # [-]
xflr_Cd_2d = xflr_data_2d[:,1] # [-]
#xflr_Cm = xflr_data_2d[:,3] # [-]

# Couldn't make it work with np.genfromtxt so had to make this helper function
def extract_raw_rows(filename, start, end):
    with open(filename, "r") as f:
        lines = f.readlines()
    rows = lines[start:end]

    result = []
    for line in rows:
        line = line.strip()
        print(line)
        parts = line.split(",")    # split by comma
        pair = [float(parts[0]), float(parts[1])]
        result.append(pair)
    
    return np.array(result)

def plot_cp_at_alpha(alpha: float, viscous: bool = True):
    n = int(alpha*2 + 10)
    start = int(7 + 186*n)
    pressure_coeffs = extract_raw_rows("XFLR5_fulldata_2D_Re=2e5.csv", start, start + 180)

    cp_inviscous, cp_viscous = pressure_coeffs[:, 0], pressure_coeffs[:, 1]

    airfoil_coords = np.loadtxt("SD6060-104-88_180.dat", comments='#', dtype=float)
    x = airfoil_coords[:, 0]
    y = airfoil_coords[:, 1]

    #plt.plot(x, y, linestyle='-', linewidth=1.5, color='black')
    if viscous:
        plt.plot(x, cp_viscous, linestyle='-', linewidth=1.5, label = 'XFOIL')
    else: plt.plot(x, cp_inviscous, linestyle='-', linewidth=1.5)
    #plt.axis("equal")
    plt.grid()
    plt.gca().invert_yaxis()

    #print(pressure_coeffs)



# ----- PLotting -----
# Plotting lift curve
# plt.title('Lift curve')
# plt.plot(xflr_alpha[:-1], xflr_Cl[:-1], marker='o', markersize=5,markerfacecolor='orange', markeredgecolor='black', linestyle='-', linewidth=1.5, color='black')
# plt.grid()
# plt.xlabel(r'$\alpha$ [deg]')
# plt.ylabel(r'C$_{\text{l}}$ [-]')
# plt.legend(('XFoil'))
# plt.show()

# print(plot_cp_at_alpha(-5))
# print(plot_cp_at_alpha(-4.5))