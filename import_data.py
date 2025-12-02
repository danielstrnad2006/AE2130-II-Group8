import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

"""
Get all relevant data from the .txt files for 2D data
"""


run_nr_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(0))
run_nr_2d = run_nr_2d.astype(int)
alpha_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(3))
rho_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(7))
pressure_pressureSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range(8,33))
ports_loc_pressureSide_2d = np.array([0,  0.0035626,  0.0133331,  0.0366108,  0.072922,  0.1135604,  0.1559135,  0.1991328,  0.2428443,  0.2868627,  0.3310518,  0.3753128,  0.4195991,  0.4638793,  0.508156,  0.552486,  0.5969223,  0.6413685,  0.68579,  0.7302401,  0.7747357,  0.8193114,  0.8638589,  0.908108,  1])
pressure_suctionSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (34, 58))
ports_loc_suctionSide_2d = np.array([0,  0.0043123,  0.0147147,  0.0392479,  0.0779506,  0.120143,  0.1632276,  0.2067013,  0.2503792,  0.2941554,  0.3379772,  0.3818675,  0.4257527,  0.4696278,  0.5135062,  0.5573662,  0.6012075,  0.6450502,  0.688901,  0.7328011,  0.7767783,  0.8207965,  0.8647978,  1]  )
pressure_total_wake_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (58, 105))
ports_loc_total_wake_2d = np.array([0, 0.012, 0.021, 0.027, 0.033, 0.039, 0.045, 0.051, 0.057, 0.063, 0.069, 0.072, 0.075, 0.078, 0.081, 0.084, 0.087, 0.09, 0.093, 0.096, 0.099, 0.102, 0.105, 0.108, 0.111, 0.114, 0.117, 0.12, 0.123, 0.126, 0.129, 0.132, 0.135, 0.138, 0.141, 0.144, 0.147, 0.15, 0.156, 0.162, 0.168, 0.174, 0.18, 0.186, 0.195, 0.207, 0.219
])
pressure_static_wake_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (106, 118))
ports_loc_static_wake_2d = np.array([0.0435, 0.0555, 0.0675, 0.0795, 0.0915, 0.1035, 0.1155, 0.1275, 0.1395, 0.1515, 0.1635, 0.1755])
pressure_pitot_total_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 105)
pressure_pitot_static_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 118)
# Debug print removed



class AirfoilTest: 
    def __init__(self, run_nr, alpha, rho, p_pressureSide, p_suctionSide, p_total_wake, p_static_wake, p_pitot_total, p_pitot_static):
        self.run_nr = run_nr
        self.alpha = alpha
        self.rho = rho
        self.pressure_pressureSide_distribution = interpolate.interp1d(ports_loc_pressureSide_2d, p_pressureSide, kind="quadratic", fill_value = "extrapolate")
        
        self.pressure_suctionSide_distribution = interpolate.interp1d(ports_loc_suctionSide_2d, p_suctionSide, kind="quadratic", fill_value = "extrapolate")
        
        self.pressure_static_wake_distribution = interpolate.interp1d(ports_loc_static_wake_2d, p_static_wake, kind="quadratic", fill_value = (p_static_wake[0], p_static_wake[-1]))
        self.pressure_total_wake_distribution = interpolate.interp1d(ports_loc_total_wake_2d, p_total_wake, kind="quadratic", fill_value = (p_total_wake[0], p_total_wake[-1]))

test_cases = {}
for i, run_nr in enumerate(run_nr_2d):
    airfoil_test = AirfoilTest(
        run_nr=run_nr,
        alpha=alpha_2d[i],
        rho=rho_2d[i],
        p_pressureSide=pressure_pressureSide_2d[i],
        p_suctionSide=pressure_suctionSide_2d[i],
        p_total_wake=pressure_total_wake_2d[i],
        p_static_wake=pressure_static_wake_2d[i],
        p_pitot_total=pressure_pitot_total_2d[i],
        p_pitot_static=pressure_pitot_static_2d[i]
    )
    test_cases[int(run_nr)] = airfoil_test

# Example plot for run number 5
x_axis = np.linspace(0, 1, 100)

plt.plot( x_axis, test_cases[30].c_p_pressureSide_distribution(x_axis))
plt.point( )
plt.plot( x_axis, test_cases[30].c_p_suctionSide_distribution(x_axis))
plt.show()
