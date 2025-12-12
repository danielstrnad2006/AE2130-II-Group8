import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from constants import const

# --- Get the relevant info from experimental data ---

run_nr = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(0))
run_nr = run_nr.astype(int)
alpha = np.deg2rad(np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(2)))
rho = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(7))
p_bar = 100 * np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(4))
delta_p = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(3))
pressure_suctionSide = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range(8,33))
ports_loc_suctionSide = np.array([0,  0.0035626,  0.0133331,  0.0366108,  0.072922,  0.1135604,  0.1559135,  0.1991328,  0.2428443,  0.2868627,  0.3310518,  0.3753128,  0.4195991,  0.4638793,  0.508156,  0.552486,  0.5969223,  0.6413685,  0.68579,  0.7302401,  0.7747357,  0.8193114,  0.8638589,  0.908108,  1])
pressure_pressureSide = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (33, 57))
ports_loc_pressureSide = np.array([0,  0.0043123,  0.0147147,  0.0392479,  0.0779506,  0.120143,  0.1632276,  0.2067013,  0.2503792,  0.2941554,  0.3379772,  0.3818675,  0.4257527,  0.4696278,  0.5135062,  0.5573662,  0.6012075,  0.6450502,  0.688901,  0.7328011,  0.7767783,  0.8207965,  0.8647978,  1]  )


pressure_total_wake = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (57, 104))
ports_loc_total_wake = np.array([0, 0.012, 0.021, 0.027, 0.033, 0.039, 0.045, 0.051, 0.057, 0.063, 0.069, 0.072, 0.075, 0.078, 0.081, 0.084, 0.087, 0.09, 0.093, 0.096, 0.099, 0.102, 0.105, 0.108, 0.111, 0.114, 0.117, 0.12, 0.123, 0.126, 0.129, 0.132, 0.135, 0.138, 0.141, 0.144, 0.147, 0.15, 0.156, 0.162, 0.168, 0.174, 0.18, 0.186, 0.195, 0.207, 0.219
])
pressure_static_wake = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (105, 117))
ports_loc_static_wake = np.array([0.0435, 0.0555, 0.0675, 0.0795, 0.0915, 0.1035, 0.1155, 0.1275, 0.1395, 0.1515, 0.1635, 0.1755])
pressure_pitot_total = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 104)
pressure_pitot_static = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 117)

temperature = 273.15 + np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(5))

# --- Import Constants ---
R = 287.05


mu_0 = 1.716e-5
T_0 = 27.15
S = 110.4


# --- Classes ---
class ExperimentRun:
    def __init__(self, run_nr) -> None:
        self.run_idx = run_nr - 1
        self.T = temperature[self.run_idx]
        self.alpha = alpha[self.run_idx]

    def calculate_constants(self) -> None:
        # Density
        self.rho = (p_bar[self.run_idx])/(R*self.T)
        # self.rho2 = rho[self.run_idx]
        # print(self.rho, self.rho2)

        # Dynamic viscosity w/ Sutherlands law
        self.mu = mu_0 * (self.T/T_0)**(3/2) * (T_0+S)/(self.T+S)
        # print(self.mu)

        # Dynamic pressure q_infinity
        self.q = 0.211804 + 1.928442*delta_p[self.run_idx] + 1.879374e-4* delta_p[self.run_idx]**2
        # print(self.q)

        # Static pressure
        self.p_static = pressure_pitot_total[self.run_idx] - self.q
        # print(self.p_static, pressure_pitot_static[self.run_idx])
        # -> We can see that the difference between measured data and the sensor data is about 14.3 Pa

        # Velocity
        self.V = m.sqrt((2*self.q)/self.rho)
        #print(self.V)

    def calculate_coefficients(self) -> None:
        # Find pressure coefficients, notice that the pressure given by the ports is already a pressure difference
        Cp_upper = pressure_suctionSide[self.run_idx, :] / self.q   # suction side (upper)
        Cp_lower = pressure_pressureSide[self.run_idx, :] / self.q  # pressure side (lower)
        x_upper = ports_loc_suctionSide
        x_lower = ports_loc_pressureSide        
        # print(c_p_suctionSide_arr, c_p_pressureSide_arr)

        # Split the integral into two for an easy calculation of normal coefficient
        Cn = np.trapezoid(Cp_lower, x_lower) - np.trapezoid(Cp_upper, x_upper)
        self.Cn = Cn
        # print(self.Cn)

        # Calculate moment coefficient around 0.25c
        Cm_lower = -np. trapezoid(Cp_lower * ports_loc_pressureSide, ports_loc_pressureSide)
        Cm_upper = +np. trapezoid(Cp_upper * ports_loc_suctionSide, ports_loc_suctionSide)
        self.Cm = (Cm_lower + Cm_upper) + 0.25*self.Cn

        # Calculate drag coefficient - 1. find the pressures
        y_t  = ports_loc_total_wake
        pt_y = pressure_total_wake[self.run_idx, :]
        y_s  = ports_loc_static_wake
        p_y  = pressure_static_wake[self.run_idx, :]
        # 2. Interpolate wake static pressure onto total-wake y-grid
        p_on_t = sp.interpolate.interp1d(y_s, p_y, kind="linear", bounds_error=False, fill_value=(p_y[0], p_y[-1]))(y_t)
        # 3. Find local velocity from pitot: U(y) = sqrt(2*(pt - p)/rho)
        q_y = np.clip(pt_y - p_on_t, 0.0, None)
        U_y = np.sqrt(2.0 * q_y / self.rho)
        # print(U_y)
        # Find drag
        D = self.rho * np.trapezoid(U_y * (self.V - U_y), y_t) + np.trapezoid(self.p_static - p_on_t, y_t)
        # print(D)
        self.Cd = D/(self.q * const['chord'])
        # print(self.Cd)

        # Calculate Cl
        self.Cl = self.Cn * (m.cos(self.alpha) + (m.sin(self.alpha)**2)/(m.cos(self.alpha))) - self.Cd * m.tan(self.alpha)
        # print(self.Cl)

# Run1 = ExperimentRun(1)
# Run1.calculate_constants()
# Run1.calculate_coefficients()


# Run2 = ExperimentRun(10)
# Run2.calculate_constants()
# Run2.calculate_coefficients()

Experiment = []
Cl = []
Cd = []
Cm = []
for i in run_nr:
    run = ExperimentRun(i)
    run.calculate_constants()
    run.calculate_coefficients()
    Cl.append(run.Cl)
    Cd.append(run.Cd)
    Cm.append(run.Cm)
    Experiment.append(run)

# plt.plot(np.rad2deg(alpha), Cl, label = 'Other code')
# plt.show()
