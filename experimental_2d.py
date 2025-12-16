import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

"""
Get all relevant data from the .txt files for 2D data
"""


run_nr_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(0))
run_nr_2d = run_nr_2d.astype(int)
alpha_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(2))
rho_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(7))
p_bar_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(4))
delta_p_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(3))
pressure_suctionSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range(8,33))
ports_loc_suctionSide_2d = np.array([0,  0.0035626,  0.0133331,  0.0366108,  0.072922,  0.1135604,  0.1559135,  0.1991328,  0.2428443,  0.2868627,  0.3310518,  0.3753128,  0.4195991,  0.4638793,  0.508156,  0.552486,  0.5969223,  0.6413685,  0.68579,  0.7302401,  0.7747357,  0.8193114,  0.8638589,  0.908108,  1])
pressure_pressureSide_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (33, 57))
ports_loc_pressureSide_2d = np.array([0,  0.0043123,  0.0147147,  0.0392479,  0.0779506,  0.120143,  0.1632276,  0.2067013,  0.2503792,  0.2941554,  0.3379772,  0.3818675,  0.4257527,  0.4696278,  0.5135062,  0.5573662,  0.6012075,  0.6450502,  0.688901,  0.7328011,  0.7767783,  0.8207965,  0.8647978,  1]  )

y_contrib_suctionSide_2d = np.array([0.000038577, 0.0000800575, 0.0001053025, 0.000127796, 0.0001126315, 0.00008465, 0.000062229, 0.0000445705, 0.00003, 0.0000175355, 6.37249999999998E-06, -4.14100000000004E-06, -0.0000142985, -0.0000242025, -0.0000337015, -0.000042139, -0.000048886, -0.0000544005, -0.0000593975, -0.00006366, -0.0000668585, -0.0000686435, -0.000068136, -0.000097423, -0.0000638345
])
y_contrib_pressureSide_2d = np.array([0.0028588, 0.00546375, 0.00600135, 0.00639975, 0.00497405, 0.0032738, 0.0021592, 0.00139345, 0.000821499999999999, 0.000354099999999999, -4.96999999999992E-05, -0.00041595, -0.000763750000000001, -0.0011086, -0.00144845, -0.0017891, -0.00213685, -0.00247445, -0.0027874, -0.00304505, -0.0032151, -0.003308, -0.00641365, -0.0047437
])
pressure_total_wake_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (57, 104))
ports_loc_total_wake_2d = np.array([0, 0.012, 0.021, 0.027, 0.033, 0.039, 0.045, 0.051, 0.057, 0.063, 0.069, 0.072, 0.075, 0.078, 0.081, 0.084, 0.087, 0.09, 0.093, 0.096, 0.099, 0.102, 0.105, 0.108, 0.111, 0.114, 0.117, 0.12, 0.123, 0.126, 0.129, 0.132, 0.135, 0.138, 0.141, 0.144, 0.147, 0.15, 0.156, 0.162, 0.168, 0.174, 0.18, 0.186, 0.195, 0.207, 0.219
])
pressure_static_wake_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=range (105, 117))
ports_loc_static_wake_2d = np.array([0.0435, 0.0555, 0.0675, 0.0795, 0.0915, 0.1035, 0.1155, 0.1275, 0.1395, 0.1515, 0.1635, 0.1755])
pressure_pitot_total_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 104)
pressure_pitot_static_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols= 117)


coefs_axial_suctionSide_2d = np.array([1, 0.90788566767644, 0.647248613962005, 0.480809723951739, 0.332336934691063, 0.232881546972098, 0.167560947918414, 0.120525970990327, 0.0835927888996176, 0.0529308034674796, 0.026556005831046, 0.00227287544682064, -0.0209680149513494, -0.0435649973021308, -0.0655725915358173, -0.0860924250616744, -0.102904497066308, -0.115764066166278, -0.127272744463318, -0.137696804286988, -0.145690500767598, -0.151220937074955, -0.153271412402153, -0.150095671569553, -0.137611962892752])
coefs_normal_suctionSide_2d = np.array([0, 0.419217860339591, 0.762278972374461, 0.876824959358168, 0.943160729589686, 0.972505107996807, 0.985861718869681, 0.992710174379632, 0.996499997814342, 0.998598182475958, 0.999647327088059, 0.999997417015266, 0.999780147006831, 0.999050594819935, 0.99784780164075, 0.99628715456288, 0.994691240779535, 0.993276739375613, 0.991867757574958, 0.990474426771914, 0.989330216856883, 0.988499988968221, 0.988184129674348, 0.988671476971033, 0.990486217808609])


coefs_axial_pressureSide_2d = np.array([1, 0.798381793639415, 0.44781173529397, 0.266842260814572, 0.153364969894653, 0.0930099904875881, 0.0603799247157211, 0.0393542209962685, 0.0245976759304083, 0.0129808461683564, 0.00319245870964433, -0.00545214872686113, -0.0135021824989098, -0.0213033777989277, -0.0292114749708079, -0.0367876672323331, -0.0447439612866815, -0.0526176511070318, -0.0600681188641278, -0.0667300495959605, -0.0715374993158656, -0.074220753673238, -0.0756868536204657, -0.0699998036767682])
coefs_normal_pressureSide_2d = np.array([0, 0.602151568614672, 0.894127871019019, 0.963740218027331, 0.98816961398801, 0.995665175482953, 0.998175467886946, 0.999225322582338, 0.999697431395531, 0.999915745266947, 0.999994904090709, 0.999985136926675, 0.999908841378936, 0.999773057295683, 0.999573253808859, 0.999323104676162, 0.998998487450495, 0.998614731912152, 0.99819428023613, 0.997771066167445, 0.997437910945655, 0.997241836128116, 0.997131636339472, 0.997547005150742])

temperature_2d = np.genfromtxt('raw_Group8_2d.txt', skip_header=2, usecols=(5))


"""
Initialize an object for each test scenario which automatically calculates all the pressure distributions and coefficients.
"""

class AirfoilTest: 
    def __init__(self, run_nr, alpha, rho, p_pressureSide, p_suctionSide, p_total_wake, p_static_wake, p_pitot_total, p_pitot_static, p_barometric, delta_p_b, temperature):
        self.run_nr = run_nr
        
        self.dynamic_pressure = 0.211804 + 1.928442 * (delta_p_b) + 1.879374e-4 * (delta_p_b)**2
        
        ### Is this black magic or correct ???
        self.static_pressure = p_pitot_total - self.dynamic_pressure  #+ p_barometric * 100 
        ###
        print("Static pressure:", self.static_pressure, "Dynamic pressure:", self.dynamic_pressure)

        self.alpha = alpha
        self.rho = rho

        self.temperature = temperature + 273.15  # Convert to Kelvin
        self.u_inf = np.sqrt(2 * self.dynamic_pressure / rho)
        self.viscosity = 1.716e-5 * (self.temperature / 273.15)**1.5 * (273.15 + 110.4) / (self.temperature + 110.4)  # Sutherland's law

        self.reynolds_number = (rho * self.u_inf * 0.16) / self.viscosity  # Calculation of Reynolds number based on chord length of 16 cm
        print("Temp:", self.temperature,"U_inf:", self.u_inf, "Re:", self.reynolds_number, "density:", self.rho)

        c_p_suctionSide_arr = np.array([(p_suctionSide[i]-self.static_pressure)/self.dynamic_pressure for i in range(len(ports_loc_suctionSide_2d))])
        c_p_pressureSide_arr = np.array([(p_pressureSide[i]-self.static_pressure)/self.dynamic_pressure for i in range(len(ports_loc_pressureSide_2d))])

        
        self.cp_normal_suctionSide_distribution = interpolate.interp1d(x=ports_loc_suctionSide_2d, y=c_p_suctionSide_arr, kind=1, fill_value = 0)
        
        self.cp_normal_pressureSide_distribution = interpolate.interp1d(x=ports_loc_pressureSide_2d, y=c_p_pressureSide_arr, kind=1, fill_value = 0)
        
        self.c_axial = -(sum([c_p_suctionSide_arr[i] * y_contrib_suctionSide_2d[i] for i in range(len(ports_loc_suctionSide_2d))]) + sum([c_p_pressureSide_arr[i] * y_contrib_pressureSide_2d[i] for i in range(len(ports_loc_pressureSide_2d))]))
        
        domain = np.linspace(0,1, 300)
        integrate_c_p_normal_suctionSide = self.cp_normal_suctionSide_distribution(domain)
        integrate_c_p_normal_pressureSide = self.cp_normal_pressureSide_distribution(domain)


    
        
        self.pressure_static_wake_distribution = interpolate.interp1d(ports_loc_static_wake_2d, p_static_wake, kind="linear", fill_value = (p_static_wake[0], p_static_wake[-1]), bounds_error=False)
        self.pressure_total_wake_distribution = interpolate.interp1d(ports_loc_total_wake_2d, p_total_wake, kind="linear", fill_value = (p_total_wake[0], p_total_wake[-1]), bounds_error=False)
        
        
        
        """
        Drag calculation based on wake survey
        """
        domain_wake = np.linspace(0.0, 0.219, 300)

        p_s_wake = self.pressure_static_wake_distribution(domain_wake)
        p_t_wake = self.pressure_total_wake_distribution(domain_wake)

        self.pitot_u_inf = np.sqrt(2 * (p_pitot_total - p_pitot_static) / self.rho)
        wake_p_t_inf = (p_t_wake[0]+p_t_wake[-1])/2
        wake_p_s_inf = (p_s_wake[0]+p_s_wake[-1])/2
        self.wake_u_inf = np.sqrt(2 * (wake_p_t_inf - wake_p_s_inf) / self.rho)
        # Wake velocity from Bernoulli
        self.u_wake = np.sqrt( 2* (p_t_wake - p_s_wake) / self.rho )
        self.u_wake_distribution = interpolate.interp1d(domain_wake, self.u_wake, kind="linear", fill_value = (self.u_wake[0], self.u_wake[-1]), bounds_error=False) 

        # Momentum deficit term
        momentum_deficit_pitot = self.rho * sp.integrate.trapezoid(
            (self.pitot_u_inf - self.u_wake) * self.u_wake,
            x=domain_wake
        )
        momentum_deficit_freestream = self.rho * sp.integrate.trapezoid(
            (self.u_inf - self.u_wake) * self.u_wake,
            x=domain_wake
        )
        momentum_deficit_wake = self.rho * sp.integrate.trapezoid(
            (self.wake_u_inf - self.u_wake) * self.u_wake,
            x=domain_wake
        )
        # Pressure deficit term (wake not fully recovered)
        pressure_deficit = sp.integrate.trapezoid(
            (self.static_pressure - p_s_wake),
            x=domain_wake
        )

        # Total drag 
        drag_integral_freestream =pressure_deficit  + momentum_deficit_freestream
        drag_integral_pitot = pressure_deficit  + momentum_deficit_pitot
        drag_integral_wake = pressure_deficit  + momentum_deficit_wake 
        # drag_integral = np.abs(momentum_deficit) + np.abs(pressure_deficit)

        self.c_drag_pitot = drag_integral_pitot / (0.5 * self.rho * self.wake_u_inf**2 * 0.16)
        self.c_drag_wake = drag_integral_wake / (0.5 * self.rho * self.wake_u_inf**2 * 0.16)
        self.c_drag = drag_integral_freestream / (0.5 * self.rho * self.u_inf**2 * 0.16)

        print()


        self.c_normal = - sp.integrate.trapezoid(integrate_c_p_normal_suctionSide, x=domain) + sp.integrate.trapezoid(integrate_c_p_normal_pressureSide, x=domain)
        
        self.c_moment_LE = - sp.integrate.trapezoid(integrate_c_p_normal_suctionSide*domain, x=domain) + sp.integrate.trapezoid(integrate_c_p_normal_pressureSide*domain, x=domain)
        self.c_moment_025c = self.c_moment_LE + 0.25 * self.c_normal

        self.c_lift_pressure = self.c_normal * np.cos(np.deg2rad(self.alpha)) - self.c_axial * np.sin(np.deg2rad(self.alpha))
        self.c_drag_pressure = self.c_normal * np.sin(np.deg2rad(self.alpha)) + self.c_axial * np.cos(np.deg2rad(self.alpha))
        print("c_normal:", self.c_normal, "c_axial:", self.c_axial)
        
        self.c_lift = self.c_normal * (np.cos(np.deg2rad(self.alpha)) + np.sin(np.deg2rad(self.alpha))**2/np.cos(np.deg2rad(self.alpha))) - self.c_drag * np.tan(np.deg2rad(self.alpha))
        
       
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
        p_pitot_static=pressure_pitot_static_2d[i],
        p_barometric=p_bar_2d[i],
        delta_p_b=delta_p_2d[i],
        temperature=temperature_2d[i]
    )
    test_cases[int(run_nr)] = airfoil_test


if __name__ == "__main__":
# Example plots for user to visualize results

# Plot c_p distribution for user-specified run numbers

    i = input("Enter run number to plot c_p distribution (e.g., 1-41), otherwise press enter: ")
    if i != "":
        i = int(i)
        if i in range(1,42):
        
            plt.figure()
            plt.plot(ports_loc_pressureSide_2d, test_cases[i].cp_normal_pressureSide_distribution(ports_loc_pressureSide_2d), label="c_p distribution over the pressure side", color = "red", marker = "x")
            plt.plot(ports_loc_suctionSide_2d, test_cases[i].cp_normal_suctionSide_distribution(ports_loc_suctionSide_2d), label="c_p distribution over the suction side", color = "blue", marker = "x")
            plt.xlabel("Position along chord")
            plt.ylabel("Pressure coefficient (c_p)")
            plt.title(f"c_p distribution at angle of attack = {test_cases[i].alpha}°, and Reynolds number = {test_cases[i].reynolds_number:.2e}")
            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid(True, axis="both")
            plt.axhline(y=0, color='k', linewidth=0.5)
            plt.show()
            print ("Normal force coefficient is computed to be: ", test_cases[i].c_normal)
            print ("Normal force coefficient is computed to be: ", test_cases[i].c_normal)

# Plot Wake pressure distributions for user-specified run numbers
    i = input("Enter run number to plot c_p distribution (e.g., 1-41), otherwise press enter: ")
    if i != "":
        i = int(i)
        if i in range(1,42):
            x_axis = np.linspace(0, 0.219, 100)
            fig, ax1 = plt.subplots()
            ax1.figure.set_size_inches(6, 4)

            ax1.plot(x_axis, test_cases[i].pressure_total_wake_distribution(x_axis), label="wake total pressure")
            ax1.plot(x_axis, test_cases[i].pressure_static_wake_distribution(x_axis), label="wake static pressure")
            ax1.set_xlabel("Position along the wake rake [m]")
            ax1.set_ylabel("Pressure relative to reference pressure [Pa]")
            ax1.invert_yaxis()
            ax1.grid(True, axis="both")
            ax1.axhline(y=0, color='k', linewidth=0.5)
            
            ax2 = ax1.twinx()
            domain_wake = np.linspace(0, 0.219, 200)
            integrate_pressure_static_wake = test_cases[i].pressure_static_wake_distribution(domain_wake)
            integrate_pressure_total_wake = test_cases[i].pressure_total_wake_distribution(domain_wake)
            u_wake_axis = test_cases[i].u_wake_distribution(domain_wake)
            
            ax2.axhline(y=test_cases[i].u_inf, color='r', linestyle='--', label=f"u_inf (obtained from dynamic pressure) = {test_cases[i].u_inf:.2f} m/s")
            ax2.axhline(y=test_cases[i].pitot_u_inf, color='b', linestyle='--', label=f"u_inf (by pitot-static tube) = {test_cases[i].pitot_u_inf:.2f} m/s")
            ax2.axhline(y=test_cases[i].wake_u_inf, color='g', linestyle='--', label=f"u_inf (by total pressure at edges of wake rake) = {test_cases[i].wake_u_inf:.2f} m/s")
            ax2.plot(domain_wake, u_wake_axis, color='orange', label="u_wake [m/s]")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.legend(loc="upper right")
            
            ax1.legend(loc="lower left")
            plt.title(f"pressure in wake distribution at angle of attack = {test_cases[i].alpha}°, \n and Reynolds number = {test_cases[i].reynolds_number:.2e}")
            plt.show()
    
    if input("Do you want to plot c_lift vs alpha? (y/n): ") == "y":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        x_axis = [test_cases[i].alpha for i in test_cases]

        c_lift_pressure_axis = [test_cases[i].c_lift_pressure for i in test_cases]
        c_lift_axis = [test_cases[i].c_lift for i in test_cases]
        
        ax1.plot(x_axis, c_lift_pressure_axis, label="Lift polar (from pressure distribution)", marker='x', color="green")
        ax1.plot(x_axis, c_lift_axis, label="Lift polar  (incl. viscous drag)", marker='x', color="purple")
        ax1.set_xlabel("Angle of attack (degrees)")
        ax1.set_ylabel("Lift coefficient")
        ax1.set_title("Lift coefficient vs Angle of attack")
        ax1.grid(True, axis="both")
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.legend()
        
        x_axis_drag_pressure = [test_cases[i].c_drag_pressure for i in test_cases]
        x_axis_drag_freestream = [test_cases[i].c_drag for i in test_cases]
        x_axis_drag_wake = [test_cases[i].c_drag_wake for i in test_cases]
        x_axis_drag_pitot = [test_cases[i].c_drag_pitot for i in test_cases]
        ax2.plot(x_axis_drag_pressure, c_lift_pressure_axis, label="Drag polar(from pressure distribution)", marker='x', color="green")
        ax2.plot(x_axis_drag_freestream, c_lift_axis, label="Drag polar(incl. viscous drag), using the reference freestream velocity", marker='x', color="purple")
        ax2.plot(x_axis_drag_wake, c_lift_axis, label="Drag polar (incl. viscous drag), using the freestream velocity at the edges of the wake", marker='x', color="orange")
        ax2.plot(x_axis_drag_pitot, c_lift_axis, label="Drag polar (incl. viscous drag), using the freestream velocity from the pitot-static tube", marker='x', color="red")
        ax2.set_xlabel("Drag coefficient (c_drag)")
        ax2.set_ylabel("Lift coefficient (c_lift)")
        ax2.set_title("Lift coefficient vs Drag coefficient")
        ax2.grid(True, axis="both")
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

