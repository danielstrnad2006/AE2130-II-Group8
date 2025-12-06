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



"""
Initialize an object for each test scenario which automatically calculates all the pressure distributions and coefficients.
"""

class AirfoilTest: 
    def __init__(self, run_nr, alpha, rho, p_pressureSide, p_suctionSide, p_total_wake, p_static_wake, p_pitot_total, p_pitot_static, p_barometric, delta_p_b):
        self.run_nr = run_nr
        
        self.dynamic_pressure = 0.211804 + 1.928442 * (delta_p_b) + 1.879374e-4 * (delta_p_b)**2
        
        self.static_pressure = p_barometric * 100
        self.alpha = alpha
        self.rho = rho
        c_p_axial__suctionSide_arr = np.array([p_suctionSide[i]*coefs_axial_suctionSide_2d[i]/self.dynamic_pressure for i in range(len(ports_loc_suctionSide_2d))])
        c_p_normal__suctionSide_arr = np.array([p_suctionSide[i]*coefs_normal_suctionSide_2d[i]/self.dynamic_pressure for i in range(len(ports_loc_suctionSide_2d))])
        c_p_axial__pressureSide_arr = np.array([p_pressureSide[i]*coefs_axial_pressureSide_2d[i]/self.dynamic_pressure for i in range(len(ports_loc_pressureSide_2d))])
        c_p_normal__pressureSide_arr = np.array([p_pressureSide[i]*coefs_normal_pressureSide_2d[i]/self.dynamic_pressure for i in range(len(ports_loc_pressureSide_2d))])

        
        self.cp_axial_suctionSide_distribution = interpolate.interp1d(ports_loc_suctionSide_2d, c_p_axial__suctionSide_arr, kind="linear", fill_value = "extrapolate")
        self.cp_normal_suctionSide_distribution = interpolate.interp1d(ports_loc_suctionSide_2d, c_p_normal__suctionSide_arr, kind="linear", fill_value = "extrapolate")
        
        self.cp_axial_pressureSide_distribution = interpolate.interp1d(ports_loc_pressureSide_2d, c_p_axial__pressureSide_arr, kind="linear", fill_value = "extrapolate")
        self.cp_normal_pressureSide_distribution = interpolate.interp1d(ports_loc_pressureSide_2d, c_p_normal__pressureSide_arr, kind="linear", fill_value = "extrapolate")
        

        self.pressure_static_wake_distribution = interpolate.interp1d(ports_loc_static_wake_2d, p_static_wake, kind="linear", fill_value = (p_static_wake[0], p_static_wake[-1]), bounds_error=False)
        self.pressure_total_wake_distribution = interpolate.interp1d(ports_loc_total_wake_2d, p_total_wake, kind="linear", fill_value = (p_total_wake[0], p_total_wake[-1]), bounds_error=False)

        domain = np.linspace(0,1, 200)
        integrate_c_p_axial_suctionSide = self.cp_axial_suctionSide_distribution(domain)
        integrate_c_p_normal_suctionSide = self.cp_normal_suctionSide_distribution(domain)
        integrate_c_p_axial_pressureSide = self.cp_axial_pressureSide_distribution(domain)
        integrate_c_p_normal_pressureSide = self.cp_normal_pressureSide_distribution(domain)

    
        self.c_normal = - sp.integrate.trapezoid(integrate_c_p_normal_suctionSide, x=domain) + sp.integrate.trapezoid(integrate_c_p_normal_pressureSide, x=domain)
        self.c_axial = sp.integrate.trapezoid(integrate_c_p_axial_suctionSide, x=domain) + sp.integrate.trapezoid(integrate_c_p_axial_pressureSide, x=domain)

        self.c_lift = self.c_normal * np.cos(np.deg2rad(self.alpha)) - self.c_axial * np.sin(np.deg2rad(self.alpha))
        self.c_drag = self.c_normal * np.sin(np.deg2rad(self.alpha)) + self.c_axial * np.cos(np.deg2rad(self.alpha))

    
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
        delta_p_b=delta_p_2d[i]
    )
    test_cases[int(run_nr)] = airfoil_test


if __name__ == "__main__":
# Example plots for user to visualize results

# Plot c_p distribution for user-specified run numbers
    
    i = None
    i = int(input("Enter run number to plot c_p distribution (e.g., 1-41), otherwise press enter: "))
    if i in range(1,42):
        x_axis = np.linspace(0, 1, 100)
        plt.figure()
        plt.plot(x_axis, test_cases[i].cp_normal_pressureSide_distribution(x_axis), label="pressure side")
        plt.plot(x_axis, test_cases[i].cp_normal_suctionSide_distribution(x_axis), label="suction side")
        plt.xlabel("Position along chord")
        plt.ylabel("Pressure coefficient (c_p)")
        plt.title(f"c_p distribution at angle of attack = {test_cases[i].alpha}°")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid(True, axis="both")
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.show()
        print ("Normal force coefficient is computed to be: ", test_cases[i].c_normal)
        print ("Axial force coefficient is computed to be: ", test_cases[i].c_axial)

# Plot Wake pressure distributions for user-specified run numbers
    i = None
    i = int(input("Enter run number to plot c_p distribution (e.g., 1-41), otherwise press enter: "))
    if i in range(1,42):
        x_axis = np.linspace(-0.10, 0.30, 200)
        plt.figure()
        plt.plot(x_axis, test_cases[i].pressure_total_wake_distribution(x_axis), label="wake total pressure")
        plt.plot(x_axis, test_cases[i].pressure_static_wake_distribution(x_axis), label="wake static pressure")
        plt.xlabel("Position along the wake [m]")
        plt.ylabel("Pressure [Pa]")
        plt.title(f"pressure in wake distribution at angle of attack = {test_cases[i].alpha}°")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid(True, axis="both")
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.show()


    if input("Do you want to plot c_lift vs alpha? (y/n): ") == "y":
        x_axis = [test_cases[i].alpha for i in test_cases]
        c_lift_axis = [test_cases[i].c_lift for i in test_cases]
        plt.figure()
        plt.plot(x_axis, c_lift_axis, marker='o')
        plt.xlabel("Angle of attack (degrees)")
        plt.ylabel("Lift coefficient (c_lift)")
        plt.title("Lift coefficient vs Angle of attack")
        plt.grid(True, axis="both")
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.show()


if input("Do you want to plot c_lift vs c_drag? (y/n): ") == "y":
        x_axis = [test_cases[i].c_drag for i in test_cases]
        c_lift_axis = [test_cases[i].c_lift for i in test_cases]
        plt.figure()
        plt.plot(x_axis, c_lift_axis, marker='o')
        plt.xlabel("Drag coefficient (c_drag)")
        plt.ylabel("Lift coefficient (c_lift)")
        plt.title("Lift coefficient vs Drag coefficient")
        plt.grid(True, axis="both")
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.show()

