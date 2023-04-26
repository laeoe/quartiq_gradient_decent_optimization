import numpy as np
from qutip import basis, enr_thermal_dm, destroy, mesolve, expect, wigner, plot_wigner, Options
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from function_definitions import potential, hamiltonian, initial_state, initial_energy, left_boundary, right_boundary, range_of_motion, time_partial_recompression, evolve, simulate, extract_wigner, marginal, mean_values, variances, energy, negativity_volume
from function_definitions import plot_mean_value_and_std_deviation, plot_potential, plot_wigner, plot_trace, plot_energy_error, plot_negativity_volume, plot_probability_last_two_fock_states
from function_definitions import number_of_fock_states, set_numerical_par

def get_negativity_volume(physical_parameters):
    s = True

    n_x_points_per_unit = 10
    n_p_points_per_unit = 10
    n_time_points_per_unit = 10
    tolerance = {
        "delta_x" : 5,
        "delta_p" : 5,
        "n_fock" : 0
    }

    numerical_parameters = set_numerical_par(n_x_pu = n_x_points_per_unit, n_p_pu = n_p_points_per_unit, 
                                            n_t_pu = n_time_points_per_unit, phys_par = physical_parameters, tolerance=tolerance)

    if s == True:
        largest_fock_state = basis(numerical_parameters["dimension_Hilbert_space"], numerical_parameters["dimension_Hilbert_space"]-1)

    parameters = {
        "physical": physical_parameters,
        "numerical" : numerical_parameters
    }
    final_time = time_partial_recompression(parameters)
    parameters["physical"]["final_time"] = final_time
    parameters["numerical"]["time_steps"] = int(final_time*10)

    rho_list = simulate(parameters)

    # stuf that i dont know if needed:
    
    if s == True:
        rho_f = rho_list[-1]
        mean_x, mean_p = mean_values(rho_f, numerical_parameters)
        var_x, var_p = variances(rho_f, numerical_parameters)
        time_step = parameters["numerical"]["time_steps"]-1
    

        

    return negativity_volume(rho_list[-1], parameters["numerical"])


# physical_parameters = {
#     "n_bar" : 0,
#     "x_s" : 0,
#     "alpha" : 1e-3,
#     "beta" : 0,
#     "final_time" : 10
# }

# physical_parameters = {
#     "n_bar" : 0,
#     "x_s" : 0,
#     "alpha" : 1e-3,
#     "beta" : 0.5,
#     "final_time" : 10
# }

# physical_parameters = {
#     "n_bar" : 5,
#     "x_s" : 0,
#     "alpha" : 1e-3,
#     "beta" : 0,
#     "final_time" : 10
# }

physical_parameters = {
    "n_bar" : 0,
    "x_s" : np.linspace(0, 4, 20),
    "alpha" : 1,
    "beta" : 0,
    "final_time" : 10
}

negativities = []
for xs in physical_parameters["x_s"]:
    physical_parameters["x_s"] = xs
    neg = get_negativity_volume(physical_parameters)
    negativities.append(neg)
    print("x_s: ", xs, "negativity: ", neg)


plt.plot(physical_parameters["x_s"], negativities)
plt.xlabel(r"$x_s$")
plt.ylabel(r"$\mathcal{N}$")
plt.show()

#print("Negativity volume at t_f: ", get_negativity_volume(physical_parameters))

