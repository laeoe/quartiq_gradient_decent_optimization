import numpy as np
from qutip import basis, enr_thermal_dm, destroy, mesolve, expect, wigner, plot_wigner, Options
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from function_definitions import potential, hamiltonian, initial_state, initial_energy, left_boundary, right_boundary, range_of_motion, time_partial_recompression, evolve, simulate, extract_wigner, marginal, mean_values, variances, energy, negativity_volume
from function_definitions import plot_mean_value_and_std_deviation, plot_potential, plot_wigner, plot_trace, plot_energy_error, plot_negativity_volume, plot_probability_last_two_fock_states
from function_definitions import number_of_fock_states, set_numerical_par


# start_physical_parameter_range = {
#     "n_bar" : [0],
#     "x_s" : (0, 2),
#     "alpha" : (1e-3, 1),
#     "beta" : (0, 1),
#     "final_time" : 10
# }
start_physical_parameter_range = {
    "n_bar" : [0],
    "x_s" : (1, 3),
    "alpha" : (1e-3, 1),
    "beta" : (0, 1),
    "final_time" : 10
}

physical_parameter_range = {
    "n_bar" : [0, 1, 2, 3, 4],
    "x_s" : (0, 3.5),
    "alpha" : (1e-3, 1),
    "beta" : (0, 1),
    "final_time" : 10
}


def get_negativity_volume(physical_parameters):
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

    parameters = {
        "physical": physical_parameters,
        "numerical" : numerical_parameters
    }
    final_time = time_partial_recompression(parameters)
    parameters["physical"]["final_time"] = final_time
    parameters["numerical"]["time_steps"] = int(final_time*10)

    rho_list = simulate(parameters)

    return negativity_volume(rho_list[-1], parameters["numerical"])



#print("Negativity volume at t_f: ", get_negativity_volume(physical_parameters))


# create a random set of physical parameters within the physical_parameter_range as starting params
def random_start(param_ranges):
    physical_parameters = {}
    for key in param_ranges:
        if key == "final_time":
            physical_parameters[key] = param_ranges[key]
        elif key == "n_bar":
            #physical_parameters[key] = np.random.choice(physical_parameter_range[key])
            physical_parameters[key] = 0
        else:
            physical_parameters[key] = np.random.uniform(param_ranges[key][0], param_ranges[key][1])
    return physical_parameters






# calculate the gradient of the negativity volume with respect to the physical parameters
def gradient_negativity_volume(params):
    delta = 1e-8
    gradient = {}
    n0 = get_negativity_volume(params)
    for key in params:
        if key == "final_time":
            gradient[key] = 0
        
        elif key == "n_bar":
            gradient[key] = 0
            # if params[key] < 4:
            #     params[key] += 1
            #     gradient[key] = (n0 - get_negativity_volume(params))
            #     params[key] -= 1

            # elif params[key] == 4:
            #     params[key] -= 1
            #     gradient[key] = (get_negativity_volume(params) - n0)
            #     params[key] += 1

        else:
            params[key] += delta
            gradient[key] = (n0 - get_negativity_volume(params)) / delta
            params[key] -= delta
    return gradient, n0







def grad_decent_step(initial_param, grad_step):
    grad_vec, initial_params_neg = gradient_negativity_volume(initial_param)
    new_param = {}
    for key in initial_param:
        if key == "final_time":
                new_param[key] = initial_param[key]
        elif key == "n_bar":
            new_param[key] = initial_param[key] - grad_vec[key]
            if new_param[key] < physical_parameter_range[key][0]:
                new_param[key] = physical_parameter_range[key][0]
            elif new_param[key] > physical_parameter_range[key][1]: 
                new_param[key] = physical_parameter_range[key][1]

        else:
            new_param[key] = initial_param[key] - grad_step * grad_vec[key]

            #check if the new parameter is within range
            if new_param[key] < physical_parameter_range[key][0]:
                new_param[key] = physical_parameter_range[key][0]

            elif new_param[key] > physical_parameter_range[key][1]:
                new_param[key] = physical_parameter_range[key][1]
    
    return new_param, grad_vec, initial_param, initial_params_neg





if __name__ == "__main__":
    grad_step = 2e-4
    #starting_params = random_physical_parameters()
    # starting_params = {
    # "n_bar" : 0,
    # "x_s" : 9,
    # "alpha" : 1e-3,
    # "beta" : 0,
    # "final_time" : 10
    # }
    # starting_params = {
    # "n_bar" : 0,
    # "x_s" : 2,
    # "alpha" : 0.99,
    # "beta" : 0.001,
    # "final_time" : 10
    # }
    starting_params = random_start(start_physical_parameter_range)
    print("initial Starting params: ", starting_params)

    negativity = []
    pars = []

    max_n_steps = 50
    tolerance = 1e-6
    for i in range(max_n_steps):
        new_params, grad_vec, initial_params, initial_params_neg = grad_decent_step(starting_params, grad_step)
        change_in_params = {key: new_params[key] - initial_params[key] for key in new_params}
        # break if the change in parameters is less than the tolerance
        
        if i  == 0:
            print("Iteration: ", i)
            print("Initial params: ", initial_params)
            print("Initial params neg: ", initial_params_neg)
            #print("New params: ", new_params)
            #print("Gradient vector: ", grad_vec)
            #print("change in params: ", {key: new_params[key] - initial_params[key] for key in new_params})
            print("")
        
        negativity.append(initial_params_neg)
        pars.append(starting_params)
        print("Iteration: ", i)
        print("negativity: ", initial_params_neg)

        if all(abs(change_in_params[key]) < tolerance for key in change_in_params):
            print("Change in params is less than tolerance")
            print("Iteration: ", i)
            print("final params: ", initial_params)
            print("final params neg: ", initial_params_neg)
            break



        starting_params = new_params


# plt.plot(negativity)
# plt.ylabel('negativity volume')
# plt.xlabel('iteration')
# plt.grid()
# plt.title("Negativity volume vs iteration")

# create a plot of the negativity volume for each point vs the parameters plots for x_s, alpha, beta. 
# the points whould be coloured by a colormap in order to visualize the gradient decent path

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# label the global plot 
fig.suptitle("Negativity volume vs parameters")

# create a colormap for the points
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=0, vmax=len(pars))
colors = [cmap(norm(i)) for i in range(len(pars))]
# plot the points
for i in range(len(pars)):
    axs[0].scatter(pars[i]["x_s"], negativity[i], color=colors[i])
    axs[1].scatter(pars[i]["alpha"], negativity[i], color=colors[i])
    axs[2].scatter(pars[i]["beta"], negativity[i], color=colors[i])

# show the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
#fig.colorbar(sm, ax=axs.ravel().tolist())

# label the colorbar vertically with the iteration number
cbar = fig.colorbar(sm, ax=axs.ravel().tolist())
cbar.ax.set_ylabel("iteration", rotation=270, labelpad=20)



# axs[0].plot([pars[i]["x_s"] for i in range(len(pars))], negativity)
# axs[0].set_title("Negativity volume vs x_s")
axs[0].set_xlabel("x_s")
axs[0].set_ylabel("Negativity volume")
axs[0].grid()

# axs[1].plot([pars[i]["alpha"] for i in range(len(pars))], negativity)
# axs[1].set_title("Negativity volume vs alpha")
axs[1].set_xlabel("alpha")
axs[1].set_ylabel("Negativity volume")
axs[1].grid()

# axs[2].plot([pars[i]["beta"] for i in range(len(pars))], negativity)
# axs[2].set_title("Negativity volume vs beta")
axs[2].set_xlabel("beta")
axs[2].set_ylabel("Negativity volume")
axs[2].grid()







# save the parameters with the greatest negativity volume and the corresponding negativity volume

max_negativity = max(negativity) 
arg_max_negativity = negativity.index(max_negativity)

with open("best_params.txt", "w") as f:
    f.write("Parameters with the greatest negativity volume: \n")
    f.write("Negativity volume: " + str(max_negativity) + "\n")
    for key in pars[arg_max_negativity]:
        f.write(key + " = " + str(pars[arg_max_negativity][key]) + "\n")
    
plt.show()



