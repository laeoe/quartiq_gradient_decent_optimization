import numpy as np
from qutip import  enr_thermal_dm, destroy, mesolve, expect, wigner, Options
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
plt.rcParams['text.usetex'] = True



# PHYSICS
# -------


def potential(x, phys_par):

    """
    Returns the potential energy of the system evaluated at x.

    Parameters
    ----------
    x : float / Qobj
        Point / Hilbert space operator.
        
    phys_par : dictionary
               List of the physical parameters of the simulation.

    Returns
    -------
    Based on the input, either: a float being the value of the potential energy at x / a Qobj representing the potential energy operator in the Hilbert space.
    """

    beta = phys_par["beta"]
    alpha = phys_par["alpha"]
    x_s = phys_par["x_s"]

    V2 = beta*x*x/4
    xr = x - x_s
    V4 = alpha*xr*xr*xr*xr/4

    return V2 + V4


def hamiltonian(par):

    """
    Returns the Hamiltonian of the system.

    Parameters
    ----------
    par : dictionary
          List of the physical parameters of the simulation.

    Returns
    -------
    An instance of qutip.Qobj representing the Hamiltonian of the system.
    """

    b = destroy(par["numerical"]["dimension_Hilbert_space"])
    x = b.dag() + b
    p = 1.j*(b.dag() - b)
    
    K = p*p/4
    V = potential(x, par["physical"])
    H = K + V

    return H


def energy(state, par):

    """
    Returns the energy of the system in the input state.

    Parameters
    ----------
    state : Qobj
            Density matrix of the system.

    par : dictionary
          List of the physical parameters of the simulation.

    Returns
    -------
    A float representing the energy of the system.
    """

    H = hamiltonian(par)

    E = expect(H, state)

    return E


def initial_state(par):

    """
    Returns the density matrix for the thermal initial state.

    Parameters
    ----------
    par : dictionary
          List of the parameters of the simulation.

    Returns
    -------
    An instance of qutip.Qobj representing the thermal state of the system at time t=0, initialized according to the input parameters.
    """

    n_bar = par["physical"]["n_bar"]
    d_H = par["numerical"]["dimension_Hilbert_space"]

    rho = enr_thermal_dm([d_H], d_H, n_bar)

    return rho


def initial_energy(par):

    """
    Returns the energy of the system at time t=0.

    Parameters
    ----------
    par : dictionary
          List of the physical parameters of the simulation.

    Returns
    -------
    A float representing the initial energy of the system.
    """

    rho_0 = initial_state(par)

    H = hamiltonian(par)

    E0 = expect(H, rho_0)

    return E0


def left_boundary(par):

    """
    Returns the left boundary of the region of motion in x.

    Parameters
    ----------
    par : dictionary
          List of the parameters of the simulation.

    Returns
    -------
    A float representing the left boundary of the classical region of motion in x, based on energy conservation.
    """

    beta = par["physical"]["beta"]
    alpha = par["physical"]["alpha"]
    xs = par["physical"]["x_s"]

    E0 = initial_energy(par)

    if beta == 0:
        x_l = -(np.sqrt(2)*E0**(1/4))/(alpha**(1/4)) + xs

    elif xs == 0:
        x_l = -np.sqrt(-beta + np.sqrt(beta**2 + 16*alpha*E0))/np.sqrt(2*alpha)

    else:
        x_l = xs - np.sqrt(4*xs**2 - (2*(beta + 6*alpha*xs**2))/(3.*alpha) + (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/(3.*alpha*(beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 +
        128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) +
        (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 -
        alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha))/2. - np.sqrt(8*xs**2 - (4*(beta + 6*alpha*xs**2))/(3.*alpha) - (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/
        (3.*alpha*(beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 -
        alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) - (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 +
        1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha) - (96*xs**3 - (16*xs*(beta +
        6*alpha*xs**2))/alpha)/(4.*np.sqrt(4*xs**2 - (2*(beta + 6*alpha*xs**2))/(3.*alpha) + (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/(3.*alpha*(beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 +
        128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) +
        (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 +
        192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha))))/2.

    return x_l


def right_boundary(par):

    """
    Returns the right boundary of the region of motion in x.

    Parameters
    ----------
    par : dictionary
          List of the physical parameters of the simulation.

    Returns
    -------
    A float representing the right boundary of the classical region of motion in x, based on energy conservation.
    """
    
    beta = par["physical"]["beta"]
    alpha = par["physical"]["alpha"]
    xs = par["physical"]["x_s"]

    E0 = initial_energy(par)

    if beta == 0:
        x_r = (np.sqrt(2)*E0**(1/4))/(alpha**(1/4)) + xs

    elif xs == 0:
        x_r = np.sqrt(-beta + np.sqrt(beta**2 + 16*alpha*E0))/np.sqrt(2*alpha)

    else:
        x_r = xs - np.sqrt(4*xs**2 - (2*(beta + 6*alpha*xs**2))/(3.*alpha) + (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/(3.*alpha*(beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 +
        6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 +
        192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) + (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 +
        128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 -
        16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha))/2. + np.sqrt(8*xs**2 - (4*(beta + 6*alpha*xs**2))/(3.*alpha) - (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/(3.*alpha*(beta**3 + 144*alpha*beta*E0 +
        18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 -
        alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) - (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 +
        128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 -
        16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha) - (96*xs**3 - (16*xs*(beta + 6*alpha*xs**2))/alpha)/(4.*np.sqrt(4*xs**2 - (2*(beta + 6*alpha*xs**2))/(3.*alpha) +
        (beta**2 - 48*alpha*E0 + 12*alpha*beta*xs**2)/(3.*alpha*(beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 +
        1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 - alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333) +
        (beta**3 + 144*alpha*beta*E0 + 18*alpha*beta**2*xs**2 + 6*np.sqrt(3)*np.sqrt(4*alpha*beta**4*E0 + 128*alpha**2*beta**2*E0**2 + 1024*alpha**3*E0**3 + 80*alpha**2*beta**3*E0*xs**2 - 768*alpha**3*beta*E0**2*xs**2 -
        alpha**2*beta**4*xs**4 + 192*alpha**3*beta**2*E0*xs**4 - 16*alpha**3*beta**3*xs**6))**0.3333333333333333/(3.*alpha))))/2.

    return x_r


def range_of_motion(par):

    """
    Returns the boundaries of the 2D region of motion in the phase space.

    Parameters
    ----------
    par : dictionary
          List of the physical parameters of the simulation.

    Returns
    -------
    x_range: list
             Left and right boundaries of the classical region of motion in x.
             
    p_range: list
             Left and right boundaries of the classical region of motion in p.
    """
    
    x_l = left_boundary(par)
    x_r = right_boundary(par)

    x_range = [x_l, x_r]

    E0 = initial_energy(par)

    p_range = [-2*np.sqrt(E0), 2*np.sqrt(E0)]

    return x_range, p_range


def time_partial_recompression(par):

    """
    Returns an approximation of the time required by the wave packet to bounce at the quartic walls and compress again to the original size.

    Parameters
    ----------
    par : dictionary
          List of the parameters of the simulation.

    Computation
    -----------
    The time for partial recompression is roughly approximated as follows:

    1. The system is treated as a classical particle with velocity equal to the thermal average of the initial distribution:
       v_0 = sigma_p/m
       where sigma_p is the standard deviation in p of the initial phase space distribution and m is the mass of the particle;

    2. The potential is treated as a potential well with infinite walls placed at the boundaries of the region of motion, x_l and x_r;

    3. After these approximations, the time of partial recompression can be estimated as:
       t_pc = (x_r - x_l)/v0

    Returns
    -------
    A float representing the approximated time for partial recompression.
    """

    n_bar = par["physical"]["n_bar"]

    sigma_p = np.sqrt(2*n_bar + 1)

    range_x = range_of_motion(par)[0]
    delta_x = range_x[1] - range_x[0]

    time_p_r = delta_x/sigma_p

    return time_p_r


def evolve(rho_0, par, internal_steps=3000, max_steps=100000):

    """
    Returns the state of the system after evolution.

    Parameters
    ----------
    rho_0 : QObj
            Density matrix for the initial state of the system.

    par : dictionary
          List of the parameters of the simulation.

    internal_steps: int
                    Number of computational steps set as input to the master equation solver of qutip (mesolve), at the first attempt.

    internal_steps: int
                    Maximum number of computational steps allowed to the master equation solver.

    Returns
    -------
    A qutip.Result instance containing result.states, a list of the system density matrices at the instants of time specified in the dictionary par.
    """

    t_steps = par["numerical"]["time_steps"]
    t_f = par["physical"]["final_time"]
    d_H = par["numerical"]["dimension_Hilbert_space"]
    
    times = np.linspace(0, t_f, t_steps)

    b = destroy(d_H)
    x = b.dag() + b
    p = 1.j*(b.dag() - b)

    K = p*p/4
    V = potential(x, par["physical"])
    H = K + V

    while internal_steps <= max_steps:
        try:
            return mesolve(H, rho_0, times, options=Options(nsteps=internal_steps))
        except Exception:
            internal_steps += 1000
            continue


def simulate(par):

    """
    Prepares the system to the thermal initial state and computes its evolution.

    Parameters
    ----------
    par : dictionary
          List of the parameters of the simulation.

    Returns
    -------
    A qutip.Result instance containing result.states, a list of the system density matrices at the instants of time specified in the dictionary par.
    """
    
    rho_0 = initial_state(par)

    result = evolve(rho_0, par)
    rhos = result.states

    return rhos


def extract_wigner(rho, num_par):

    """
    Computes the wigner function associated to the input state.

    Parameters
    ----------
    rho : Qobj
          Density matrix of the system.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    Returns
    -------
    An array of the values representing the Wigner function calculated over the range specified in num_par.
    """

    x_c = num_par["x_coordinates"]
    p_c = num_par["p_coordinates"]

    return wigner(rho, x_c, p_c, g=1)


def marginal(wigner, num_par):

    """
    Computes the marginal probability distributions along x and p from the input Wigner function.

    Parameters
    ----------
    wigner : array
             Array of the values of the Wigner function on the discretized phase space.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    Note!
    -----
    The marginals are computed correctly only if the x and p ranges are large enough to make the Wigner function fit inside the grid.

    Returns
    -------
    marginal_x : array
                 Array of the values of the marginal probability in x.
    marginal_p : array
                 Array of the values of the marginal probability in p.
    """

    dp = np.mean(np.diff(num_par["p_coordinates"]))
    marginal_x = dp*np.sum(wigner, 0)
    
    dx = np.mean(np.diff(num_par["x_coordinates"]))
    marginal_p = dx*np.sum(wigner, 1)

    return marginal_x, marginal_p


def mean_values(state, num_par):

    """
    Computes the expectation values of the x and p operators for the input state:
    <x>, <p>

    Parameters
    ----------
    state : Qobj
            Density matrix of the system.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    Returns
    -------
    mean_x : float
             Expectation value of x.
    mean_p : float
             Expectation value of p.
    """

    b = destroy(num_par["dimension_Hilbert_space"])
    x = b.dag() + b
    p = 1.j*(b.dag() - b)
    
    mean_x = expect(x, state)
    mean_p = expect(p, state)

    return mean_x, mean_p


def variances(state, num_par):

    """
    Computes the variances of the input state in x and p.
    <x^2> - <x>^2, <p^2> - <p>^2

    Parameters
    ----------
    state : Qobj
            Density matrix of the system.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    Returns
    -------
    variance_x : float
                 Variance in x.
    variance_p : float
                 Variance in p.
    """

    b = destroy(num_par["dimension_Hilbert_space"])
    x = b.dag() + b
    p = 1.j*(b.dag() - b)
    
    variance_x = expect(x*x, state) - (expect(x, state))**2
    variance_p = expect(p*p, state) - (expect(p, state))**2

    return variance_x, variance_p


def negativity_volume(rho, num_par):

    """
    Computes the negativity volume of the Wigner function associated with the state rho.

    Parameters
    ----------
    rho : Qobj
          Density matrix of the system.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    Returns
    -------
    negativity : float
                 Negativity volume of the Wigner function of the system.
    """

    wigner = extract_wigner(rho, num_par)
    dx = np.mean(np.diff(num_par["x_coordinates"]))
    dp = np.mean(np.diff(num_par["p_coordinates"]))

    negativity = (1/2)*np.sum(np.abs(wigner)-wigner)*dx*dp
    
    return negativity



# PLOTTING
# --------


def plot_potential(par):

    """
    Plots the potential energy of the system as a function of x, together with the total energy of the system.

    Parameters
    ----------
    par : dictionary
          List of the parameters of the simulation.
    """

    x_c = par["numerical"]["x_coordinates"]
    Vx = potential(x_c, par["physical"])

    E0 = initial_energy(par)

    fontsize = 20

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$x / x_0$', fontsize=fontsize)
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(x_c, Vx, label=r'$V (x) / \hbar \Omega$')
    ax.plot(x_c, x_c*0 + E0, label=r'$\mathrm{tr} (\hat{H} \hat{\rho} (0)) / \hbar \Omega$')
    plt.legend(loc='upper right', fontsize=fontsize)


def plot_mean_value_and_std_deviation(rho_list, par):

    """
    Plots the mean value and the standard deviation in x and p as a function of the time of evolution.

    Parameters
    ----------
    rho_list : list of Qobj
               List of the density matrices of the system throughout evolution.

    par : dictionary
          List of the parameters of the simulation.
    """

    times = np.linspace(0, par["physical"]["final_time"], par["numerical"]["time_steps"])
    mvx_list = []
    sdx_list = []
    mvp_list = []
    sdp_list = []

    for r in rho_list:
        mvx, mvp = mean_values(r, par["numerical"])
        vx, vp = variances(r, par["numerical"])
        mvx_list.append(mvx)
        sdx_list.append(np.sqrt(vx))
        mvp_list.append(mvp)
        sdp_list.append(np.sqrt(vp))
    
    fontsize = 20

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$\Omega t$', fontsize=fontsize)
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(times, mvx_list, label=r'$[ \langle \hat{x} \rangle - \sigma_x, \langle \hat{x} \rangle + \sigma_x ]/x_0$')
    ax.fill_between(times, np.asarray(mvx_list) - np.asarray(sdx_list), np.asarray(mvx_list) + np.asarray(sdx_list), alpha=0.5)
    ax.plot(times, mvp_list, label=r'$[ \langle \hat{p} \rangle - \sigma_p, \langle \hat{p} \rangle + \sigma_p ]/p_0$')
    ax.fill_between(times, np.asarray(mvp_list) - np.asarray(sdp_list), np.asarray(mvp_list) + np.asarray(sdp_list), alpha=0.5)
    plt.legend(loc='upper right', fontsize=fontsize)


def plot_wigner(rho, num_par, time=None, show_marg_x=True, show_marg_p=True):

    """
    Plots the Wigner function of the system at a given time, along with the marginal distributions in x and p.

    Parameters
    ----------
    rho : QObj
          Density matrices of the system.

    num_par : dictionary
              List of the numerical parameters of the simulation.

    time : float or None
           Instant of time corresponding to the input state rho (None, if the state is not a result of evolution).
    """
    
    wigner = extract_wigner(rho, num_par)

    x_c = num_par["x_coordinates"]
    p_c = num_par["p_coordinates"]

    marginal_x, marginal_p = marginal(wigner, num_par)

    cmap = "seismic"
    fontsize = 20
    shape = (3, 13)
    half_range_w = np.max(abs(wigner))

    fig = plt.figure(figsize=(9, 8))

    ax_w = plt.subplot2grid(shape=shape, loc=(1, 0), colspan=8, rowspan=2)
    ax_w.contourf(x_c, p_c, wigner, levels=100, vmin=-half_range_w, vmax=half_range_w, cmap=cmap)
    ax_w.set(title='', xlim=(min(x_c), max(x_c)), ylim=(min(p_c), max(p_c)))
    ax_w.set_xlabel(r'$x / x_0$', fontsize=fontsize)
    ax_w.set_ylabel(r'$p / p_0$', fontsize=fontsize)
    ax_w.tick_params(direction="in", top=True, right=True, labelsize=fontsize)

    if show_marg_x:
        ax_marg_x = plt.subplot2grid(shape=shape, loc=(0, 0), colspan=8)
        ax_marg_x.plot(x_c, marginal_x)
        ax_marg_x.set(xlim=(min(x_c), max(x_c)), ylim=(0, max(0.8, 1.1*max(marginal_x))))
        ax_marg_x.set_ylabel(r'$P(x) x_0$', fontsize=fontsize)
        ax_marg_x.tick_params(direction="in", top=True, right=True, labelbottom=False, labelsize=fontsize)

    if show_marg_p:
        ax_marg_p = plt.subplot2grid(shape=shape, loc=(1, 8), rowspan=2, colspan=4)
        ax_marg_p.plot(marginal_p, p_c)
        ax_marg_p.set(ylim=(min(p_c), max(p_c)), xlim=(0, max(0.8, 1.1*max(marginal_p))))
        ax_marg_p.set_xlabel(r'$P(p) p_0$', fontsize=fontsize)
        ax_marg_p.tick_params(direction="in", top=True, right=True, labelleft=False, labelsize=fontsize)

    
    if show_marg_p:
        ax_colorbar = plt.subplot2grid(shape=shape, loc=(1, 12), rowspan=2)
    else:
        ax_colorbar = plt.subplot2grid(shape=shape, loc=(1, 8), rowspan=2)
    crange = np.linspace(-0.3, 0.3, x_c.size, endpoint=True)
    cplot = np.repeat(crange[None], p_c.size, axis=0)
    colorbar_ghost_plot = ax_colorbar.contourf(x_c, p_c, cplot, levels=100, vmin=np.min(crange), vmax=np.max(crange), cmap=cmap)
    cbar = fig.colorbar(colorbar_ghost_plot, cax=ax_colorbar, format=tick.FormatStrFormatter('$%.2f$'))
    ax_colorbar.tick_params(labelsize=fontsize)

    if time is not None:
        ax_time = plt.subplot2grid(shape=shape, loc=(0, 8), colspan=5)
        ax_time.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)
        ax_time.spines['bottom'].set_visible(False)
        ax_time.spines['left'].set_visible(False)
        ax_time.text(0.2, 0.13, '$\Omega t = %.1f$ \n \n' % time, fontsize=fontsize)
        fig.tight_layout()


def plot_trace(rho_list, par):

    """
    Plots the discrepancy between the trace of the evolving state and unity as a function of time.

    Parameters
    ----------
    rho_list : list of Qobj
               List of the density matrices of the system throughout evolution.

    par : dictionary
          List of the parameters of the simulation.
    """

    times = np.linspace(0, par["physical"]["final_time"], par["numerical"]["time_steps"])
    trace_list = []

    for r in rho_list:
        trace_list.append(np.absolute(r.tr()-1))
    
    fontsize = 20

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$\Omega t$', fontsize=fontsize)
    ax.set_ylabel(r'$\vert \mathrm{tr} (\hat{\rho} (t)) - 1 \vert$', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(times, trace_list)


def plot_energy_error(rho_list, par):

    """
    Plots the relative error between the energy of the evolving state and the initial energy as a function of time.

    Parameters
    ----------
    rho_list : list of Qobj
               List of the density matrices of the system throughout evolution.

    par : dictionary
          List of the parameters of the simulation.
    """

    times = np.linspace(0, par["physical"]["final_time"], par["numerical"]["time_steps"])
    energy_list = []

    E0 = energy(rho_list[0], par)

    for r in rho_list:
        energy_list.append(np.absolute((energy(r, par) - E0)/E0))
    
    fontsize = 20

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$\Omega t$', fontsize=fontsize)
    ax.set_ylabel(r'$\vert E(t) - E (0))/E (0) \vert$', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(times, energy_list)


def plot_probability_last_two_fock_states(rho_list, par):

    """
    Plots the sum of the probability to find the system in the two highest Fock states as a function of time, namely
    <N|\rho|N> + <N-1|\rho|N-1>

    Parameters
    ----------
    rho_list : list of Qobj
               List of the density matrices of the system throughout evolution.

    par : dictionary
          List of the parameters of the simulation.
    """

    times = np.linspace(0, par["physical"]["final_time"], par["numerical"]["time_steps"])
    p_list = []

    for r in rho_list:
        p_list.append(np.absolute(r[-1, -1])**2 + np.absolute(r[-2, -2])**2)
    
    fontsize = 20

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$\Omega t$', fontsize=fontsize)
    ax.set_ylabel(r'$\langle N \vert \rho \vert N \rangle + \langle N-1 \vert \rho \vert N-1 \rangle$', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(times, p_list)


def plot_negativity_volume(rho_list, par):

    """
    Plots the negativity volume of the evolving Wigner function as a function of time.

    Parameters
    ----------
    rho_list : list of Qobj
               List of the density matrices of the system throughout evolution.

    par : dictionary
          List of the parameters of the simulation.
    """
    
    times = np.linspace(0, par["physical"]["final_time"], par["numerical"]["time_steps"])
    negativity_list = []

    for r in rho_list:
        negativity_list.append(negativity_volume(r, par["numerical"]))
    
    fontsize = 20

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlabel(r'$\Omega t$', fontsize=fontsize)
    ax.set_ylabel(r'$\mathcal{N}$', fontsize=fontsize)
    ax.tick_params(direction="in", top=True, right=True, labelsize=fontsize)
    ax.plot(times, negativity_list)



# ESTIMATION OF THE NUMERICAL PARAMETERS
# --------------------------------------


def number_of_fock_states(par):

    """
    Returns the minimal number of Fock states necessary (but not always sufficient!) to correctly simulate the state evolution.

    Parameters
    ----------
    par : dictionary
          List of the physical parameters of the simulation.

    Computation
    -----------
    The estimation of the minimal number of Fock states is based on the following argument.
    The spatial extent of a Fock state scales like:
    sigma_{x, n} = sqrt(2 n + 1)
    where n is the number of phonons.
    Given the size of the range of motion delta_x, one then needs all the Fock states up to n_max such that
    n_max = (delta_x^2 - 1)/2

    Returns
    -------
    An int being the minimal number of required Fock states.
    """

    range_x = range_of_motion(par)[0]
    delta_x = range_x[1] - range_x[0]

    n_fock = ((delta_x**2)-1)/2

    return int(n_fock)


def set_numerical_par(n_x_pu, n_p_pu, n_t_pu, phys_par, tolerance = {"delta_x" : 0, "delta_p" : 0, "n_fock" : 0}):

    """
    Initializes and returns the dictionary of the numerical parameters.

    Parameters
    ----------
    n_x : int
          Number of points per unit in x.

    n_p : int
          Number of points per unit in p.

    n_t : int
          Number of points per unit in time.

    phys_par : dictionary
               List of the physical parameters of the simulation.

    tolerance : dict
                Dictionary for the adjustment of the x- and p-ranges and the number of Fock states.
                "delta_x" : float
                            Interval to be added to the left and to the right of the automatically computed x-range.
                "delta_p" : float
                            Interval to be added to the left and to the right of the automatically computed p-range.
                "n_fock" : int
                           Number to be added to the automatically computed minimal number of Fock states.

    Returns
    -------
    Dictionary of the numerical parameters, being automatically estimated through the functions 'range_of_motion' and 'number_of_fock_states', and corrected manually through the input 'tolerance'.
    """

    final_time = phys_par["final_time"]
    n_bar = phys_par["n_bar"]

    num_par_0 = {
        "x_coordinates" : np.linspace(3*np.sqrt(2*n_bar + 1), 3*np.sqrt(2*n_bar + 1), 100),
        "p_coordinates" : np.linspace(3*np.sqrt(2*n_bar + 1), 3*np.sqrt(2*n_bar + 1), 100),
        "dimension_Hilbert_space" : 10*(n_bar+1),
        "time_steps": 0
        }

    par_0 = {
    "physical": phys_par,
    "numerical" : num_par_0
    }
    
    range_x, range_p = range_of_motion(par_0)
    n_fock = number_of_fock_states(par_0)

    delta_x = (range_x[1] - range_x[0]) + 2*tolerance["delta_x"]
    delta_p = range_p[1] - range_p[0] + 2*tolerance["delta_p"]
    n_x = int(delta_x*n_x_pu)
    n_p = int(delta_p*n_p_pu)
    n_t = int(final_time*n_t_pu)

    num_par = {
        "x_coordinates" : np.linspace(range_x[0] - tolerance["delta_x"], range_x[1] + tolerance["delta_x"], n_x),
        "p_coordinates" : np.linspace(range_p[0] - tolerance["delta_p"], range_p[1] + tolerance["delta_p"], n_p),
        "dimension_Hilbert_space" : n_fock + tolerance["n_fock"],
        "time_steps": n_t
        }

    # print("Numerical parameters\n")
    # print("x-range: (", num_par["x_coordinates"][0], " -- ", num_par["x_coordinates"][-1], ') x_0')
    # print("Number of points along x: ", n_x)
    # print("p-range: (", num_par["p_coordinates"][0], " -- ", num_par["p_coordinates"][-1], ') p_0')
    # print("Number of points along p: ", n_p)
    # print("Dimension of the Hilbert space: ", num_par["dimension_Hilbert_space"])
    # print("Number of points in time: ", num_par["time_steps"])

    return num_par