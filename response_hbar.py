import numba as nb
import numpy as np
import frenkel


def return_four_wavefunctions(ket, bra_star, initialize=True, beta=None, omega_j=None, bath_q=None, bath_p=None):
    """
    :param ket: ket (first) part of the density matrix: |a>, shape (n,)
    :param bra_star: bra (latter) part of the density matrix transformed to vector: |b>, shape (n,).
    complex-conjugated and transposed to form a vector |b>
    :param initialize: whether or not to initialize baths from scratch
    :param beta: 1/KT
    :param omega_j: the bath frequencies
    :param bath_q: bath normal mode positions
    :param bath_p: bath normal mode momenta
    :return: the four wavefunctions needed to propagate |ket><bra| in accordance with Ehrenfest
    dynamics
    """

    # system wavefunctions
    if np.shape(ket) != np.shape(bra_star):
        raise ValueError('ket and bra not in the same Hilbert space')

    # numerical stability
    eps = 1e-19

    # (a)
    norm_squared_a = np.dot(np.conj(ket), ket)
    wav_a = ket / (np.sqrt(norm_squared_a) + eps)
    # (b)
    norm_squared_b = np.dot(np.conj(bra_star), bra_star)
    wav_b = bra_star / (np.sqrt(norm_squared_b) + eps)
    # (c)
    wav_plus = ket + bra_star
    norm_squared_c = np.dot(np.conj(wav_plus), wav_plus)
    wav_c = wav_plus / (np.sqrt(norm_squared_c) + eps)
    # (d)
    wav_minus = ket + 1.0j * bra_star
    norm_squared_d = np.dot(np.conj(wav_minus), wav_minus)
    wav_d = wav_minus / (np.sqrt(norm_squared_d) + eps)

    # system baths
    if initialize:
        # initialize bath position and momenta from frequencies and temperature
        # sanity checks
        if len(beta.shape) != 2 or len(omega_j.shape) != 2:
            raise ValueError('Shape of beta and/or omega is incorrect')

        bath_q_a, bath_p_a = frenkel.harmonic_bath_initializer(beta, omega_j)
        bath_q_b, bath_p_b = frenkel.harmonic_bath_initializer(beta, omega_j)
        bath_q_c, bath_p_c = frenkel.harmonic_bath_initializer(beta, omega_j)
        bath_q_d, bath_p_d = frenkel.harmonic_bath_initializer(beta, omega_j)

    else:
        # copy over the provided bath momenta and positions
        # sanity checks
        if len(bath_q.shape) != 2 or len(bath_p.shape) != 2:
            raise ValueError('Provide proper bath momenta and/or positions')
        bath_q_a, bath_p_a = (np.copy(bath_q), np.copy(bath_p))
        bath_q_b, bath_p_b = (np.copy(bath_q), np.copy(bath_p))
        bath_q_c, bath_p_c = (np.copy(bath_q), np.copy(bath_p))
        bath_q_d, bath_p_d = (np.copy(bath_q), np.copy(bath_p))

    return {'a': [wav_a, norm_squared_a.real, bath_q_a, bath_p_a],
            'b': [wav_b, norm_squared_b.real, bath_q_b, bath_p_b],
            'c': [wav_c, norm_squared_c.real, bath_q_c, bath_p_c],
            'd': [wav_d, norm_squared_d.real, bath_q_d, bath_p_d]}


@nb.jit(nopython=True)
def ehrenfest_update(hs, vs, hbar, wavefunction, q_bath, p_bath, coupling_j,
                     bath_aux, timestep):
    """
    Update system and bath EOM over full timestep using split operator.

    - Analytical evolution of harmonic bath over a half timestep
    - Num. exact evolution of system over a full timestep
    - Analytical evolution of harmonic bath over a half timestep

    Input:
    1. hs -- [nhilbert, nhilbert]
    2. vs -- [nbaths, nhilbert, nhilbert]
    3. hbar -- scalar
    4. wavefunction -- [nhilbert]
    5. q_bath -- [nbaths, nosc]
    6. p_bath -- [nbaths, nosc]
    7. coupling_j -- [nbaths, nosc]
    8. bath_aux -- [4, nbaths, nosc]
    9. timestep -- scalar

    Output:
    1. wavefunction_new -- [nhilbert]
    2. q_new -- [nbaths, nosc]
    3. p_new -- [nbaths, nosc]
    """
    # Calculate the subsystem back-reaction on the bath
    system_force = frenkel.calculate_ehrenfest_system_force(wavefunction, vs)

    # Update the bath over a half timestep
    q_temp, p_temp = frenkel.update_harmonic_bath(q_bath, p_bath, system_force, bath_aux)

    # Update the bath back-reaction on the subsystem
    vbt = frenkel.calculate_vbt_linear(coupling_j, q_temp)

    # update subsystem Hamiltonian
    new_hs = frenkel.update_subsystem_hamiltonian(hs, vbt, vs)  # system-bath coupling

    # update wavefunction over a full timestep
    wavefunction_new = frenkel.update_wavefunction(new_hs, hbar, timestep, wavefunction)

    # Update the system back-reaction on the bath
    system_force = frenkel.calculate_ehrenfest_system_force(wavefunction, vs)

    # Update bath over a half timestep
    q_new, p_new = frenkel.update_harmonic_bath(q_temp, p_temp, system_force, bath_aux)

    return wavefunction_new, q_new, p_new


def return_density_matrix(wavefunction_dict):
    """
    :param wavefunction_dict: dictionary containing the four required wavefunctions and
                              their initial squared norms
    :return: density matrix corresponding to the set of wavefunctions: |i><j| form, not vice versa
    """
    # extract values
    wav_a = wavefunction_dict['a'][0]
    norm_squared_a = wavefunction_dict['a'][1]
    wav_b = wavefunction_dict['b'][0]
    norm_squared_b = wavefunction_dict['b'][1]
    wav_c = wavefunction_dict['c'][0]
    norm_squared_c = wavefunction_dict['c'][1]
    wav_d = wavefunction_dict['d'][0]
    norm_squared_d = wavefunction_dict['d'][1]

    before_norm = (norm_squared_c * np.outer(wav_c, np.conj(wav_c)) +
                   1.0j * norm_squared_d * np.outer(wav_d, np.conj(wav_d)) -
                   (1.0 + 1.0j) *
                   (norm_squared_a * np.outer(wav_a, np.conj(wav_a)) +
                   norm_squared_b * np.outer(wav_b, np.conj(wav_b))))

    return before_norm / 2.0


def act(ket, bra_star, operator):
    """
    :param ket: ket of the density matrix |k>, shape (nhilbert,)
    :param bra_star: bra of the density matrix <b|, shape (nhilbert,), complex conjugate transposed into |b>
    :param operator: list, contains operator matrix shape (nhilbert, nhilbert) and direction {'L', 'R'}
    :return: ket bra transformed operator
    """
    matrix = operator[0]
    direction = operator[1]

    assert(np.shape(matrix)[0] == np.shape(matrix)[1] == np.shape(ket)[0] == np.shape(bra_star)[0])

    if direction == 'L':
        # left-acting
        new_ket = np.dot(matrix, ket)
        new_bra_star = np.copy(bra_star)
    elif direction == 'R':
        new_ket = np.copy(ket)
        # complex conjugate to get <bra|
        bra = np.conj(bra_star)
        # apply the operator
        new_bra = np.dot(bra, matrix)
        # complex conjugate to get |new_bra_star>
        new_bra_star = np.conj(new_bra)

    return new_ket, new_bra_star


def aggregate(rho_a, rho_b, rho_c, rho_d, n_sa, n_sb, n_sc, n_sd):
    """
    :return: returns a density matrix given the constituent normalized density matrices and their
    squared magnitudes
    """
    return (0.5 * (n_sc * rho_c + 1.0j * n_sd * rho_d -
            (1.0 + 1.0j) * (n_sa * rho_a + n_sb * rho_b)))


def compute_response_function(ket_g, bra_g_star, operator_list, parameters, seed_rng, is_damped=True):
    """
    :param ket_g: the ground state ket |g> shape (n_hilbert,)
    :param bra_g_star: the ground state bra <g| shape (n_hilbert,). complex-conjugated and transposed to form a vector |g>
    :param operator_list: list of operators to invoke at t = 0, t1, t2, and t3
    :param parameters: Ehrenfest dynamics parameters
    :return: 3-dimensional response function given rho_g and the operator_list
    """
    
    print("==========================================================")
    print("               Calculating response function              ")
    print("==========================================================")
    
    traj_dt = parameters['dt']  # time step for Ehrenfest propagation
    traj_dT = parameters['dT']  # time interval for the waiting time t2
    traj_tmax = parameters['traj_tmax']  # max of t1 and t3
    t2_max = parameters['t2_max'] # max for t2
    traj_tot = parameters['traj_tot']  # number of trajectories to run
    n_hilbert = parameters['n_hilbert']
    hbar = parameters['hbar']

    ham_sys = parameters['hsys']
    ham_sys_bath = parameters['vsys']
    coupling = parameters['coupling']
    omega_c = parameters['omega_c']
    beta = parameters['beta']
    name_bath = parameters['bath']
    nosc = parameters['nosc']
    bath_coeff = parameters['aa']
    nbaths = parameters['nbaths']

    wav_ids = ['a', 'b', 'c', 'd']

    # interval for t2 should be a multiple of dt
    # assert(np.isclose(traj_dT % traj_dt, 0))

    # number of time steps, t1 and t3
    tsteps = int(traj_tmax/traj_dt) + 1
    # number of time steps for t2
    tsteps_t2 = int(t2_max/traj_dt) + 1
    # number of t2 intervals (for branching off into t3)
    n_intervals_t2 = int(t2_max/traj_dT) + 1

    # construct bath (coupling, frequencies, etc)
    coupling_j, omega_j, bath_aux = frenkel.construct_boson_bath_new(coupling, omega_c, name_bath, nosc,
                                                                     bath_coeff, nbaths, traj_dt)

    out_array = np.zeros((tsteps, n_intervals_t2, tsteps), dtype=complex)

    # seed random number generator
    np.random.seed(seed_rng) 

    # independent runs
    for k in range(traj_tot):

        # 1. forward propagation

        # initialize some arrays
        coeffs_t1 = np.empty((4,), dtype=float)  # order is N^2 of a, b, c, and d
        coeffs_t2 = np.empty((tsteps, 4, 4), dtype=float)
        coeffs_t3 = np.empty((tsteps, n_intervals_t2, 4, 4, 4), dtype=float)

        # initialize the intermediate outputs
        out_t3 = np.empty((tsteps, n_intervals_t2, 4, 4, 4, n_hilbert, tsteps), dtype=complex)

        # operator at t0, no baths yet (step 1)
        ket_0, bra_0_star = act(ket_g, bra_g_star, operator_list[0])

        wavefunctions = return_four_wavefunctions(ket_0, bra_0_star, initialize=True, beta=beta,
                                                  omega_j=omega_j)  # bath is initialized

        for wav_x_index, wav_x_id in enumerate(wav_ids):
            wav_x = wavefunctions[wav_x_id]
            coeffs_t1[wav_x_index] = wav_x[1]  # norm squared

            for t1_index in range(tsteps):
                
                # initialize
                ket_xx, bra_xx_star = act(wav_x[0], wav_x[0], operator_list[1])

                # split into four wavefunctions, continue baths
                wavs_xx = return_four_wavefunctions(ket_xx, bra_xx_star, initialize=False,
                                                    bath_q=wav_x[2], bath_p=wav_x[3])

                # record coefficients
                for wav_xx_index, wav_xx_id in enumerate(wav_ids):
                    wav_xx = wavs_xx[wav_xx_id]
                    coeffs_t2[t1_index, wav_x_index, wav_xx_index] = wav_xx[1]  # norm squared

                    t2_interval_index = 0
                    # propagate through t2
                    for t2_index in range(tsteps_t2):

                        # update the output
                        # out_t2[t1_index, wav_x_index, wav_xx_index, :, t2_index] = wav_xx[0]

                        # one layer deeper
                        if np.isclose((t2_index*traj_dt) % traj_dT, 0):

                            # time to branch off into t3
                            ket_xxx, bra_xxx_star = act(wav_xx[0], wav_xx[0], operator_list[2])

                            # split
                            wavs_xxx = return_four_wavefunctions(ket_xxx, bra_xxx_star, initialize=False,
                                                                 bath_q=wav_xx[2], bath_p=wav_xx[3])

                            # record coefficients
                            for wav_xxx_index, wav_xxx_id in enumerate(wav_ids):
                                wav_xxx = wavs_xxx[wav_xxx_id]
                                coeffs_t3[t1_index, t2_interval_index, wav_x_index, wav_xx_index, wav_xxx_index] = wav_xxx[1]

                                # propagate through t3
                                for t3_index in range(tsteps):

                                    # update the output
                                    out_t3[t1_index, t2_interval_index, wav_x_index, wav_xx_index, wav_xxx_index, :, t3_index] = wav_xxx[0]

                                    # time-propagate
                                    wav_xxx[0], wav_xxx[2], wav_xxx[3] = ehrenfest_update(ham_sys, ham_sys_bath, hbar, wav_xxx[0], wav_xxx[2],
                                                                                           wav_xxx[3], coupling_j, bath_aux, traj_dt)

                            t2_interval_index += 1

                        # time-propagate
                        wav_xx[0], wav_xx[2], wav_xx[3] = ehrenfest_update(ham_sys, ham_sys_bath, hbar, wav_xx[0], wav_xx[2],
                                                                           wav_xx[3], coupling_j, bath_aux, traj_dt)

                # time-propagate
                wav_x[0], wav_x[2], wav_x[3] = ehrenfest_update(ham_sys, ham_sys_bath, hbar, wav_x[0], wav_x[2], wav_x[3],
                                                                coupling_j, bath_aux, traj_dt)

        # 2. time to back-propagate
        # from wavefunctions to density matrices
        rhos_out = np.multiply(out_t3[:,:,:,:,:,:,np.newaxis,:], np.conj(out_t3[:,:,:,:,:,np.newaxis,:,:]))
        del out_t3
        
        rhos_out = aggregate(rhos_out[:,:,:,:,0,:,:,:],
                             rhos_out[:,:,:,:,1,:,:,:],
                             rhos_out[:,:,:,:,2,:,:,:],
                             rhos_out[:,:,:,:,3,:,:,:],
                             coeffs_t3[:,:,:,:,0,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t3[:,:,:,:,1,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t3[:,:,:,:,2,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t3[:,:,:,:,3,np.newaxis,np.newaxis,np.newaxis]
                             )
        del coeffs_t3
        
        rhos_out = aggregate(rhos_out[:,:,:,0,:,:,:],
                             rhos_out[:,:,:,1,:,:,:],
                             rhos_out[:,:,:,2,:,:,:],
                             rhos_out[:,:,:,3,:,:,:],
                             coeffs_t2[:,np.newaxis,:,0,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t2[:,np.newaxis,:,1,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t2[:,np.newaxis,:,2,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t2[:,np.newaxis,:,3,np.newaxis,np.newaxis,np.newaxis]
                             )
        del coeffs_t2
        
        rhos_out = aggregate(rhos_out[:,:,0,:,:,:],
                             rhos_out[:,:,1,:,:,:],
                             rhos_out[:,:,2,:,:,:],
                             rhos_out[:,:,3,:,:,:],
                             coeffs_t1[np.newaxis,np.newaxis,0,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t1[np.newaxis,np.newaxis,1,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t1[np.newaxis,np.newaxis,2,np.newaxis,np.newaxis,np.newaxis],
                             coeffs_t1[np.newaxis,np.newaxis,3,np.newaxis,np.newaxis,np.newaxis]
                             )
        del coeffs_t1
        
        # now rhos_out has shape (n_t1, n_t2, n_hilbert, n_hilbert, n_t3)
        # prepare output
        operator_t3 = operator_list[3]
        t_damp = 0.
        
        for t1_index in range(tsteps):
            for t2_index in range(n_intervals_t2):
                for t3_index in range(tsteps):
                    # apply the last operator
                    out_value = np.trace(np.dot(operator_t3, rhos_out[t1_index,t2_index,:,:,t3_index]))
                    # reverse order, t3, t2, t1
                    out_array[t3_index, t2_index, t1_index] += out_value

                    if is_damped and t1_index>=t_damp and t3_index>=t_damp:
                        out_array[t3_index, t2_index, t1_index] *= switch_to_zero(t3_index, t_damp+int(0.5*tsteps), tsteps)
                        out_array[t3_index, t2_index, t1_index] *= switch_to_zero(t1_index, t_damp+int(0.5*tsteps), tsteps)

    return out_array/traj_tot


def switch_to_zero(x, x0, x1):
    """
    Switching function to go from 1 at x0 to 0 at x1
    """
    # cubic spline with 0 derivative at both ends
    y = (x-x0)/(x1-x0)
    
    if x > x0:
        switch = 1 - 3*y**2 + 2*y**3
    else:
        switch = 1
    
    return switch


def single_time_cf(ket, bra_star, parameters):
    """
    Dynamics for the density matrix |ket><bra|
    :param ket: ket wavefunction
    :param bra_star: bra star wavefunction, i.e. (<bra_star|)*
    :param parameters: parameter dictionary
    :return: single time dynamics for the initial condition |ket><bra|
    """

    print("===========================================================================")
    print("               Calculating correlation function (single time)              ")
    print("===========================================================================")

    traj_dt = parameters['dt']  # time step for Ehrenfest propagation
    traj_tmax = parameters['traj_tmax']  # max time
    traj_tot = parameters['traj_tot']  # number of trajectories to run
    n_hilbert = parameters['n_hilbert']
    hbar = parameters['hbar']

    ham_sys = parameters['hsys']
    ham_sys_bath = parameters['vsys']
    coupling = parameters['coupling']
    omega_c = parameters['omega_c']
    beta = parameters['beta']
    name_bath = parameters['bath']
    nosc = parameters['nosc']
    bath_coeff = parameters['aa']
    nbaths = parameters['nbaths']

    wav_ids = ['a', 'b', 'c', 'd']

    # number of time steps
    tsteps = int(traj_tmax / traj_dt) + 1

    # construct bath (coupling, frequencies, etc)
    coupling_j, omega_j, bath_aux = frenkel.construct_boson_bath_new(coupling, omega_c, name_bath, nosc,
                                                                     bath_coeff, nbaths, traj_dt)

    out_array = np.zeros((n_hilbert, n_hilbert, tsteps), dtype=complex)

    # independent runs
    for k in range(traj_tot):

        # coefficients (squared norms for the starting wavefunctions)
        coeffs = np.empty((4,), dtype=float)

        # intermediate output
        sigma_out = np.zeros((4, n_hilbert, tsteps), dtype=complex)

        # returns lists with indices 0: wavefunction, 1: norm squared, 2: bath positions, 3: bath momenta
        wavefunctions = return_four_wavefunctions(ket, bra_star, initialize=True, beta=beta, omega_j=omega_j)

        # run the four wavefunctions
        for wav_x_index, wav_x_id in enumerate(wav_ids):

            wav_x = wavefunctions[wav_x_id]
            coeffs[wav_x_index] = wav_x[1]

            for t in range(tsteps):

                # record the value of the wavefunction and then propagate it
                sigma_out[wav_x_index, :, t] = wav_x[0]
                wav_x[0], wav_x[2], wav_x[3] = ehrenfest_update(ham_sys, ham_sys_bath, hbar, wav_x[0], wav_x[2], wav_x[3],
                                                                coupling_j, bath_aux, traj_dt)

        # obtain the density matrices
        rhos_out = np.multiply(sigma_out[:, :, np.newaxis, :], np.conj(sigma_out[:, np.newaxis, :, :]))
        rhos_out = aggregate(rhos_out[0], rhos_out[1], rhos_out[2], rhos_out[3],
                             coeffs[0], coeffs[1], coeffs[2], coeffs[3])

        out_array += rhos_out

    return out_array/traj_tot


def single_time_wf(ket, bra_star, parameters):
    """
    Dynamics for the density matrix |ket><bra|
    :param ket: ket wavefunction
    :param bra_star: bra star wavefunction, i.e. (<bra_star|)*
    :param parameters: parameter dictionary
    :return: single time dynamics for the initial condition |ket><bra|
    """

    print("===========================================================================")
    print("               Calculating correlation function (single time)              ")
    print("===========================================================================")

    traj_dt = parameters['dt']  # time step for Ehrenfest propagation
    traj_tmax = parameters['traj_tmax']  # max time
    traj_tot = parameters['traj_tot']  # number of trajectories to run
    n_hilbert = parameters['n_hilbert']
    hbar = parameters['hbar']

    ham_sys = parameters['hsys']
    ham_sys_bath = parameters['vsys']
    coupling = parameters['coupling']
    omega_c = parameters['omega_c']
    beta = parameters['beta']
    name_bath = parameters['bath']
    nosc = parameters['nosc']
    bath_coeff = parameters['aa']
    nbaths = parameters['nbaths']

    wav_ids = ['a', 'b', 'c', 'd']

    # number of time steps
    tsteps = int(traj_tmax / traj_dt) + 1

    # construct bath (coupling, frequencies, etc)
    coupling_j, omega_j, bath_aux = frenkel.construct_boson_bath_new(coupling, omega_c, name_bath, nosc,
                                                                     bath_coeff, nbaths, traj_dt)

    out_array = np.zeros((4, n_hilbert, tsteps), dtype=complex)

    # independent runs
    for k in range(traj_tot):

        # coefficients (squared norms for the starting wavefunctions)
        coeffs = np.empty((4,), dtype=float)

        # intermediate output
        sigma_out = np.zeros((4, n_hilbert, tsteps), dtype=complex)

        # returns lists with indices 0: wavefunction, 1: norm squared, 2: bath positions, 3: bath momenta
        wavefunctions = return_four_wavefunctions(ket, bra_star, initialize=True, beta=beta, omega_j=omega_j)

        # run the four wavefunctions
        for wav_x_index, wav_x_id in enumerate(wav_ids):

            wav_x = wavefunctions[wav_x_id]
            coeffs[wav_x_index] = wav_x[1]

            for t in range(tsteps):

                # record the value of the wavefunction and then propagate it
                sigma_out[wav_x_index, :, t] = wav_x[0]
                wav_x[0], wav_x[2], wav_x[3] = ehrenfest_update(ham_sys, ham_sys_bath, hbar, wav_x[0], wav_x[2], wav_x[3],
                                                                coupling_j, bath_aux, traj_dt)


        out_array += sigma_out

    return out_array/traj_tot


def absorption(ket, bra_star, dipole, parameters, e_min, e_max, de, t_damp=0):
    """
    Returns the absorption spectrum for a density matrix i.c. |ket><bra|
    :param ket: ket |g> shape (n_hilbert,)
    :param bra_star: bra <g| shape (n_hilbert,). complex-conjugated and transposed to form a vector |g>
    :param dipole: dipole moment for the system for the light-matter interaction
    :param parameters, emin, emax, de: the propagation parameters, and energy range for the resulting absorption spectrum
    """
    energies = np.arange(e_min, e_max, de)
    omegas = energies/parameters['hbar']

    dt = parameters['dt']
    times = np.arange(0., parameters['traj_tmax']+dt, dt)

    # apply operator
    mu_ket = np.dot(dipole, ket)
    # run dynamics
    rho_t_array = single_time_cf(mu_ket, bra_star, parameters)
    
    # Fourier transform
    C_omega = np.zeros(len(omegas))
    expi = np.exp(1j*np.outer(omegas, times))

    for t, time in enumerate(times):
        weight = dt - 0.5*dt*(t==0 or t==len(times)-1)
        Ct = np.trace(np.dot(dipole, rho_t_array[:,:,t]))
        C_omega += weight*(expi[:,t]*Ct).real

    return energies, C_omega


def two_dimensional(ket, bra_star, dipole, parameters, e1_min, e1_max, de1, e3_min, e3_max, de3, seed_rng, is_damped=True):
    
    print("==========================================================")
    print("               Calculating 2D Spectrum                    ")
    print("==========================================================")

    energy1s = np.arange(e1_min, e1_max, de1)
    energy3s = np.arange(e3_min, e3_max, de3)
    omega1s = energy1s/parameters['hbar']
    omega3s = energy3s/parameters['hbar']

    # compute the response functions in turn
    mu_p = np.tril(dipole)
    mu_m = np.triu(dipole)

    # six response functions
    phi_1 = compute_response_function(ket, bra_star, [[mu_m, 'R'], [mu_p, 'L'], [mu_p, 'R'], mu_m], parameters, seed_rng, is_damped=is_damped)
    phi_2 = compute_response_function(ket, bra_star, [[mu_m, 'R'], [mu_p, 'R'], [mu_p, 'L'], mu_m], parameters, seed_rng, is_damped=is_damped)
    phi_3 = compute_response_function(ket, bra_star, [[mu_m, 'R'], [mu_p, 'L'], [mu_p, 'L'], mu_m], parameters, seed_rng, is_damped=is_damped)
    phi_4 = compute_response_function(ket, bra_star, [[mu_p, 'L'], [mu_m, 'R'], [mu_p, 'R'], mu_m], parameters, seed_rng, is_damped=is_damped)
    phi_5 = compute_response_function(ket, bra_star, [[mu_p, 'L'], [mu_m, 'L'], [mu_p, 'L'], mu_m], parameters, seed_rng, is_damped=is_damped)
    phi_6 = compute_response_function(ket, bra_star, [[mu_p, 'L'], [mu_m, 'R'], [mu_p, 'L'], mu_m], parameters, seed_rng, is_damped=is_damped)

    # rephasing and non-rephasing components
    # R_rp = R2 + R3 - np.conj(R1)
    # R_nr = R1 + R4 - np.conj(R2)

    R_rp = phi_1 + phi_2 - phi_3
    R_nr = phi_4 + phi_5 - phi_6
    
    # Fourier transform
    print("===========================================================")
    print("               Performing 2D Fourier Transform             ")
    print("===========================================================")

    dt = parameters['dt']
    # time axis for t1 and t3
    times = np.arange(0., parameters['traj_tmax']+dt, dt)
    expi1 = np.exp(1j*np.outer(omega1s, times))
    expi1[:,0] *= 0.5*dt
    expi1[:,1:] *= dt
    expi3 = np.exp(1j*np.outer(omega3s, times))
    expi3[:,0] *= 0.5*dt
    expi3[:,1:] *= dt
    spectrum = np.einsum('ws,xu,uts->xtw', expi1, expi3, R_nr).real
    spectrum += np.einsum('ws,xu,uts->xtw', expi1.conj(), expi3, R_rp).real

    print("Done")
    
    # dimensions: omega_3, t2, omega_1
    return spectrum
