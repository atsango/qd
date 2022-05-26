import numba as nb
import numpy as np


# from harmonic_bath.py:
def construct_boson_bath_new(coupling, omega_c, name, nosc,
                             coupling_coeff, nbaths, timestep):
    """Discretizes a spectral density into a group of harmonic oscillators.

    Current options: debye, ohmic, holstein, none

    Input:
    1. coupling -- [nbaths]
                -- quantity proportional to the reorganization energy
    2. omega_c -- [nbaths]
               -- characteristic frequency of the bath
    3. name -- str -- bath type: debye, ohmic, holstein, or none
    4. nosc -- int -- number of oscillators per bath
    5. coupling_coeff -- plus or minus 1
    6. nbaths -- number of independent baths

    Output:
    1. coupling_j [nbaths, nosc]
    2. omega_j [nbaths, nosc]
    """
    # TODO:
    # 1. remove restriction of making all baths the same when nbaths > 1
    # 2. Add option to discretize an arbitrary spectral density

    # create arrays to store omega_j, coupling_j
    coupling_j = np.empty([nbaths, nosc], dtype=float)
    omega_j = np.empty_like(coupling_j)

    for i in range(nbaths):

        # choices of bath type
        if name.lower() == 'debye':
            lamb = coupling[i]
            j = np.arange(0.5, nosc + 0.5, 1.0)
            omega_j[i, :] = omega_c[i] * np.tan(0.5 * np.pi * j / nosc)
            coupling_j[i, :] = np.sqrt(2 * lamb / nosc) * omega_j[i]
        elif name.lower() == 'ohmic':
            zeta = coupling[i]
            j = np.arange(0.5, nosc + 0.5, 1.0)
            omega_j[i, :] = - omega_c[i] * np.log(j / nosc)
            coupling_j[i, :] = np.sqrt(zeta * omega_c[i] / nosc) * omega_j[i]
        elif name.lower() == 'holstein':
            gamma = coupling[i]
            omega_j[i, :] = omega_c[i] * np.ones([1], dtype=float)
            coupling_j[i, :] = np.sqrt(2 * omega_c[i]) * gamma * np.ones([1], dtype=float)
        elif name.lower() == 'none':
            omega_j[i, :] = np.ones([1], dtype=float)
            coupling_j[i, :] = np.zeros([1], dtype=float)
        else:
            raise ValueError('Unknown bath type: {:s}'.format(name))

    # Renormalize the couplings by aa
    coupling_j *= coupling_coeff

    #
    # build auxiliary arrays for bath evolution
    #

    # Define quantities for bath evolution
    bath_cos = np.cos(omega_j * 0.5 * timestep)
    bath_sin_owj = np.sin(omega_j * 0.5 * timestep) / omega_j
    bath_sin_twj = np.sin(omega_j * 0.5 * timestep) * omega_j
    cwj2 = coupling_j / (omega_j * omega_j)

    # pack the auxiliary components
    bath_aux = np.stack((cwj2, bath_cos, bath_sin_owj, bath_sin_twj))

    return coupling_j, omega_j, bath_aux


def harmonic_bath_initializer(beta, omega_j):
    """Initialize bath variables according to thermal distribution.

    Input:
    1. beta --- scalar
    2. omega_j --- [nbaths, nosc]

    Output:
    1. positions --- array [nbaths, nosc]
    2. momenta --- array [nbaths, nosc]
    """
    # TODO: add modification for beta --> infty
    # TODO: add option for a Boltzmann initializer

    # useful quantities
    gg = 2 * np.tanh(0.5 * beta * omega_j)
    q_stddev = np.sqrt(1.0 / (gg * omega_j))
    p_stddev = np.sqrt(omega_j / gg)

    # get the number of baths
    nbaths = omega_j.shape[0]
    nosc = omega_j.shape[1]

    # Initialize all baths
    positions = q_stddev * np.random.randn(nbaths, nosc)
    momenta = p_stddev * np.random.randn(nbaths, nosc)

    return positions, momenta


@nb.jit(nopython=True)
def update_harmonic_bath(q_bath, p_bath, system_force, bath_aux):
    """Update positions & momenta over delta t / 2

    H = 1/2 sum_k [p_k^2 + w_k^2 q_k^2 + c_k * system_force * q_k]

    Input:
    1. q_bath -- [nbaths, nmodes]
    2. p_bath -- [nbaths, nmodes]
    3. system_force -- [nbaths]
    4. bath_aux -- [4, ...]

    bath_aux[0] = coupling_j / omega_j**2
    bath_aux[1] = cos(omega_j * delta t / 2)
    bath_aux[2] = sin(omega_j * delta t / 2) / (omega_j * delta t)
    bath_aux[3] = sin(omega_j * delta t / 2) * (omega_j * delta t)
    """
    # use shifted coordinates
    # for a constant added force on the position
    weighted_system_force = bath_aux[0] * system_force
    q_shifted = q_bath + weighted_system_force
    q_new = q_shifted * bath_aux[1] + p_bath * bath_aux[2] - weighted_system_force
    p_new = p_bath * bath_aux[1] - q_shifted * bath_aux[3]

    return q_new, p_new

# from ehrenfest_evolution.py:

@nb.jit(nopython=True)
def calculate_ehrenfest_system_force(wavefunction, vs):
    """Calculate the Ehrefest system force.

        H_{sb} = sum_{j} vs_{j}
        F_s[j] = <psi(t)| vs[j, :, :] |psi(t)>

    Input:
    1. wavefunction -- [nhilbert]
    2. vs -- [nsb, nhilbert, nhilbert]

    Output;
    1. fs -- [nsb]
    """
    force = np.empty_like(vs[:, 0, 0])
    nsb = len(force)
    for i in range(nsb):
        force[i] = np.dot(np.conj(wavefunction),
                          np.dot(vs[i, :, :], wavefunction))

    # reshape to avoid problems with broadcasting later
    force = force.reshape(nsb, 1)
    return force.real

# from subsystem propagator.py:

@nb.jit(nopython=True)
def update_wavefunction(hamiltonian, hbar, timestep, wavefunction):
    """Construct the subsystem's forward propagator subsystem over dt.

        propagator = e^{-i H dt}
        H = hs + sum_{i} vbt[i] * vs[i, :, :]

    Input:
    1. ham_s -- [nhilbert, nhilbert]
    2. vbt -- [nbaths, nhilbert, nhilbert]
    3. vs -- [nbaths, nhilbert, nhilbert]

    Output:
    1. propagator -- [nhilbert, nhilbert]
    """
    # diagonalize
    eigenenergies, unitary = np.linalg.eigh(hamiltonian)
    unitaryt = np.transpose(unitary)
    eig_propagator = np.diag(np.exp(-1j/hbar * eigenenergies * timestep))

    # propagator
    propagator = np.dot(unitary, np.dot(eig_propagator, unitaryt))

    # update wavefunction over a full timestep
    return np.dot(propagator, wavefunction)


@nb.jit(nopython=True)
def update_subsystem_hamiltonian(hs, vbt, vs):
    """Update the subsystem Hamiltonian with a static bath.

        H = hs + sum_{i} vbt[i] * vs[i, :, :]

    Input:
    1. hs -- [nhilbert, nhilbert]
    2. vbt -- [nbaths, nhilbert, nhilbert]
    3. vs -- [nbaths, nhilbert, nhilbert]

    Output:
    1. new_hs -- [nhilbert, nhilbert]
    """
    # Set up analytical evolution over the subsystem
    new_hs = np.copy(hs)
    for i in range(len(vbt)):
        new_hs += vbt[i] * vs[i, :, :]

    return new_hs


# from system_bath_coupling.py:
@nb.jit(nopython=True)
def calculate_vbt_linear(coupling_j, q_bath):
    """Calculate the bath part of the H_{sb}.

        H_{sb} = sum_{j} vs[j, :, :] * vbt[j]
        vbt[j] = sum_{k} c[j, k] * q[j, k]

    Input:
    1. coupling_j -- [nbaths, nosc]
    2. q_bath -- [nbaths, nosc]

    Output:
    1. vbt -- [nbaths]
    """
    return np.sum(coupling_j * q_bath, axis=-1)
