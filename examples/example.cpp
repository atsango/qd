#include "quantum_dynamics.hpp"
#include <iostream>

int main() {
    // Create simulation parameters
    qd::Parameters params = {
        0.01,      // dt
        0.1,       // dT
        10.0,      // traj_tmax
        5.0,       // t2_max
        1000,      // traj_tot
        10,        // n_hilbert
        1.0,       // hbar
        1.0        // beta
    };

    // Create bath
    qd::Bath bath(1, 10, qd::BathType::DEBYE, 
                 qd::Vector::Ones(1), qd::Vector::Ones(1), 1.0);
    bath.initialize(params);

    // Create system
    qd::System system(params);

    // Create initial wavefunction
    qd::ComplexVector ket = qd::ComplexVector::Random(params.n_hilbert);
    qd::ComplexVector bra_star = qd::ComplexVector::Random(params.n_hilbert);

    // Calculate wavefunctions
    qd::WavefunctionManager manager(params);
    auto wavefunctions = manager.get_wavefunctions(ket, bra_star);

    // Calculate response function
    qd::ResponseFunction response(params, bath);
    std::vector<qd::ComplexMatrix> operators(1, qd::ComplexMatrix::Random(params.n_hilbert, params.n_hilbert));
    auto response_matrix = response.compute(ket, bra_star, operators);

    std::cout << "Response function calculated successfully!\n";
    return 0;
}
