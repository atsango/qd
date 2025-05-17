#include "quantum_dynamics.hpp"
#include <cmath>
#include <stdexcept>

namespace qd {

    // Bath implementation
    Bath::Bath(int nbaths, int nosc, BathType type, const Vector& coupling, 
               const Vector& omega_c, double coupling_coeff)
        : type_(type), nbaths_(nbaths), nosc_(nosc) {
        q_.resize(nbaths_, nosc_);
        p_.resize(nbaths_, nosc_);
        coupling_.resize(nbaths_, nosc_);
        omega_.resize(nbaths_, nosc_);
        bath_aux_.resize(4, nbaths_ * nosc_);

        // Initialize coupling and omega based on bath type
        for (int i = 0; i < nbaths_; ++i) {
            double lamb = coupling[i];
            double wc = omega_c[i];

            if (type_ == BathType::DEBYE) {
                for (int j = 0; j < nosc_; ++j) {
                    double j_val = j + 0.5;
                    omega_(i, j) = wc * std::tan(0.5 * M_PI * j_val / nosc_);
                    coupling_(i, j) = std::sqrt(2 * lamb / nosc_) * omega_(i, j);
                }
            } else if (type_ == BathType::OHMIC) {
                for (int j = 0; j < nosc_; ++j) {
                    double j_val = j + 0.5;
                    omega_(i, j) = -wc * std::log(j_val / nosc_);
                    coupling_(i, j) = std::sqrt(lamb * wc / nosc_) * omega_(i, j);
                }
            } else if (type_ == BathType::HOLSTEIN) {
                for (int j = 0; j < nosc_; ++j) {
                    omega_(i, j) = wc;
                    coupling_(i, j) = std::sqrt(2 * wc) * lamb;
                }
            } else if (type_ == BathType::NONE) {
                for (int j = 0; j < nosc_; ++j) {
                    omega_(i, j) = 1.0;
                    coupling_(i, j) = 0.0;
                }
            }
        }

        // Renormalize couplings
        coupling_ *= coupling_coeff;
    }

    void Bath::initialize(const Parameters& params) {
        // Calculate auxiliary quantities
        Matrix bath_cos = std::cos(omega_ * 0.5 * params.dt);
        Matrix bath_sin_owj = std::sin(omega_ * 0.5 * params.dt) / omega_;
        Matrix bath_sin_twj = std::sin(omega_ * 0.5 * params.dt) * omega_;
        Matrix cwj2 = coupling_ / (omega_ * omega_);

        // Pack auxiliary components
        bath_aux_.block(0, 0, 1, nbaths_ * nosc_) = cwj2;
        bath_aux_.block(1, 0, 1, nbaths_ * nosc_) = bath_cos;
        bath_aux_.block(2, 0, 1, nbaths_ * nosc_) = bath_sin_owj;
        bath_aux_.block(3, 0, 1, nbaths_ * nosc_) = bath_sin_twj;

        // Initialize bath variables
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 1.0);

        // Calculate standard deviations
        Matrix gg = 2 * std::tanh(0.5 * params.beta * omega_);
        Matrix q_stddev = std::sqrt(1.0 / (gg * omega_));
        Matrix p_stddev = std::sqrt(omega_ / gg);

        // Initialize positions and momenta
        for (int i = 0; i < nbaths_; ++i) {
            for (int j = 0; j < nosc_; ++j) {
                q_(i, j) = d(gen) * q_stddev(i, j);
                p_(i, j) = d(gen) * p_stddev(i, j);
            }
        }
    }

    void Bath::update(const ComplexVector& wavefunction, const std::vector<ComplexMatrix>& vs) {
        ComplexVector system_force = calculate_ehrenfest_system_force(wavefunction, vs);
        std::tie(q_, p_) = update_harmonic_bath(q_, p_, system_force, bath_aux_);
    }

    // System implementation
    System::System(const Parameters& params)
        : params_(params) {
        hamiltonian_.resize(params.n_hilbert, params.n_hilbert);
    }

    void System::propagate(const Bath& bath, const ComplexVector& wavefunction) {
        // Update Hamiltonian with bath coupling
        ComplexMatrix vbt = calculate_vbt_linear(bath.get_coupling(), bath.get_q());
        hamiltonian_ = update_subsystem_hamiltonian(vbt, vs_);

        // Update wavefunction
        wavefunction_ = update_wavefunction(hamiltonian_, params_.hbar, 
                                          params_.dt, wavefunction);
    }

    ComplexMatrix System::get_density_matrix() const {
        return wavefunction_ * wavefunction_.adjoint();
    }

    // WavefunctionManager implementation
    WavefunctionManager::WavefunctionManager(const Parameters& params)
        : params_(params) {}

    std::vector<Wavefunction> WavefunctionManager::get_wavefunctions(
        const ComplexVector& ket, const ComplexVector& bra_star) {
        if (ket.size() != bra_star.size()) {
            throw std::invalid_argument("Ket and bra must be in the same Hilbert space");
        }

        double eps = 1e-19;
        std::vector<Wavefunction> wavefunctions(4);

        // Calculate norms
        double norm_a = (ket.cwiseAbs2().sum() + eps);
        double norm_b = (bra_star.cwiseAbs2().sum() + eps);

        // Create normalized wavefunctions
        wavefunctions[0] = {ket / std::sqrt(norm_a), norm_a};
        wavefunctions[1] = {bra_star / std::sqrt(norm_b), norm_b};

        // Create combined wavefunctions
        ComplexVector wav_plus = ket + bra_star;
        double norm_c = (wav_plus.cwiseAbs2().sum() + eps);
        wavefunctions[2] = {wav_plus / std::sqrt(norm_c), norm_c};

        ComplexVector wav_minus = ket + Complex(0, 1) * bra_star;
        double norm_d = (wav_minus.cwiseAbs2().sum() + eps);
        wavefunctions[3] = {wav_minus / std::sqrt(norm_d), norm_d};

        return wavefunctions;
    }

    // ResponseFunction implementation
    ResponseFunction::ResponseFunction(const Parameters& params, const Bath& bath)
        : params_(params), bath_(bath) {}

    ComplexMatrix ResponseFunction::compute(const ComplexVector& ket_g, 
                                           const ComplexVector& bra_g_star,
                                           const std::vector<ComplexMatrix>& operators) {
        // Implementation of response function calculation
        // This would involve propagating the wavefunctions and calculating the response
        return ComplexMatrix::Zero(params_.n_hilbert, params_.n_hilbert);
    }

    // Utility functions
    std::vector<Wavefunction> return_four_wavefunctions(const ComplexVector& ket, 
                                                       const ComplexVector& bra_star,
                                                       const Parameters& params) {
        WavefunctionManager manager(params);
        return manager.get_wavefunctions(ket, bra_star);
    }

    ComplexMatrix calculate_ehrenfest_system_force(const ComplexVector& wavefunction, 
                                                  const std::vector<ComplexMatrix>& vs) {
        ComplexMatrix force = ComplexMatrix::Zero(vs[0].rows(), 1);
        for (size_t i = 0; i < vs.size(); ++i) {
            force += wavefunction.adjoint() * vs[i] * wavefunction;
        }
        return force.real();
    }

    std::pair<Matrix, Matrix> update_harmonic_bath(const Matrix& q_bath, const Matrix& p_bath,
                                                  const ComplexVector& system_force,
                                                  const Matrix& bath_aux) {
        // Calculate weighted system force
        Matrix weighted_force = bath_aux.block(0, 0, 1, q_bath.size()) * system_force;
        Matrix q_shifted = q_bath + weighted_force;

        // Update positions
        Matrix q_new = q_shifted * bath_aux.block(1, 0, 1, q_bath.size()) + 
                      p_bath * bath_aux.block(2, 0, 1, q_bath.size()) - weighted_force;

        // Update momenta
        Matrix p_new = p_bath * bath_aux.block(1, 0, 1, q_bath.size()) - 
                      q_shifted * bath_aux.block(3, 0, 1, q_bath.size());

        return {q_new, p_new};
    }

    ComplexVector update_wavefunction(const ComplexMatrix& hamiltonian, double hbar,
                                     double timestep, const ComplexVector& wavefunction) {
        // Diagonalize Hamiltonian
        Eigen::EigenSolver<ComplexMatrix> es(hamiltonian);
        ComplexMatrix eigenvectors = es.eigenvectors();
        ComplexVector eigenvalues = es.eigenvalues();

        // Construct propagator
        ComplexMatrix eig_propagator = ComplexMatrix::Zero(eigenvalues.size(), eigenvalues.size());
        for (int i = 0; i < eigenvalues.size(); ++i) {
            eig_propagator(i, i) = std::exp(-Complex(0, 1) / hbar * eigenvalues(i) * timestep);
        }

        // Apply propagator
        ComplexMatrix propagator = eigenvectors * eig_propagator * eigenvectors.inverse();
        return propagator * wavefunction;
    }

}
