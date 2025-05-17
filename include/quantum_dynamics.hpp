#pragma once

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <random>

namespace qd {
    using Complex = std::complex<double>;
    using Matrix = Eigen::MatrixXd;
    using ComplexMatrix = Eigen::MatrixXcd;
    using Vector = Eigen::VectorXd;
    using ComplexVector = Eigen::VectorXcd;

    // Bath types
    enum class BathType {
        DEBYE,
        OHMIC,
        HOLSTEIN,
        NONE
    };

    // Wavefunction structure
    struct Wavefunction {
        ComplexVector state;
        double norm_squared;
        Matrix bath_q;
        Matrix bath_p;
    };

    // Simulation parameters
    struct Parameters {
        double dt;          // Time step
        double dT;          // Waiting time interval
        double traj_tmax;   // Maximum trajectory time
        double t2_max;      // Maximum waiting time
        int traj_tot;       // Number of trajectories
        int n_hilbert;      // Hilbert space dimension
        double hbar;        // Reduced Planck constant
        double beta;        // Inverse temperature
    };

    // Bath class
    class Bath {
    public:
        Bath(int nbaths, int nosc, BathType type, const Vector& coupling, 
             const Vector& omega_c, double coupling_coeff);

        void initialize(const Parameters& params);
        void update(const ComplexVector& wavefunction, const std::vector<ComplexMatrix>& vs);

        Matrix get_q() const { return q_; }
        Matrix get_p() const { return p_; }
        Matrix get_coupling() const { return coupling_; }
        Matrix get_omega() const { return omega_; }

    private:
        Matrix q_, p_;           // Bath coordinates
        Matrix coupling_;        // Coupling strengths
        Matrix omega_;           // Bath frequencies
        Matrix bath_aux_;        // Auxiliary bath evolution matrices
        BathType type_;
        int nbaths_, nosc_;
    };

    // System class
    class System {
    public:
        System(const Parameters& params);

        void propagate(const Bath& bath, const ComplexVector& wavefunction);
        ComplexMatrix get_density_matrix() const;

    private:
        ComplexMatrix hamiltonian_;
        Parameters params_;
    };

    // Wavefunction manager
    class WavefunctionManager {
    public:
        WavefunctionManager(const Parameters& params);

        std::vector<Wavefunction> get_wavefunctions(const ComplexVector& ket, 
                                                    const ComplexVector& bra_star);

    private:
        Parameters params_;
    };

    // Response function calculator
    class ResponseFunction {
    public:
        ResponseFunction(const Parameters& params, const Bath& bath);

        ComplexMatrix compute(const ComplexVector& ket_g, const ComplexVector& bra_g_star,
                             const std::vector<ComplexMatrix>& operators);

    private:
        Parameters params_;
        Bath bath_;
    };

    // Utility functions
    std::vector<Wavefunction> return_four_wavefunctions(const ComplexVector& ket, 
                                                       const ComplexVector& bra_star,
                                                       const Parameters& params);

    ComplexMatrix calculate_ehrenfest_system_force(const ComplexVector& wavefunction, 
                                                  const std::vector<ComplexMatrix>& vs);

    std::pair<Matrix, Matrix> update_harmonic_bath(const Matrix& q_bath, const Matrix& p_bath,
                                                  const ComplexVector& system_force,
                                                  const Matrix& bath_aux);

    ComplexVector update_wavefunction(const ComplexMatrix& hamiltonian, double hbar,
                                     double timestep, const ComplexVector& wavefunction);
}
