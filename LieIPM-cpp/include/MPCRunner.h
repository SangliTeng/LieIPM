#pragma once

#include <memory>
#include <stdexcept>
#include "LieIPM.h"
#include "LieIPMCasadiUtils.h"
#include "LinearSolverWrapper.h"
#include "LieIPMUtils.h"

// Enumeration for linear solver selection
enum class LinearSolverType {
    EIGEN_LU,
    MUMPS
};

// Simplified output structure
struct MPCOutput {
    LieIPMData result;
    int status;
};

// MPCRunner class - simplified interface
template<typename Scalar = casadi_real>
class MPCRunner {
public:
    // Constructor takes parameters and utils from outside
    MPCRunner(const LieIPMParam& param, 
              LieIPMCasadiUtils& utils, 
              LinearSolverType solver_type = LinearSolverType::EIGEN_LU)
        : param_(param), utils_(utils), solver_type_(solver_type) {
        
        // Create appropriate linear solver based on enum
        createLinearSolver();
        
        // Create IPM solver based on solver type
        createIPMSolver();
    }

    virtual ~MPCRunner() = default;

    // Main solve interface - takes solver state as input parameters
    MPCOutput& solve(const Vec& x, const Vec& y, const Vec& z, const Vec& s, Scalar mu_scalar) {
        try {
            // Set solver initial guess internally
            Vec mu_vec = Vec::Constant(1, mu_scalar);
            setAllInputs(0, x);      // x
            setAllInputs(1, y);      // y
            setAllInputs(2, z);      // z
            setAllInputs(3, s);      // s
            setAllInputs(4, mu_vec); // mu
            
            // std::cout << "[MPCRunner] Solving with mu=" << mu_scalar 
            //           << ", x.size=" << x.size() 
            //           << ", solver=" << (solver_type_ == LinearSolverType::MUMPS ? "MUMPS" : "EIGEN_LU") << std::endl;
            
            // Solve using IPM solver
            output_.status = output_.result.flag;
            output_init_ = output_; // store the previous result first
            output_.result = solveIPM(x, y, z, s, mu_scalar);
            if(output_.status != -3) {
                output_init_ = output_; // if success, update...
            }
            
            
        } catch (const std::exception& e) {
            output_.status = -1; // Error flag
            std::cerr << "[MPCRunner] Error: " << e.what() << std::endl;
        }
        
        return output_;
    }

    // New run interface - uses stored result as warm start, x0 as initial state
    MPCOutput& run(const Vec& x0_input) {
        try {
            // Set x0 as MPC initial state
            set_mpc_initial_state(x0_input);
            
            // Use previous result as warm start directly
            Vec x_init = output_init_.result.xu;
            Vec y_init = output_init_.result.lam;
            Vec z_init = output_init_.result.mu;
            Vec s_init = output_init_.result.s;
            casadi_real mu_init = output_init_.result.mu_log.empty() ? 1e-2 : output_init_.result.mu_log.back();
            
            // std::cout << "[MPCRunner] Running with x0.size=" << x0_input.size() 
            //           << ", mu=" << mu_init
            //           << ", solver=" << (solver_type_ == LinearSolverType::MUMPS ? "MUMPS" : "EIGEN_LU") << std::endl;
            
            // Call the existing solve function
            return solve(x_init, y_init, z_init, s_init, mu_init);
            
        } catch (const std::exception& e) {
            output_.status = -1;
            std::cerr << "[MPCRunner] Error in run: " << e.what() << std::endl;
            return output_;
        }
    }

    // Get the latest output
    const MPCOutput& getOutput() const { return output_; }

    // Get solver type
    LinearSolverType getSolverType() const { return solver_type_; }

    // Get parameter reference for modification
    LieIPMParam& getParams() { return param_; }
    const LieIPMParam& getParams() const { return param_; }
    
    // Helper function to solve with appropriate IPM
    LieIPMData solveIPM(const Vec& x, const Vec& y, const Vec& z, const Vec& s, Scalar mu) {
        switch (solver_type_) {
            case LinearSolverType::EIGEN_LU: {
                auto* ipm = static_cast<LieIPM<LieIPMCasadiUtils, EigenLUSolverWrapper<Scalar, long long>>*>(imp_solver_.get());
                return ipm->solve(x, y, z, s, mu);
            }
            case LinearSolverType::MUMPS: {
                auto* ipm = static_cast<LieIPM<LieIPMCasadiUtils, MUMPSSolverWrapper<Scalar, long long>>*>(imp_solver_.get());
                return ipm->solve(x, y, z, s, mu);
            }
            default:
                throw std::runtime_error("Unknown linear solver type");
        }
    }

    // Unified function to set all inputs to all CasADi wrappers
    void setAllInputs(int idx, const Vec& data) {
        std::vector<casadi_real> data_vec(data.data(), data.data() + data.size());
        utils_.f.set_input(idx, data_vec);
        utils_.df.set_input(idx, data_vec);
        
        // Set KKT matrix based on solver type
        if (solver_type_ == LinearSolverType::MUMPS) {
            utils_.Asym_kkt.set_input(idx, data_vec);  // MUMPS uses symmetric KKT
        } else {
            utils_.A_kkt.set_input(idx, data_vec);     // EigenLU uses asymmetric KKT
        }
        
        utils_.b_kkt.set_input(idx, data_vec);
        utils_.hg.set_input(idx, data_vec);
    }

    // Set solver initial guess (x, y, z, s, mu)
    void set_solver_initial_guess(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu_vec) {
        setAllInputs(0, x);      // x
        setAllInputs(1, y);      // y
        setAllInputs(2, z);      // z
        setAllInputs(3, s);      // s
        setAllInputs(4, mu_vec); // mu
    }

    // Set MPC initial state
    void set_mpc_initial_state(const Vec& x0) {
        setAllInputs(5, x0);     // x0
    }

    // Set control parameters (weights and time step)
    void set_mpc_control_param(const Vec& ssw, const Vec& tsw, const Vec& iw, const Vec& dt) {
        setAllInputs(6, ssw);    // stage_state_weights
        setAllInputs(7, tsw);    // terminal_state_weights
        setAllInputs(8, iw);     // input_weights
        setAllInputs(11, dt);    // dt
    }

    // Set MPC reference trajectory
    void set_mpc_reference(const Vec& xd, const Vec& ud) {
        setAllInputs(9, xd);     // xd (desired state trajectory)
        setAllInputs(10, ud);    // ud (desired input trajectory)
    }

private:
    // Member variables
    LieIPMParam param_;
    LieIPMCasadiUtils& utils_;
    LinearSolverType solver_type_;
    MPCOutput output_; // Internal output storage

    MPCOutput output_init_;
    
    // Solver instances
    std::unique_ptr<LinearSolverWrapper<Scalar, long long>> linear_solver_;
    
    // Custom deleter for type-erased pointer
    struct IPMDeleter {
        LinearSolverType solver_type;
        
        void operator()(void* ptr) const {
            if (!ptr) return;
            
            switch (solver_type) {
                case LinearSolverType::EIGEN_LU:
                    delete static_cast<LieIPM<LieIPMCasadiUtils, EigenLUSolverWrapper<Scalar, long long>>*>(ptr);
                    break;
                case LinearSolverType::MUMPS:
                    delete static_cast<LieIPM<LieIPMCasadiUtils, MUMPSSolverWrapper<Scalar, long long>>*>(ptr);
                    break;
            }
        }
    };
    
    std::unique_ptr<void, IPMDeleter> imp_solver_; // Type-erased pointer with custom deleter

    void createLinearSolver() {
        switch (solver_type_) {
            case LinearSolverType::EIGEN_LU:
                linear_solver_ = std::make_unique<EigenLUSolverWrapper<Scalar, long long>>();
                break;
            case LinearSolverType::MUMPS:
                linear_solver_ = std::make_unique<MUMPSSolverWrapper<Scalar, long long>>();
                break;
            default:
                throw std::runtime_error("Unknown linear solver type");
        }
    }

    void createIPMSolver() {
        switch (solver_type_) {
            case LinearSolverType::EIGEN_LU: {
                auto* eigen_solver = static_cast<EigenLUSolverWrapper<Scalar, long long>*>(linear_solver_.get());
                auto ipm = std::make_unique<LieIPM<LieIPMCasadiUtils, EigenLUSolverWrapper<Scalar, long long>>>(
                    param_, utils_, *eigen_solver);
                imp_solver_ = std::unique_ptr<void, IPMDeleter>(
                    reinterpret_cast<void*>(ipm.release()), 
                    IPMDeleter{solver_type_}
                );
                break;
            }
            case LinearSolverType::MUMPS: {
                auto* mumps_solver = static_cast<MUMPSSolverWrapper<Scalar, long long>*>(linear_solver_.get());
                auto ipm = std::make_unique<LieIPM<LieIPMCasadiUtils, MUMPSSolverWrapper<Scalar, long long>>>(
                    param_, utils_, *mumps_solver);
                imp_solver_ = std::unique_ptr<void, IPMDeleter>(
                    reinterpret_cast<void*>(ipm.release()), 
                    IPMDeleter{solver_type_}
                );
                break;
            }
            default:
                throw std::runtime_error("Unknown linear solver type");
        }
    }
};


