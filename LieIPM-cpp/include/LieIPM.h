#pragma once
// Adapted interior-point method header to work with LieIPMCasadiUtils.
// The Utils type is expected to expose the Fun_* API (Fun_f, Fun_df, Fun_A_kkt, Fun_b_kkt, Fun_hg, Fun_retract).

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <LinearSolverWrapper.h>
#include <LieIPMUtils.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::microseconds;

#include <matio.h>

// using Vec = Eigen::VectorXd;
using Vec   = std::conditional_t<std::is_same_v<casadi_real, double>, Eigen::VectorXd, Eigen::VectorXf>;
// using Mat = Eigen::SparseMatrix<casadi_real, Eigen::ColMajor, long long>;
using SpMat = Eigen::SparseMatrix<casadi_real, Eigen::ColMajor, long long>;

template <typename Utils, typename Solver>
// template <typename Utils>
class LieIPM {
public:
    LieIPM(const LieIPMParam& param, Utils &utils,
           Solver &solver)
        : param_(param), utils_(std::move(utils)), lin_solver_(solver) {
        
        // TSL TODO: 
        // 1. Maybe we should remove the template for utils. 
        // 2. Use output buffer to provide the 
        
        // Must allow in-place modifucation of *A_kkt_ptr_ for inertia correction.
        A_kkt_ptr_ = reinterpret_cast<Eigen::Map<SpMat>*>(A_kkt_buffer_);
        // A_kkt_ptr_ = reinterpret_cast<Eigen::Map<const SpMat>*>(A_kkt_buffer_);
        df_ptr_ = reinterpret_cast<Eigen::Map<const Vec>*>(df_buffer_);
        b_kkt_ptr_ = reinterpret_cast<Eigen::Map<const Vec>*>(b_kkt_buffer_);
        h_tiral_ptr_ = reinterpret_cast<Eigen::Map<const Vec>*>(h_buffer_);
        g_trial_ptr_ = reinterpret_cast<Eigen::Map<const Vec>*>(g_buffer_);
        
        // std::cout << lin_solver_.getSolverType() << std::endl;
        if (lin_solver_.getSolverType() == SolverType::MUMPS) {
            utils_.Fun_Asym_kkt_get_buffer(A_kkt_ptr_);
            lin_solver_.analyzePattern(A_kkt_ptr_);
        } else {
            utils_.Fun_A_kkt_get_buffer(A_kkt_ptr_);
            lin_solver_.analyzePattern(A_kkt_ptr_);
        }        
    }

    LieIPMData solve(Vec x,
                     Vec y,
                     Vec z,
                     Vec s,
                     casadi_real          mu) {
        LieIPMData data;
        const auto t0 = now();

        int flag = -1;
        int iter = 0;
        
        std::vector<casadi_real> cost_log, mu_log, barrier_kkt_violation_log, overall_kkt_violation_log;
        std::vector<int>    line_search_ct_log, line_search_flag_log, nnz_log;
        std::vector<casadi_real> t_lin_sys_log, t_forloop_log;
        casadi_real term1_0_unscaled = 0, term2_0_unscaled = 0, term3_0_unscaled = 0;
        
        Vec x_new = x, y_new = y, z_new = z, s_new = s;

        // auto t_forloop_start = high_resolution_clock::now();

        for (iter = 1; iter <= param_.iter_max; ++iter) {
            // Timing start
            auto t_block_start = high_resolution_clock::now();

            Vec mu_vec = Vec::Constant(1, mu);  // Utils expects μ as a vector

            if (lin_solver_.getSolverType() == SolverType::MUMPS){
                utils_.Fun_Asym_kkt(x, y, z, s, mu_vec, A_kkt_ptr_); // symmetric A
            } else {
                utils_.Fun_A_kkt(x, y, z, s, mu_vec, A_kkt_ptr_); // asymmetric A
            }

            // std::cout<<"Before: "<< A_kkt_ptr_->coeffRef(1000, 1000) << "\n";
            // A_kkt_ptr_->coeffRef(1000, 1000) = A_kkt_ptr_->coeffRef(1000, 1000) + 1e-3;
            // std::cout<<" After: "<< A_kkt_ptr_->coeffRef(1000, 1000) << "\n";

            utils_.Fun_b_kkt(x, y, z, s, mu_vec, b_kkt_ptr_);
 
            casadi_real fk = utils_.Fun_f(x, y, z, s, mu_vec);

            utils_.Fun_df(x, y, z, s, mu_vec, df_ptr_);

            nx_ = (*df_ptr_).size();
            ny_ = y.size();
            nz_ = z.size();
            ns_ = s.size();

            // Split RHS
            Vec hk = (*b_kkt_ptr_).segment(nx_, ny_);
            Vec gk = (*b_kkt_ptr_).segment(nx_ + ny_, nz_) - s;
            
            Eigen::SparseMatrix<casadi_real> dhk =
                Eigen::SparseMatrix<casadi_real>((*A_kkt_ptr_).transpose().middleRows(nx_, ny_).leftCols(nx_));
            Eigen::SparseMatrix<casadi_real> dgk =
                Eigen::SparseMatrix<casadi_real>((*A_kkt_ptr_).transpose().middleRows(nx_ + ny_, nz_).leftCols(nx_));

            auto t_block_end = high_resolution_clock::now();
            auto t_block_elapsed = duration_cast<std::chrono::microseconds>(t_block_end - t_block_start).count();
            // std::cout << "[LieIPM] Block execution time: " << t_block_elapsed << " us" << std::endl;

            casadi_real sd;
            if ((ny_ + nz_) > 0) {
                sd = std::max(static_cast<casadi_real>(param_.s_max),
                              static_cast<casadi_real>((y.lpNorm<1>() + z.lpNorm<1>()) /
                                                       static_cast<casadi_real>(ny_ + nz_)))
                     / param_.s_max;
            } else {
                sd = static_cast<casadi_real>(1.0);
            }

            casadi_real sc;
            if (nz_ > 0) {
                sc = std::max(static_cast<casadi_real>(param_.s_max),
                              static_cast<casadi_real>(z.lpNorm<1>() / static_cast<casadi_real>(nz_)))
                     / param_.s_max;
            } else {
                sc = static_cast<casadi_real>(1.0);
            }

            // KKT violation metrics
            casadi_real E, E0;
            casadi_real term1_unscaled = 0, term2_unscaled = 0, term3_unscaled = 0;

            if (nz_ > 0) {
                term1_unscaled =
                    ((*df_ptr_).template cast<casadi_real>() +
                      dhk.transpose() * y.template cast<casadi_real>() +
                      dgk.transpose() * z.template cast<casadi_real>()).lpNorm<Eigen::Infinity>();
                casadi_real term1 = term1_unscaled / sd;
                term2_unscaled = hk.template cast<casadi_real>().lpNorm<Eigen::Infinity>();
                casadi_real term2 = term2_unscaled;
                term3_unscaled =
                    (z.template cast<casadi_real>().cwiseProduct(s.template cast<casadi_real>()) -
                      Vec::Constant(nz_, mu)).lpNorm<Eigen::Infinity>();
                casadi_real term3 = term3_unscaled / sc;
                E  = std::max({term1, term2, term3});

                term1_0_unscaled = term1_unscaled;
                casadi_real term1_0 = term1_0_unscaled / sd;
                term2_0_unscaled = term2_unscaled;
                casadi_real term2_0 = term2_0_unscaled;
                term3_0_unscaled =
                    (z.template cast<casadi_real>().cwiseProduct(s.template cast<casadi_real>()))
                        .lpNorm<Eigen::Infinity>();
                casadi_real term3_0 = term3_0_unscaled / sc;
                E0 = std::max({term1_0, term2_0, term3_0});
            } else {
                term1_unscaled =
                    ((*df_ptr_).template cast<casadi_real>() +
                      dhk.transpose() * y.template cast<casadi_real>()).lpNorm<Eigen::Infinity>();
                casadi_real term1 = term1_unscaled / sd;
                term2_unscaled = hk.template cast<casadi_real>().lpNorm<Eigen::Infinity>();
                casadi_real term2 = term2_unscaled;
                E  = std::max(term1, term2);
                E0 = E;
                term1_0_unscaled = term1_unscaled;
                term2_0_unscaled = term2_unscaled;
            }

            barrier_kkt_violation_log.push_back(E);
            overall_kkt_violation_log.push_back(E0);
            
            // std::cout <<"[LieIPM] KKT violaiton: " << E0 <<std::endl;
            
            // Termination test
            if (E0 < param_.overall_kkt_acceptance_rate) {
                flag = 1;  // Converged
                // std::cout << "[LieIPM] Converged at iter " << iter
                //           << ", E0=" << E0
                //           << " | unscaled: term1=" << term1_0_unscaled
                //           << ", term2=" << term2_0_unscaled
                //           << ", term3=" << term3_0_unscaled
                //           << " (sd=" << sd << ", sc=" << sc << ")" << std::endl;
                break;
            }

            if ( now() - t0 > param_.time_limit) {
                flag = -2;
                break;
            }
            
            // Barrier parameter update
            if (E < 10 * mu) {
                mu = std::min(static_cast<casadi_real>(param_.mu_decaying_rate * mu),
                              static_cast<casadi_real>(param_.mu_decaying_rate_sq *
                                                       std::pow(static_cast<casadi_real>(mu), 1.999)));
                mu = std::max(static_cast<casadi_real>(mu),
                              static_cast<casadi_real>(param_.overall_kkt_acceptance_rate / 10.0));
            }

            // const auto t_lin_0 = high_resolution_clock::now();

            // std::cout << "(rows, cols) = " << A_kkt_ptr_->rows() << ", " << A_kkt_ptr_->cols() << " dim_x = " << nx_ << std::endl;;
            Vec sol_k = Vec::Zero(nx_ + ny_ + nz_ + ns_);
            if (lin_solver_.getSolverType() == SolverType::MUMPS) {
                lin_solver_.factorize();

                Vec A_diag = Vec::Zero(A_kkt_ptr_->rows());
                for (int i = 0; i < A_kkt_ptr_->rows(); ++i) {
                    A_diag(i) = A_kkt_ptr_->coeff(i, i);
                }

                // const auto t_lin_1 = high_resolution_clock::now();
                int ic_count = this->InertiaCorrection(mu, A_diag);
                // const auto t_lin_2 = high_resolution_clock::now();
                // auto elapsed_ms = duration_cast<std::chrono::microseconds>(t_lin_2 - t_lin_1).count(); // long long, 毫秒
                // std::cout << "[LieIPM - IC] execution time: " << elapsed_ms << " us" << std::endl;

                // std::cout << "[LieIPM] Inertia correction count: " << ic_count << std::endl;
                // lin_solver_.getInfo(&ic_utils_); // 获取惯性数目
                // std::cout << "[MUMPS] n_minus: " << ic_utils_.n_minus
                //           << ", n_zero: " << ic_utils_.flag_zeros << std::endl;

                if (ic_count < 0) {
                    flag = -2; // Inertia correction failed
                    break;
                }

                Vec b_kkt_new = (*b_kkt_ptr_).segment(0, nx_ + ny_ + nz_);
                b_kkt_new.tail(nz_) =
                    b_kkt_new.tail(nz_) - (*b_kkt_ptr_).tail(ns_).cwiseQuotient(z);
                
                Vec sol_k_xyz_2 = -lin_solver_.solve(&b_kkt_new);
                
                sol_k.head(nx_ + ny_ + nz_) = sol_k_xyz_2;
                sol_k.tail(ns_) =
                    ((-(*b_kkt_ptr_).tail(ns_).template cast<casadi_real>()) -
                     s.template cast<casadi_real>().cwiseProduct(sol_k_xyz_2.tail(nz_).template cast<casadi_real>()))
                        .cwiseQuotient(z.template cast<casadi_real>());

            } else {
                lin_solver_.factorize(A_kkt_ptr_);
                sol_k = -lin_solver_.solve(b_kkt_ptr_);

                // const auto t_lin_1 = high_resolution_clock::now();
                // auto elapsed_ms = duration_cast<std::chrono::microseconds>(t_lin_1 - t_lin_0).count(); // long long, 毫秒
                // std::cout << "[EigenLU] Lin solver execution time: " << elapsed_ms << " us" << std::endl;
            }

            // TODO, we need a module that perform inertia correction
            // pass (ptr_, diag_original, dimension, ic_utils) to fun_ic.
            
            if ((sol_k.array().isNaN()).any() ||
                sol_k.template lpNorm<Eigen::Infinity>() > static_cast<casadi_real>(1e8)) {
                flag = -3; // Numerical failure
                std::cout << "[LieIPM] Numerical failure detected in solution at iter " << iter
                          << ", stopping iterations.\n";
                break;
            }

            const Vec& dx   = sol_k.head(nx_);
            const Vec& dy   = sol_k.segment(nx_, ny_);
            const Vec& dz   = sol_k.segment(nx_ + ny_, nz_);
            const Vec& ds   = sol_k.tail(ns_);

            casadi_real progress = (*df_ptr_).dot(dx);
            if (ns_ > 0) {
                progress -= mu * (ds.array() / s.array()).sum();
            }

            casadi_real tau = std::max(param_.tau_min, static_cast<casadi_real>(1.0) - mu);

            casadi_real alpha_s_max = 1.0, alpha_z_max = 1.0;
            if ((ds.array() < static_cast<casadi_real>(0)).any()) {
                Vec alpha_s_arr = (-tau * s.array() / ds.array()).matrix();
                alpha_s_arr =
                    (ds.array() < static_cast<casadi_real>(0))
                        .select(alpha_s_arr, static_cast<casadi_real>(2.0));
                alpha_s_max =
                    std::min(static_cast<casadi_real>(alpha_s_arr.minCoeff()),
                             static_cast<casadi_real>(1.0));
            }

            if ((dz.array() < static_cast<casadi_real>(0)).any()) {
                Vec alpha_z_arr = (-tau * z.array() / dz.array()).matrix();
                alpha_z_arr =
                    (dz.array() < static_cast<casadi_real>(0))
                        .select(alpha_z_arr, static_cast<casadi_real>(2.0));
                alpha_z_max =
                    std::min(static_cast<casadi_real>(alpha_z_arr.minCoeff()),
                             static_cast<casadi_real>(1.0));
            }

            // --- Filter + Armijo line-search ---
            casadi_real alpha = 1.0;
            casadi_real r0 = hk.lpNorm<1>() + gk.cwiseMax(0.0).sum();
            casadi_real r0_min = 1e-4 * std::max((casadi_real)1.0, r0);

            int flag_ls = -1;
            int ct      = 1;

            Vec sol_exec(dx.size() + dy.size() + dz.size() + ds.size());

            // auto tt_lin_0 = high_resolution_clock::now();
            while (true) {
                sol_exec << alpha * dx, alpha * dy, alpha_z_max * dz, alpha_s_max * ds;
                
                utils_.Fun_retract(x, y, z, s, sol_exec, x_new, y_new, z_new, s_new);

                Vec mu_trial = Vec::Constant(1, mu);
                casadi_real f_trial = utils_.Fun_f(x_new, y_new, z_new, s_new, mu_trial);
                utils_.Fun_hg(x_new, y_new, z_new, s_new, mu_trial, h_tiral_ptr_, g_trial_ptr_);
                const Vec& hk_trial = *h_tiral_ptr_;
                const Vec& gk_trial = *g_trial_ptr_;
                
                casadi_real r = hk_trial.template lpNorm<1>() + gk_trial.cwiseMax(0.0).sum();

                bool cond1 = progress < 0.0;
                bool cond2 = cond1 ? alpha * std::pow(-progress, param_.s_psi) >
                                     param_.delta * std::pow(r, param_.s_theta)
                                   : false;
                bool cond3 = r <= r0_min;

                if (cond1 && cond2 && cond3) {
                    if (f_trial - mu * s_new.array().log().sum() <
                        fk      - mu * s.array().log().sum() +
                        param_.armojo_coeff * alpha * progress) {
                        flag_ls = 1; // Accepted by Armijo
                        // std::cout << "[LieIPM] Line search accepted at iter: " << ct
                        //           << " with Armijo condition.\n";
                        break;
                    }
                } else {
                    bool alt1 = r < (1 - param_.barrier_kkt_acceptace_rate) * r0;
                    bool alt2 = f_trial - mu * s_new.array().log().sum() <
                                fk      - mu * s.array().log().sum() -
                                param_.barrier_kkt_acceptace_rate * r0;
                    if (alt1 || alt2) {
                        flag_ls = 2; // Accepted by filter
                        // std::cout << "[LieIPM] Line search accepted at iter by filte: " << ct <<"\n";
                        break;
                    }
                }
                if (ct > param_.max_line_search_count) {
                    flag_ls = -1; // Line search failed
                    // std::cout << "[LieIPM] Line search FAILED at iter: " << ct <<"\n";
                    break;
                    
                }
                alpha *= param_.beta;
                ++ct;
            }
            // auto tt_lin_1 = high_resolution_clock::now();
            // elapsed_ms = duration_cast<std::chrono::microseconds>(tt_lin_1 - tt_lin_0).count(); // long long, 毫秒
            // std::cout << "[LieIPM] line search execution time: " << elapsed_ms << " us" << std::endl;

            line_search_ct_log.push_back(ct);
            line_search_flag_log.push_back(flag_ls);

            // --- Accept step ---
            x.swap(x_new);
            y.swap(y_new);
            z.swap(z_new);
            s.swap(s_new);

            if (nz_ > 0 && ns_ > 0) {
                Vec s_safe = s.array().max(static_cast<casadi_real>(1e-16));
                Vec z_lo   = (mu / 1e10) * s_safe.cwiseInverse();
                Vec z_hi   = (1e10 * mu) * s_safe.cwiseInverse();
                z = z.cwiseMax(z_lo).cwiseMin(z_hi);
            }

            cost_log.push_back(fk);
            mu_log.push_back(mu);

            flag = 0; // Continue iterations
            t_forloop_log.push_back(now() - t0);
            // std::cout<<std::endl;
        }

        // Fill output structure
        data.t_total = now() - t0;
        data.xu      = x;
        data.s       = s;
        data.mu      = z;
        data.lam     = y;
        data.cost_log                  = std::move(cost_log);
        data.mu_log                    = std::move(mu_log);
        data.barrier_kkt_violation_log = std::move(barrier_kkt_violation_log);
        data.overall_kkt_violation_log = std::move(overall_kkt_violation_log);
        data.dual_infeas               = term1_0_unscaled;
        data.primal_infeas             = term2_0_unscaled;
        data.comp_infeas               = term3_0_unscaled;
        data.line_search_ct_log        = std::move(line_search_ct_log);
        data.line_search_flag_log      = std::move(line_search_flag_log);
        data.nnz_log                   = std::move(nnz_log);
        data.t_lin_sys_log             = std::move(t_lin_sys_log);
        data.t_forloop_log             = std::move(t_forloop_log);
        data.iter = iter;
        data.flag = flag;
        return data;
    }

private:
    LieIPMParam param_;
    Utils       utils_;
    Solver& lin_solver_;
    InertiaCorrectionUtils ic_utils_;

    int nx_ = 0; 
    int ny_ = 0;
    int nz_ = 0; 
    int ns_ = 0;

    // Vec A_diag_; // Diagonal of the KKT matrix A_kkt, used for inertia correction

    // Buffer and Eigen::Map pointer declarations
    alignas(Eigen::Map<SpMat>) char A_kkt_buffer_[sizeof(Eigen::Map<SpMat>)];
    Eigen::Map<SpMat>* A_kkt_ptr_;

    // const version, without inertia correction
    // alignas(Eigen::Map<const SpMat>) char A_kkt_buffer_[sizeof(Eigen::Map<const SpMat>)];
    // Eigen::Map<const SpMat>* A_kkt_ptr_;

    alignas(Eigen::Map<const Vec>) char df_buffer_[sizeof(Eigen::Map<const Vec>)];
    Eigen::Map<const Vec>* df_ptr_;

    alignas(Eigen::Map<const Vec>) char b_kkt_buffer_[sizeof(Eigen::Map<const Vec>)];
    Eigen::Map<const Vec>* b_kkt_ptr_;

    alignas(Eigen::Map<const Vec>) char h_buffer_[sizeof(Eigen::Map<const Vec>)];
    Eigen::Map<const Vec>* h_tiral_ptr_;

    alignas(Eigen::Map<const Vec>) char g_buffer_[sizeof(Eigen::Map<const Vec>)];
    Eigen::Map<const Vec>* g_trial_ptr_;

    static double now() {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    }

    int InertiaCorrection(const casadi_real mu, const Vec& A_diag) {
        int ic_count = 0;
        ic_utils_.delta_c = 0.0;
        lin_solver_.getInfo(&ic_utils_);
        
        // if there are zero eigenvalues
        bool is_zeroevals = ic_utils_.flag_zeros == -6 || ic_utils_.flag_zeros == -10;
        bool is_negevals  = ic_utils_.n_minus != (ny_ + nz_);
        bool ic_required  = is_negevals || is_zeroevals;

        // std::cout << "[LieIPM] Inertia correction required: " << ic_required
        //           << ", n_minus: " << ic_utils_.n_minus
        //           << ", n_zero: " << ic_utils_.flag_zeros << std::endl;
        
        if (ic_required) {
            // IC-2: apply delta_c when zero eigenvalues detected OR when n_minus is
            // smaller than expected (some eigenvalues that should be negative are
            // near-zero/positive due to near-rank-deficient constraint Jacobian).
            // MATLAB's count_inertia uses a 1e-12 threshold which catches near-zero
            // eigenvalues that MUMPS reports as positive (flag_zeros stays 0).
            bool n_minus_too_few = ic_utils_.n_minus < (ny_ + nz_);
            if (is_zeroevals || n_minus_too_few) {
                ic_utils_.delta_c = ic_utils_.delta_bar_c * pow(mu, ic_utils_.k_c);
            }

            // IC-3
            if (ic_utils_.delta_w_last == 0.0) {
                ic_utils_.delta_w = ic_utils_.delta_bar_w_0;
            } else{
                ic_utils_.delta_w = std::max(ic_utils_.delta_bar_w_min,
                                             ic_utils_.k_w_minus * ic_utils_.delta_w_last);
            }
        }

        // while(ic_required || ic_count < 2) {
        while(ic_required) {
            // perturb blk of x
            ic_count++;
            // std::cout << "[LieIPM] delta_w_last: " << ic_utils_.delta_w_last << std::endl;
            // std::cout << "[LieIPM] delta_w: " << ic_utils_.delta_w << std::endl;
            for (int k = 0; k < nx_; ++k) {
                (*A_kkt_ptr_).coeffRef(k, k) = A_diag[k] + ic_utils_.delta_w;
            }

            // perturb blk of y only
            for (int k = 0; k < ny_; ++k) {
                (*A_kkt_ptr_).coeffRef(nx_ + k, nx_ + k) = A_diag[nx_ + k] - ic_utils_.delta_c;
            }

            // IC-4
            lin_solver_.factorize();
            lin_solver_.getInfo(&ic_utils_);
            
            is_zeroevals = ic_utils_.flag_zeros == -6 || ic_utils_.flag_zeros == -10;
            is_negevals  = ic_utils_.n_minus != (ny_ + nz_);
            ic_required  = is_negevals || is_zeroevals;

            if (ic_required) {
                // IC-5
                if (ic_utils_.delta_w_last == 0.0) {
                    ic_utils_.delta_w = ic_utils_.k_bar_w_plus * ic_utils_.delta_w;
                } else {
                    ic_utils_.delta_w = ic_utils_.k_w_plus * ic_utils_.delta_w;
                }
            } else {
                ic_utils_.delta_w_last = ic_utils_.delta_w;
                // std::cout << "[LieIPM] [Inertia correction] successful after " << ic_count <<" iterations."<< std::endl;
                break;
            }

            if (ic_utils_.delta_w > ic_utils_.delta_bar_w_max) {
                // std::cout << "[LieIPM] [Inertia correction] failed, delta_w exceeds max limit!" << std::endl;
                return -1; // IC failed
            }
        }
        return ic_count;
    }
};