#pragma once

struct LieIPMParam {
    casadi_real beta         = 0.5;
    casadi_real mu_decaying_rate = 1.0 / 1.01;   // ≈0.9901
    casadi_real mu_decaying_rate_sq = 1e-2;
    casadi_real overall_kkt_acceptance_rate = 1e-4; // set a larger number for realtime deployment
    casadi_real barrier_kkt_acceptace_rate  = 1e-6;
    casadi_real delta = 1.0; // 1e-4;
    casadi_real s_theta = 1.10;
    casadi_real s_psi = 2.30;
    casadi_real armojo_coeff = 1e-4;
    int    max_line_search_count = 20;
    casadi_real tau_min = 0.995;
    casadi_real s_max = 100.0;
    int    iter_max = 100;

    casadi_real time_limit = 0.0095;
};

struct LieIPMData {
    std::conditional_t<std::is_same_v<casadi_real, double>, Eigen::VectorXd, Eigen::VectorXf> xu, s, mu, lam;
    double t_total;
    std::vector<casadi_real> cost_log, mu_log, barrier_kkt_violation_log, overall_kkt_violation_log;
    casadi_real dual_infeas = 0, primal_infeas = 0, comp_infeas = 0;
    std::vector<int>    line_search_ct_log, line_search_flag_log, nnz_log;
    std::vector<casadi_real> t_lin_sys_log, t_forloop_log;
    int iter = 0;
    int flag = -1;
};

struct InertiaCorrectionUtils {
    // int n_plus = 0.0; // Number of positive eigenvalues

    // The mumps interface returns the number of negative eigenvalues in infog[11]
    int n_minus = 0; // 
    int flag_zeros = 0; // -10 / -6 flag means zero eigenvalues

    casadi_real delta_c = 0.;
    casadi_real delta_w_last = 0.;
    casadi_real delta_w = 0.;

    // Parameters for inertia correction
    casadi_real delta_bar_w_min = 1e-20;
    casadi_real delta_bar_w_0   = 1e-4;
    casadi_real delta_bar_w_max = 1e40;
    casadi_real k_bar_w_plus    = 100;
    casadi_real k_w_plus        = 8;
    casadi_real k_w_minus       = 1.0 / 3.0;
    casadi_real k_c             = 1.0 / 4.0;
    casadi_real delta_bar_c     = 1e-8;

};
