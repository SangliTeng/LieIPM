/*
 * exp_traj.cpp  –  LieIPM batch solver guided by a YAML config file.
 *
 * Usage:
 *   ./exp_traj <config.yaml>
 *
 * The YAML must contain at minimum: input_file, output_file.
 * See config/LieIPM/ for full schemas.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <matio.h>

#include "CasadiFunctionWrapper.h"
#include "LieIPM.h"
#include "LieIPMCasadiUtils.h"
#include "LinearSolverWrapper.h"

#include "A_kkt.h"
#include "Asym_kkt.h"
#include "b_kkt.h"
#include "df.h"
#include "f.h"
#include "hg.h"
#include "Retraction_xyz.h"

namespace fs = std::filesystem;

// ── SO(3) exponential map ─────────────────────────────────────────────────────

static Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    if (theta < 1e-10) return Eigen::Matrix3d::Identity();
    Eigen::Matrix3d W;
    W <<    0.0, -w(2),  w(1),
           w(2),   0.0, -w(0),
          -w(1),  w(0),   0.0;
    return Eigen::Matrix3d::Identity()
         + (std::sin(theta) / theta) * W
         + ((1.0 - std::cos(theta)) / (theta * theta)) * (W * W);
}

// ── Configuration ─────────────────────────────────────────────────────────────

struct ExpConfig {
    std::string input_file;
    std::string output_file;
    std::string sol_file;
    bool        save_sol  = false;
    std::string input_var = "cpp_data_all";
    int  start_1based  = 1;
    int  end_1based    = -1;   // -1 = all
    int  n_state_steps = 21;
    int  n_input_steps = 20;
    int  state_dim     = 24;
    int  input_dim     = 4;
    int  noise_seed    = 1;
    double delta_R = 0, delta_p = 0, delta_F = 0, delta_v = 0;
    double delta_u = 0, delta_y = 0, delta_z = 0, delta_s = 0, delta_mu = 0;
    bool   trivial_dual_enable = false;
    double y_val = 0.0, z_val = 0.1, s_val = 0.1, mu_val = 0.1;
    LieIPMParam ipm;
};

static ExpConfig load_config(const std::string& yaml_path) {
    auto trim = [](std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                [](unsigned char c){ return !std::isspace(c); }));
        s.erase(std::find_if(s.rbegin(), s.rend(),
                [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
        return s;
    };
    auto unquote = [&](std::string s) {
        s = trim(s);
        if (s.size() >= 2 && s.front() == s.back() &&
                (s.front() == '"' || s.front() == '\''))
            s = s.substr(1, s.size() - 2);
        return s;
    };

    std::ifstream fin(yaml_path);
    if (!fin) throw std::runtime_error("Cannot open yaml: " + yaml_path);

    std::unordered_map<std::string, std::string> kv;
    for (std::string line; std::getline(fin, line); ) {
        auto hpos = line.find('#');
        if (hpos != std::string::npos) line = line.substr(0, hpos);
        line = trim(line);
        if (line.empty()) continue;
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        auto key = trim(line.substr(0, colon));
        auto val = unquote(line.substr(colon + 1));
        if (!key.empty() && !val.empty()) kv[key] = val;
    }

    if (!kv.count("input_file") || !kv.count("output_file"))
        throw std::runtime_error("yaml must contain input_file and output_file");

    auto getd = [&](const char* k, double def) -> double {
        return kv.count(k) ? std::stod(kv.at(k)) : def;
    };
    auto geti = [&](const char* k, int def) -> int {
        return kv.count(k) ? std::stoi(kv.at(k)) : def;
    };
    auto getb = [&](const char* k, bool def) -> bool {
        return kv.count(k) ? (kv.at(k) == "true") : def;
    };

    ExpConfig cfg;
    cfg.input_file  = kv.at("input_file");
    cfg.output_file = kv.at("output_file");
    if (kv.count("sol_file"))  cfg.sol_file  = kv.at("sol_file");
    if (kv.count("input_var")) cfg.input_var = kv.at("input_var");
    cfg.save_sol      = getb("save_sol",      false);
    cfg.start_1based  = geti("start_1based",   1);
    cfg.end_1based    = geti("end_1based",     -1);
    if (kv.count("num_cases")) {
        const int nc = std::stoi(kv.at("num_cases"));
        if (nc <= 0) throw std::runtime_error("num_cases must be > 0");
        cfg.end_1based = cfg.start_1based + nc - 1;
    }
    cfg.n_state_steps = geti("n_state_steps", 21);
    cfg.n_input_steps = geti("n_input_steps", 20);
    cfg.state_dim     = geti("state_dim",     24);
    cfg.input_dim     = geti("input_dim",      4);
    cfg.noise_seed    = geti("noise_seed",     1);
    cfg.delta_R       = getd("delta_R",  0.0);
    cfg.delta_p       = getd("delta_p",  0.0);
    cfg.delta_F       = getd("delta_F",  0.0);
    cfg.delta_v       = getd("delta_v",  0.0);
    cfg.delta_u       = getd("delta_u",  0.0);
    cfg.delta_y       = getd("delta_y",  0.0);
    cfg.delta_z       = getd("delta_z",  0.0);
    cfg.delta_s       = getd("delta_s",  0.0);
    cfg.delta_mu      = getd("delta_mu", 0.0);
    cfg.trivial_dual_enable = getb("enable", false);
    cfg.y_val         = getd("y_val",  0.0);
    cfg.z_val         = getd("z_val",  0.1);
    cfg.s_val         = getd("s_val",  0.1);
    cfg.mu_val        = getd("mu_val", 0.1);
    cfg.ipm.iter_max                    = geti("iter_max",                    1000);
    cfg.ipm.overall_kkt_acceptance_rate = getd("overall_kkt_acceptance_rate", 1e-6);
    cfg.ipm.barrier_kkt_acceptace_rate  = getd("barrier_kkt_acceptance_rate", 1e-3);
    cfg.ipm.time_limit                  = getd("time_limit",                 10.0);
    return cfg;
}

// ── matio helpers ─────────────────────────────────────────────────────────────

// Extract a field from a struct cell element as a Vec.
static Vec cell_field_vec(matvar_t* s, const char* field) {
    matvar_t* fv = Mat_VarGetStructFieldByName(s, field, 0);
    if (!fv || !fv->data)
        throw std::runtime_error(std::string("Missing cell field: ") + field);
    const size_t n = fv->nbytes / sizeof(double);
    const double* d = static_cast<const double*>(fv->data);
    Vec v(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) v[static_cast<int>(i)] = static_cast<casadi_real>(d[i]);
    return v;
}

// Create a MAT double-vector variable (matio copies the data).
static matvar_t* make_dvec(const char* name, const double* data, size_t n) {
    size_t dims[2] = {n, 1};
    return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims,
                         const_cast<double*>(data), 0);
}

static matvar_t* make_dvec(const char* name, const Vec& v) {
    // Vec is Eigen::VectorXd (casadi_real=double), so .data() is double*.
    return make_dvec(name, v.data(), static_cast<size_t>(v.size()));
}

static matvar_t* make_string(const char* name, const std::string& s) {
    const std::string& text = s.empty() ? std::string("") : s;
    size_t dims[2] = {1, text.size()};
    return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, dims,
                         const_cast<char*>(text.c_str()), 0);
}

// ── Perturbation ──────────────────────────────────────────────────────────────

// Perturb the primal trajectory x (SO(3) states + inputs).
// State layout per step: R(9,col-major), p(3), F(9,col-major), v(3).
static void perturb_x(Vec& x, const ExpConfig& cfg, std::mt19937& rng) {
    const int ns = cfg.n_state_steps;
    const int sd = cfg.state_dim;   // 24
    const int u_off = ns * sd;

    for (int k = 0; k < ns; ++k) {
        const int off = k * sd;
        if (cfg.delta_R > 0.0) {
            Eigen::Map<Eigen::Matrix3d> R(x.data() + off);
            std::normal_distribution<double> nd(0.0, cfg.delta_R);
            Eigen::Vector3d w(nd(rng), nd(rng), nd(rng));
            R = (R * so3_exp(w)).eval();
        }
        if (cfg.delta_p > 0.0) {
            std::normal_distribution<double> nd(0.0, cfg.delta_p);
            for (int i = 9; i < 12; ++i) x[off + i] += nd(rng);
        }
        if (cfg.delta_F > 0.0) {
            Eigen::Map<Eigen::Matrix3d> F(x.data() + off + 12);
            std::normal_distribution<double> nd(0.0, cfg.delta_F);
            Eigen::Vector3d w(nd(rng), nd(rng), nd(rng));
            F = (F * so3_exp(w)).eval();
        }
        if (cfg.delta_v > 0.0) {
            std::normal_distribution<double> nd(0.0, cfg.delta_v);
            for (int i = 21; i < 24; ++i) x[off + i] += nd(rng);
        }
    }
    if (cfg.delta_u > 0.0) {
        std::normal_distribution<double> nd(0.0, cfg.delta_u);
        for (int k = 0; k < cfg.n_input_steps * cfg.input_dim; ++k)
            x[u_off + k] += nd(rng);
    }
}

// Log-multiplicative perturbation: v[i] *= exp(Normal(0, delta)).
static void perturb_log(Vec& v, double delta, std::mt19937& rng) {
    if (delta <= 0.0) return;
    std::normal_distribution<double> nd(0.0, delta);
    for (int i = 0; i < v.size(); ++i) v[i] *= std::exp(nd(rng));
}

// Additive perturbation on equality multipliers y.
static void perturb_y(Vec& y, double delta, std::mt19937& rng) {
    if (delta <= 0.0) return;
    std::normal_distribution<double> nd(0.0, delta);
    for (int i = 0; i < y.size(); ++i) y[i] += nd(rng);
}

// ── Per-problem solve ─────────────────────────────────────────────────────────

struct SolRecord {
    bool   ok  = false;
    int    idx = 0;    // 0-based global index
    LieIPMData data;
    std::string error_msg;
    // Original (unperturbed) problem parameters for sol_file
    Vec x0_orig, ssw_orig, tsw_orig, iw_orig, xd_orig, u_orig, dt_orig;
};

static SolRecord solve_one(const ExpConfig& cfg, int idx_0based,
                           Vec x, Vec y, Vec z, Vec s, casadi_real mu,
                           Vec x0, Vec ssw, Vec tsw, Vec iw, Vec xd, Vec u, Vec dt)
{
    SolRecord rec;
    rec.idx       = idx_0based;
    rec.x0_orig   = x0;
    rec.ssw_orig  = ssw;
    rec.tsw_orig  = tsw;
    rec.iw_orig   = iw;
    rec.xd_orig   = xd;
    rec.u_orig    = u;
    rec.dt_orig   = dt;

    try {
        // Build per-call wrappers (each solve gets its own isolated state).
        CasadiFunctionWrapper f_wrap(&f_incref, &f_decref, &f_n_in, &f_n_out,
                                     &f_sparsity_in, &f_sparsity_out, &f_work,
                                     &f, &f_name_in, &f_name_out);
        CasadiFunctionWrapper df_wrap(&df_incref, &df_decref, &df_n_in, &df_n_out,
                                      &df_sparsity_in, &df_sparsity_out, &df_work,
                                      &df, &df_name_in, &df_name_out);
        CasadiFunctionWrapper A_wrap(&A_kkt_incref, &A_kkt_decref, &A_kkt_n_in, &A_kkt_n_out,
                                     &A_kkt_sparsity_in, &A_kkt_sparsity_out, &A_kkt_work,
                                     &A_kkt, &A_kkt_name_in, &A_kkt_name_out);
        CasadiFunctionWrapper Asym_wrap(&Asym_kkt_incref, &Asym_kkt_decref,
                                        &Asym_kkt_n_in, &Asym_kkt_n_out,
                                        &Asym_kkt_sparsity_in, &Asym_kkt_sparsity_out,
                                        &Asym_kkt_work, &Asym_kkt,
                                        &Asym_kkt_name_in, &Asym_kkt_name_out);
        CasadiFunctionWrapper b_wrap(&b_kkt_incref, &b_kkt_decref, &b_kkt_n_in, &b_kkt_n_out,
                                     &b_kkt_sparsity_in, &b_kkt_sparsity_out, &b_kkt_work,
                                     &b_kkt, &b_kkt_name_in, &b_kkt_name_out);
        CasadiFunctionWrapper hg_wrap(&hg_incref, &hg_decref, &hg_n_in, &hg_n_out,
                                      &hg_sparsity_in, &hg_sparsity_out, &hg_work,
                                      &hg, &hg_name_in, &hg_name_out);
        CasadiFunctionWrapper ret_wrap(&Retraction_xyz_incref, &Retraction_xyz_decref,
                                       &Retraction_xyz_n_in, &Retraction_xyz_n_out,
                                       &Retraction_xyz_sparsity_in, &Retraction_xyz_sparsity_out,
                                       &Retraction_xyz_work, &Retraction_xyz,
                                       &Retraction_xyz_name_in, &Retraction_xyz_name_out);
        LieIPMCasadiUtils utils(f_wrap, df_wrap, A_wrap, Asym_wrap, b_wrap, hg_wrap, ret_wrap);

        // Set problem parameters (inputs 5-11) on all casadi wrappers.
        auto setParam = [&](CasadiFunctionWrapper& w) {
            w.set_input(5,  std::vector<casadi_real>(x0.data(),  x0.data()  + x0.size()));
            w.set_input(6,  std::vector<casadi_real>(ssw.data(), ssw.data() + ssw.size()));
            w.set_input(7,  std::vector<casadi_real>(tsw.data(), tsw.data() + tsw.size()));
            w.set_input(8,  std::vector<casadi_real>(iw.data(),  iw.data()  + iw.size()));
            w.set_input(9,  std::vector<casadi_real>(xd.data(),  xd.data()  + xd.size()));
            w.set_input(10, std::vector<casadi_real>(u.data(),   u.data()   + u.size()));
            w.set_input(11, std::vector<casadi_real>(dt.data(),  dt.data()  + dt.size()));
        };
        setParam(utils.f);
        setParam(utils.df);
        setParam(utils.A_kkt);
        setParam(utils.Asym_kkt);
        setParam(utils.b_kkt);
        setParam(utils.hg);

        MUMPSSolverWrapper<casadi_real, long long> lin_solver;
        LieIPM<LieIPMCasadiUtils, MUMPSSolverWrapper<casadi_real, long long>>
            ipm(cfg.ipm, utils, lin_solver);

        rec.data = ipm.solve(x, y, z, s, mu);
        rec.ok   = true;
    } catch (const std::exception& e) {
        rec.error_msg = e.what();
    }
    return rec;
}

// ── Output .mat saving ────────────────────────────────────────────────────────

static void save_stat_mat(const std::string& path,
                          const std::vector<SolRecord>& recs) {
    fs::create_directories(fs::path(path).parent_path());
    mat_t* mat = Mat_CreateVer(path.c_str(), nullptr, MAT_FT_MAT5);
    if (!mat) throw std::runtime_error("Cannot create mat: " + path);

    const int N = static_cast<int>(recs.size());
    size_t cell_dims[2] = {static_cast<size_t>(N), 1};

    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    // Write a (N,1) cell array of double vectors.
    auto write_cell = [&](const char* name,
                          const std::vector<std::vector<double>>& vv) {
        matvar_t* ca = Mat_VarCreate(name, MAT_C_CELL, MAT_T_CELL, 2,
                                     cell_dims, nullptr, 0);
        if (!ca) return;
        for (int i = 0; i < N; ++i) {
            std::vector<double> d = vv[i];
            if (d.empty()) d.push_back(kNaN);
            size_t dims[2] = {d.size(), 1};
            matvar_t* mv = Mat_VarCreate(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE,
                                         2, dims, d.data(), 0);
            if (mv) Mat_VarSetCell(ca, i, mv);
        }
        Mat_VarWrite(mat, ca, MAT_COMPRESSION_ZLIB);
        Mat_VarFree(ca);
    };

    // Write a (N,1) cell array of strings.
    auto write_cell_str = [&](const char* name,
                              const std::vector<std::string>& ss) {
        matvar_t* ca = Mat_VarCreate(name, MAT_C_CELL, MAT_T_CELL, 2,
                                     cell_dims, nullptr, 0);
        if (!ca) return;
        for (int i = 0; i < N; ++i) {
            matvar_t* mv = make_string(nullptr, ss[i]);
            if (mv) Mat_VarSetCell(ca, i, mv);
        }
        Mat_VarWrite(mat, ca, MAT_COMPRESSION_ZLIB);
        Mat_VarFree(ca);
    };

    // Write a (N,1) double vector.
    auto write_vec = [&](const char* name, const std::vector<double>& v) {
        size_t dims[2] = {static_cast<size_t>(N), 1};
        matvar_t* mv = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE,
                                     2, dims, const_cast<double*>(v.data()), 0);
        Mat_VarWrite(mat, mv, MAT_COMPRESSION_NONE);
        Mat_VarFree(mv);
    };

    // Collect data across all records.
    std::vector<std::vector<double>> xu_all(N), s_all(N), mu_dual_all(N), lam_all(N);
    std::vector<std::vector<double>> cost_all(N), mu_log_all(N), bkkt_all(N), okkt_all(N);
    std::vector<std::vector<double>> ls_ct_all(N), ls_flag_all(N), tkkt_all(N), tic_all(N);
    std::vector<double> t_total(N, kNaN), t_no_eval(N, kNaN);
    std::vector<double> v_iter(N, -1.0), v_flag(N, -1.0);
    std::vector<std::string> filenames(N);

    for (int i = 0; i < N; ++i) {
        const auto& r = recs[i];
        filenames[i] = std::to_string(r.idx + 1);
        if (r.ok) {
            const auto& d = r.data;
            xu_all[i].assign(d.xu.data(), d.xu.data() + d.xu.size());
            s_all[i].assign(d.s.data(), d.s.data() + d.s.size());
            mu_dual_all[i].assign(d.mu.data(), d.mu.data() + d.mu.size());
            lam_all[i].assign(d.lam.data(), d.lam.data() + d.lam.size());
            for (auto v : d.cost_log) cost_all[i].push_back(static_cast<double>(v));
            for (auto v : d.mu_log)   mu_log_all[i].push_back(static_cast<double>(v));
            for (auto v : d.barrier_kkt_violation_log) bkkt_all[i].push_back(static_cast<double>(v));
            for (auto v : d.overall_kkt_violation_log) okkt_all[i].push_back(static_cast<double>(v));
            for (auto v : d.line_search_ct_log)   ls_ct_all[i].push_back(static_cast<double>(v));
            for (auto v : d.line_search_flag_log) ls_flag_all[i].push_back(static_cast<double>(v));
            for (auto v : d.t_lin_sys_log)  tkkt_all[i].push_back(static_cast<double>(v));
            for (auto v : d.t_forloop_log)  tic_all[i].push_back(static_cast<double>(v));
            t_total[i]  = d.t_total;
            t_no_eval[i] = d.t_total;
            v_iter[i]   = static_cast<double>(d.iter);
            v_flag[i]   = static_cast<double>(d.flag);
        }
    }

    write_cell("xu",                         xu_all);
    write_cell("s",                          s_all);
    write_cell("mu_dual",                    mu_dual_all);
    write_cell("lam",                        lam_all);
    write_cell("cost_log",                   cost_all);
    write_cell("mu_log",                     mu_log_all);
    write_cell("barrier_kkt_violation_log",  bkkt_all);
    write_cell("overall_kkt_violation_log",  okkt_all);
    write_cell("line_search_ct_log",         ls_ct_all);
    write_cell("line_search_flag_log",       ls_flag_all);
    write_cell("t_kkt_log",                  tkkt_all);
    write_cell("t_ic_log",                   tic_all);
    write_vec("t_total",  t_total);
    write_vec("t_no_eval", t_no_eval);
    write_vec("iter",     v_iter);
    write_vec("flag",     v_flag);
    write_cell_str("filenames", filenames);

    Mat_Close(mat);
}

// Save solution cell array (same struct format as input).
static void save_sol_mat(const std::string& path, const std::string& var_name,
                         const std::vector<SolRecord>& recs) {
    fs::create_directories(fs::path(path).parent_path());
    mat_t* mat = Mat_CreateVer(path.c_str(), nullptr, MAT_FT_MAT5);
    if (!mat) throw std::runtime_error("Cannot create mat: " + path);

    const int N = static_cast<int>(recs.size());
    size_t cell_dims[2] = {1, static_cast<size_t>(N)};

    matvar_t* ca = Mat_VarCreate(var_name.c_str(), MAT_C_CELL, MAT_T_CELL, 2,
                                 cell_dims, nullptr, 0);
    if (!ca) { Mat_Close(mat); throw std::runtime_error("Cannot create cell var"); }

    const char* field_names[] = {
        "x", "y", "z", "s", "mu",
        "x0", "stage_state_weights", "terminal_state_weights",
        "input_weights", "xd", "u", "dt"
    };
    constexpr unsigned N_FIELDS = 12;

    for (int i = 0; i < N; ++i) {
        const auto& r = recs[i];
        size_t struct_dims[2] = {1, 1};
        matvar_t* sv = Mat_VarCreateStruct(nullptr, 2, struct_dims,
                                           field_names, N_FIELDS);
        // Solution vectors (use last iterate even if not converged).
        double mu_scalar = r.data.mu_log.empty() ? 0.0
                         : static_cast<double>(r.data.mu_log.back());
        size_t sc_dims[2] = {1, 1};
        Mat_VarSetStructFieldByName(sv, "x",  0, make_dvec(nullptr, r.data.xu));
        Mat_VarSetStructFieldByName(sv, "y",  0, make_dvec(nullptr, r.data.lam));
        Mat_VarSetStructFieldByName(sv, "z",  0, make_dvec(nullptr, r.data.mu));
        Mat_VarSetStructFieldByName(sv, "s",  0, make_dvec(nullptr, r.data.s));
        Mat_VarSetStructFieldByName(sv, "mu", 0,
            Mat_VarCreate(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, sc_dims, &mu_scalar, 0));
        // Problem data.
        Mat_VarSetStructFieldByName(sv, "x0",                     0, make_dvec(nullptr, r.x0_orig));
        Mat_VarSetStructFieldByName(sv, "stage_state_weights",    0, make_dvec(nullptr, r.ssw_orig));
        Mat_VarSetStructFieldByName(sv, "terminal_state_weights", 0, make_dvec(nullptr, r.tsw_orig));
        Mat_VarSetStructFieldByName(sv, "input_weights",          0, make_dvec(nullptr, r.iw_orig));
        Mat_VarSetStructFieldByName(sv, "xd",                     0, make_dvec(nullptr, r.xd_orig));
        Mat_VarSetStructFieldByName(sv, "u",                      0, make_dvec(nullptr, r.u_orig));
        Mat_VarSetStructFieldByName(sv, "dt",                     0, make_dvec(nullptr, r.dt_orig));

        Mat_VarSetCell(ca, i, sv);
    }

    Mat_VarWrite(mat, ca, MAT_COMPRESSION_ZLIB);
    Mat_VarFree(ca);
    Mat_Close(mat);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    ExpConfig cfg;
    try {
        cfg = load_config(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "[INFO] input:  " << cfg.input_file  << "\n"
              << "[INFO] output: " << cfg.output_file << "\n";
    if (cfg.save_sol && !cfg.sol_file.empty())
        std::cout << "[INFO] sol:    " << cfg.sol_file << "\n";
    std::cout << "[INFO] noise_seed=" << cfg.noise_seed
              << "  delta_R="  << cfg.delta_R
              << "  delta_p="  << cfg.delta_p
              << "  delta_F="  << cfg.delta_F
              << "  delta_v="  << cfg.delta_v
              << "  delta_u="  << cfg.delta_u
              << "\n"
              << "[INFO]            "
              << "  delta_y="  << cfg.delta_y
              << "  delta_z="  << cfg.delta_z
              << "  delta_s="  << cfg.delta_s
              << "  delta_mu=" << cfg.delta_mu
              << (cfg.trivial_dual_enable
                  ? "  trivial_dual=ON"
                  : "  trivial_dual=off")
              << "\n";

    // Load cell array.
    mat_t* mat_in = Mat_Open(cfg.input_file.c_str(), MAT_ACC_RDONLY);
    if (!mat_in) {
        std::cerr << "[ERROR] Cannot open: " << cfg.input_file << "\n";
        return 1;
    }
    matvar_t* cell = Mat_VarRead(mat_in, cfg.input_var.c_str());
    if (!cell) {
        Mat_Close(mat_in);
        std::cerr << "[ERROR] Variable '" << cfg.input_var
                  << "' not found in " << cfg.input_file << "\n";
        return 1;
    }
    const int N = static_cast<int>(cell->dims[0] * cell->dims[1]);
    std::cout << "[INFO] " << N << " problems in file.\n";

    // Apply start/end range.
    const int start0 = cfg.start_1based - 1;
    const int end0   = (cfg.end_1based < 0) ? N - 1 : cfg.end_1based - 1;
    if (start0 < 0 || start0 >= N || end0 < start0 || end0 >= N) {
        Mat_VarFree(cell); Mat_Close(mat_in);
        std::cerr << "[ERROR] Invalid range: start=" << cfg.start_1based
                  << " end=" << cfg.end_1based << " total=" << N << "\n";
        return 1;
    }
    const int n_solve = end0 - start0 + 1;
    std::cout << "[INFO] Solving [" << cfg.start_1based << ", " << (end0+1)
              << "] (" << n_solve << " problems).\n\n";

    std::vector<SolRecord> records;
    records.reserve(n_solve);
    int n_converged = 0, n_infeasible = 0;
    const int nd = static_cast<int>(std::to_string(n_solve).size());

    for (int i = start0; i <= end0; ++i) {
        matvar_t* s = Mat_VarGetCell(cell, i);
        if (!s) {
            std::cerr << "[WARN] Null cell at " << i << "\n";
            SolRecord r; r.idx = i; records.push_back(r);
            continue;
        }

        Vec x   = cell_field_vec(s, "x");
        Vec y   = cell_field_vec(s, "y");
        Vec z   = cell_field_vec(s, "z");
        Vec sv2 = cell_field_vec(s, "s");
        Vec mu_vec = cell_field_vec(s, "mu");
        Vec x0  = cell_field_vec(s, "x0");
        Vec ssw = cell_field_vec(s, "stage_state_weights");
        Vec tsw = cell_field_vec(s, "terminal_state_weights");
        Vec iw  = cell_field_vec(s, "input_weights");
        Vec xd  = cell_field_vec(s, "xd");
        Vec u   = cell_field_vec(s, "u");
        Vec dt  = cell_field_vec(s, "dt");

        // Apply perturbations using a per-problem deterministic seed.
        std::mt19937 rng(static_cast<uint32_t>(cfg.noise_seed)
                         + 1000003u * static_cast<uint32_t>(i + 1));
        perturb_x(x, cfg, rng);
        perturb_y(y, cfg.delta_y, rng);
        perturb_log(z, cfg.delta_z, rng);
        perturb_log(sv2, cfg.delta_s, rng);
        if (cfg.delta_mu > 0.0) {
            std::normal_distribution<double> nd(0.0, cfg.delta_mu);
            mu_vec[0] *= static_cast<casadi_real>(std::exp(nd(rng)));
        }

        // Apply trivial_dual override if enabled.
        if (cfg.trivial_dual_enable) {
            y.fill(static_cast<casadi_real>(cfg.y_val));
            z.fill(static_cast<casadi_real>(cfg.z_val));
            sv2.fill(static_cast<casadi_real>(cfg.s_val));
            mu_vec[0] = static_cast<casadi_real>(cfg.mu_val);
        }

        std::cout << "[" << std::setw(nd) << (i - start0 + 1) << "/" << n_solve << "]  " << std::flush;

        SolRecord rec = solve_one(cfg, i, x, y, z, sv2, mu_vec[0],
                                  x0, ssw, tsw, iw, xd, u, dt);
        if (rec.error_msg.empty()) {
            const auto& d = rec.data;
            if (d.flag == 1) ++n_converged; else ++n_infeasible;
            double kkt = d.overall_kkt_violation_log.empty() ? -1.0
                       : static_cast<double>(d.overall_kkt_violation_log.back());
            std::cout << "iter=" << std::setw(4) << d.iter
                      << "  t=" << std::fixed << std::setprecision(3) << d.t_total << "s"
                      << "  flag=" << d.flag
                      << "  kkt=" << std::scientific << std::setprecision(2) << kkt
                      << "\n";
        } else {
            ++n_infeasible;
            std::cout << "ERROR  " << rec.error_msg << "\n";
        }
        records.push_back(std::move(rec));
    }

    Mat_VarFree(cell);
    Mat_Close(mat_in);

    std::cout << "\n[INFO] converged=" << n_converged
              << "  infeasible=" << n_infeasible
              << "  total=" << n_solve << "\n";

    try {
        save_stat_mat(cfg.output_file, records);
        std::cout << "[INFO] Stats saved to " << cfg.output_file << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to save stats: " << e.what() << "\n";
        return 1;
    }

    if (cfg.save_sol && !cfg.sol_file.empty()) {
        try {
            save_sol_mat(cfg.sol_file, cfg.input_var, records);
            std::cout << "[INFO] Sol saved to " << cfg.sol_file << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to save sol: " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}
