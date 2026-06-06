// LieIPMCasadiUtils.h
#pragma once

#include "CasadiFunctionWrapper.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>

#include <chrono>
#include <iomanip>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::microseconds;

using Vec   = std::conditional_t<std::is_same_v<casadi_real, double>, Eigen::VectorXd, Eigen::VectorXf>;
using SpMat = Eigen::SparseMatrix<casadi_real, Eigen::ColMajor, long long>;

/**
 * @brief Utils for LieIPM using CasADi wrappers
 *
 * Provides unified setInput(idx, value) for any input index,
 * including treating mu as a vector input.
 * Wraps evaluation of f, df, A_kkt, b_kkt, hg, retract.
 */
struct LieIPMCasadiUtils {
    CasadiFunctionWrapper &f, &df, &A_kkt, &Asym_kkt, &b_kkt, &hg, &Retract;

    LieIPMCasadiUtils(
        CasadiFunctionWrapper &f_,
        CasadiFunctionWrapper &df_,
        CasadiFunctionWrapper &A_kkt_,
        CasadiFunctionWrapper &Asym_kkt_,
        CasadiFunctionWrapper &b_kkt_,
        CasadiFunctionWrapper &hg_,
        CasadiFunctionWrapper &Retract_
    ) : f(f_), df(df_), A_kkt(A_kkt_), Asym_kkt(Asym_kkt_), b_kkt(b_kkt_), hg(hg_), Retract(Retract_) {}

    // Objective scalar (mu as Vec)
    casadi_real Fun_f(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        setInput(f, 0, x);
        setInput(f, 1, y);
        setInput(f, 2, z);
        setInput(f, 3, s);
        setInput(f, 4, mu);
        f.eval();
        return f.output(0)[0];
    }

    // void Fun_f(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu, 
    //            Eigen::Map<const SpMat>*& mat_ptr) {
    //     setInput(f, 0, x);
    //     setInput(f, 1, y);
    //     setInput(f, 2, z);
    //     setInput(f, 3, s);
    //     setInput(f, 4, mu);
    //     f.eval();
    //     f.get_output_sparse(0, mat_ptr);
    // }


    // Objective gradient
    Vec Fun_df(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        setInput(df, 0, x);
        setInput(df, 1, y);
        setInput(df, 2, z);
        setInput(df, 3, s);
        setInput(df, 4, mu);
        df.eval();
        return Eigen::Map<const Vec>(df.output_ptr(0), df.output_size(0));
    }

    void Fun_df(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu,
               Eigen::Map<const Vec>*& mat_ptr) {
        // auto t1 = high_resolution_clock::now();  // 起始时间
        setInput(df, 0, x);
        setInput(df, 1, y);
        setInput(df, 2, z);
        setInput(df, 3, s);
        setInput(df, 4, mu);
        df.eval();
        new (mat_ptr) Eigen::Map<const Vec>(df.output_ptr(0), df.output_size(0));
        // auto t2 = high_resolution_clock::now();  // 结束时间
        // auto duration = duration_cast<microseconds>(t2 - t1).count();
        // std::cout << "[Casadi] [g, h]] assign value (v2) took " << duration << " µs" << std::endl;

        // df.get_output_sparse(0, mat_ptr);
        // int n = df.output_size(0);
        // return Eigen::Map<const Vec>(df.output_ptr(0), n);
    }


    // KKT matrix (sparse)
    SpMat Fun_A_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        setInput(A_kkt, 0, x);
        setInput(A_kkt, 1, y);
        setInput(A_kkt, 2, z);
        setInput(A_kkt, 3, s);
        setInput(A_kkt, 4, mu);
        auto t1 = high_resolution_clock::now();  // 起始时间
        A_kkt.eval();
        auto t2 = high_resolution_clock::now();  // 结束时间
        auto duration = duration_cast<microseconds>(t2 - t1).count();
        std::cout << "[Casadi] Akkt evaluation (v1) took " << duration << " µs" << std::endl;
        return A_kkt.get_output_sparse(0);
    }

    void Fun_A_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu, 
                   Eigen::Map<SpMat>*& mat_ptr) {
        setInput(A_kkt, 0, x);
        setInput(A_kkt, 1, y);
        setInput(A_kkt, 2, z);
        setInput(A_kkt, 3, s);
        setInput(A_kkt, 4, mu);
        A_kkt.eval();
        A_kkt.get_output_sparse(0, mat_ptr);
    }

    void Fun_A_kkt_get_buffer(Eigen::Map<SpMat>*& mat_ptr){
        A_kkt.eval();
        A_kkt.get_output_sparse(0, mat_ptr);
    }


    SpMat Fun_Asym_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        setInput(Asym_kkt, 0, x);
        setInput(Asym_kkt, 1, y);
        setInput(Asym_kkt, 2, z);
        setInput(Asym_kkt, 3, s);
        setInput(Asym_kkt, 4, mu);
        Asym_kkt.eval();
        return Asym_kkt.get_output_sparse(0);
    }

    void Fun_Asym_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu, 
                   Eigen::Map<SpMat>*& mat_ptr) {
        setInput(Asym_kkt, 0, x);
        setInput(Asym_kkt, 1, y);
        setInput(Asym_kkt, 2, z);
        setInput(Asym_kkt, 3, s);
        setInput(Asym_kkt, 4, mu);
        Asym_kkt.eval();
        Asym_kkt.get_output_sparse(0, mat_ptr);
    }

    void Fun_Asym_kkt_get_buffer(Eigen::Map<SpMat>*& mat_ptr){
        Asym_kkt.eval();
        Asym_kkt.get_output_sparse(0, mat_ptr);
    }

    void Fun_Asym_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu, 
                   int &rows, int &cols, int &nnz,
                const casadi_int* &colind, const casadi_int* &row,
                casadi_real* &values) {
        setInput(Asym_kkt, 0, x);
        setInput(Asym_kkt, 1, y);
        setInput(Asym_kkt, 2, z);
        setInput(Asym_kkt, 3, s);
        setInput(Asym_kkt, 4, mu);
        Asym_kkt.eval();
        Asym_kkt.get_output_buffer(0, rows, cols, nnz, colind, row, values);
    }


    // KKT RHS vector
    Vec Fun_b_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        setInput(b_kkt, 0, x);
        setInput(b_kkt, 1, y);
        setInput(b_kkt, 2, z);
        setInput(b_kkt, 3, s);
        setInput(b_kkt, 4, mu);
        b_kkt.eval();
        int n = b_kkt.output_size(0);
        return Eigen::Map<const Vec>(b_kkt.output_ptr(0), n);
    }

    void Fun_b_kkt(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu,
                  Eigen::Map<const Vec>*& mat_ptr) {
        setInput(b_kkt, 0, x);
        setInput(b_kkt, 1, y);
        setInput(b_kkt, 2, z);
        setInput(b_kkt, 3, s);
        setInput(b_kkt, 4, mu);
        // auto t1 = high_resolution_clock::now();  // 起始时间
        b_kkt.eval();
        // auto t2 = high_resolution_clock::now();  // 结束时间
        // auto duration = duration_cast<microseconds>(t2 - t1).count();
        // std::cout << "[Casadi] bkkt assign value (v2) took " << duration << " µs" << std::endl;
        new (mat_ptr) Eigen::Map<const Vec>(b_kkt.output_ptr(0), b_kkt.output_size(0));
    }


    // Constraint residuals (h, g)
    std::pair<Vec,Vec> Fun_hg(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu) {
        
        setInput(hg, 0, x);
        setInput(hg, 1, y);
        setInput(hg, 2, z);
        setInput(hg, 3, s);
        setInput(hg, 4, mu);
        hg.eval();
        // int nh = hg.output_size(0);
        // int ng = hg.output_size(1);
        Vec h = Eigen::Map<const Vec>(hg.output_ptr(0), hg.output_size(0));
        Vec g = Eigen::Map<const Vec>(hg.output_ptr(1), hg.output_size(1));
        return {h, g};
    }

    void Fun_hg(const Vec& x, const Vec& y, const Vec& z, const Vec& s, const Vec& mu,
                Eigen::Map<const Vec>*& h_ptr, Eigen::Map<const Vec>*& g_ptr) {
        // auto t1 = high_resolution_clock::now();  // 起始时间
        setInput(hg, 0, x);
        setInput(hg, 1, y);
        setInput(hg, 2, z);
        setInput(hg, 3, s);
        setInput(hg, 4, mu);
        hg.eval();
        new (h_ptr) Eigen::Map<const Vec>(hg.output_ptr(0), hg.output_size(0));
        new (g_ptr) Eigen::Map<const Vec>(hg.output_ptr(1), hg.output_size(1));
        // auto t2 = high_resolution_clock::now();  // 结束时间
        // auto duration = duration_cast<microseconds>(t2 - t1).count();
        // std::cout << "[Casadi] [g, h]] assign value (v2) took " << duration << " µs" << std::endl;

    }



    // Retraction: sol = [dx; dlam; dmu; ds]
    void Fun_retract(
        const Vec& x, const Vec& y,
        const Vec& z, const Vec& s,
        const Vec& sol,
        Vec& x_new, Vec& y_new,
        Vec& z_new, Vec& s_new
    ) {
        setInput(Retract, 0, x);
        setInput(Retract, 1, y);
        setInput(Retract, 2, z);
        setInput(Retract, 3, s);
        setInput(Retract, 4, sol);
        Retract.eval();
        x_new = Eigen::Map<const Vec>(Retract.output_ptr(0), Retract.output_size(0));
        y_new = Eigen::Map<const Vec>(Retract.output_ptr(1), Retract.output_size(1));
        z_new = Eigen::Map<const Vec>(Retract.output_ptr(2), Retract.output_size(2));
        s_new = Eigen::Map<const Vec>(Retract.output_ptr(3), Retract.output_size(3));
    }

    static void setInput(CasadiFunctionWrapper &w, int idx, const Vec &v) {
        w.set_input(idx, std::vector<casadi_real>(v.data(), v.data() + v.size()));
    }
};

// /**
// * @brief Set any input by index, including mu as Vec
// * @param w   wrapper
// * @param idx input index (0 <= idx < w.n_in())
// * @param v   vector to copy (size must match w.input_size(idx))
// */
// if (idx < 0 || idx >= w.n_in())
//     throw std::out_of_range("setInput: idx out of range");
// int expected = w.input_size(idx);
// if (v.size() != expected)
//     throw std::invalid_argument(
//         "setInput: size mismatch (" + std::to_string(v.size()) +
//         " vs " + std::to_string(expected) + ")"
//     );
// double *ptr = w.input_ptr(idx);
// for (int i = 0; i < expected; ++i) ptr[i] = v[i];
