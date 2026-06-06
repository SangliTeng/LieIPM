#pragma once
#include <vector>
#include <string>
#include <Eigen/Sparse>

#include <chrono>
using namespace std::chrono;


// === CasADi 生成的类型定义 ===
#ifndef casadi_real
#define casadi_real double
#endif
#ifndef casadi_int
#define casadi_int long long int
#endif

using SparseMat = Eigen::SparseMatrix<casadi_real, Eigen::ColMajor, long long>;

class CasadiFunctionWrapper {
public:
    // 构造：所有 CasADi 导出接口都要传进来
    CasadiFunctionWrapper(
    void (*incref)(void),
    void (*decref)(void),
    casadi_int (*n_in)(void),
    casadi_int (*n_out)(void),
    const casadi_int* (*sparsity_in)(casadi_int),
    const casadi_int* (*sparsity_out)(casadi_int),
    int (*work)(casadi_int*, casadi_int*, casadi_int*, casadi_int*),
    int (*eval)(const casadi_real**, casadi_real**,
                casadi_int*, casadi_real*, int),
    const char* (*name_in)(casadi_int),
    const char* (*name_out)(casadi_int)
    ) : incref_(incref), decref_(decref),
        n_in_func_(n_in), n_out_func_(n_out),
        sparsity_in_(sparsity_in), sparsity_out_(sparsity_out),
        work_(work), eval_func_(eval),
        name_in_func_(name_in), name_out_func_(name_out)
    {
        incref_();
        n_in_  = static_cast<int>(n_in_func_());
        n_out_ = static_cast<int>(n_out_func_());
        in_names_.resize(n_in_);
        out_names_.resize(n_out_);
        in_size_.resize(n_in_);
        out_size_.resize(n_out_);
        inbuf_.resize(n_in_);
        outbuf_.resize(n_out_);
        in_ptr_.resize(n_in_);
        out_ptr_.resize(n_out_);

        // 解析输入
        for (int i = 0; i < n_in_; ++i) {
            in_names_[i] = name_in_func_(i);
            const casadi_int* sp = sparsity_in_(i);
            int rows = int(sp[0]), cols = int(sp[1]), dense = int(sp[2]);
            int sz = dense ? rows * cols : int(sp[2 + cols]);
            in_size_[i] = sz;
            inbuf_[i].resize(sz, 1.0);
            in_ptr_[i] = inbuf_[i].data();
        }
        // 解析输出
        for (int i = 0; i < n_out_; ++i) {
            out_names_[i] = name_out_func_(i);
            const casadi_int* sp = sparsity_out_(i);
            int rows = int(sp[0]), cols = int(sp[1]), dense = int(sp[2]);
            int sz = dense ? rows * cols : int(sp[2 + cols]);
            out_size_[i] = sz;
            outbuf_[i].resize(sz, 0.0);
            out_ptr_[i] = outbuf_[i].data();
        }
        casadi_int sz_arg, sz_res, sz_iw, sz_w;
        work_(&sz_arg, &sz_res, &sz_iw, &sz_w);
        iw_.resize(sz_iw);
        w_.resize(sz_w);

        // Initialize vectors for each output's sparsity pattern
        colptr_vec_.resize(n_out_);
        rowidx_vec_.resize(n_out_);
        for (int i = 0; i < n_out_; ++i) {
            const casadi_int* sp = sparsity_out_(i);
            int cols = int(sp[1]); 
            casadi_int nnz = sp[cols + 2];
            const casadi_int* colind = sp + 2;
            const casadi_int* row = sp + cols + 3;

            colptr_vec_[i].assign(colind, colind + cols + 1);
            rowidx_vec_[i].assign(row, row + nnz);
        }
    }

    ~CasadiFunctionWrapper() {decref_();};

    int n_in()  const { return n_in_; }
    int n_out() const { return n_out_; }
    std::string input_name(int idx)  const { return in_names_.at(idx); }
    std::string output_name(int idx) const { return out_names_.at(idx); }
    int input_size(int idx)  const { return in_size_.at(idx); }
    int output_size(int idx) const { return out_size_.at(idx); }

    casadi_real* input_ptr(int idx)  { return inbuf_[idx].data(); }
    casadi_real* output_ptr(int idx) { return outbuf_[idx].data(); }
    std::vector<casadi_real>& input(int idx)  { return inbuf_.at(idx); }
    std::vector<casadi_real>& output(int idx) { return outbuf_.at(idx); }

    int eval() {
    auto t_start = std::chrono::high_resolution_clock::now();
    int flag = eval_func_(in_ptr_.data(), out_ptr_.data(), iw_.data(), w_.data(), 0);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    return flag;
    };

    Eigen::SparseMatrix<casadi_real, Eigen::ColMajor, long long> get_output_sparse(int idx);

    void get_output_sparse(int idx, void* map_buffer) {
        const casadi_int* sp = sparsity_out_(idx);
        int rows = int(sp[0]), cols = int(sp[1]), pattern = int(sp[2]);
        if (pattern == 1) {
            // Dense
            // Eigen::Map<Eigen::MatrixXd> dense(outbuf_[idx].data(), rows, cols);
            throw std::runtime_error("Dense output not supported in pointer Map overload");
        } else {
            // auto t1 = high_resolution_clock::now();  // 起始时间
            const casadi_int* colind = sp + 2;
            casadi_int nnz = sp[cols + 2];
            const casadi_int* row = sp + cols + 3;
            new (map_buffer) Eigen::Map<const SparseMat>(rows, cols, nnz, colptr_vec_[idx].data(),
                                                                        rowidx_vec_[idx].data(),
                                                                        outbuf_[idx].data());
            Eigen::Map<const SparseMat>* spmat_ptr = reinterpret_cast<Eigen::Map<const SparseMat>*>(map_buffer);    }
    }

    void get_output_buffer(int idx, int &rows, int &cols, int &nnz,
                           const casadi_int* &colind, const casadi_int* &row,
                           casadi_real* &values) 
    {
        const casadi_int* sp = sparsity_out_(idx);
        rows = int(sp[0]);
        cols = int(sp[1]);
        
        int pattern = int(sp[2]);
        if (pattern == 1) {
            throw std::runtime_error("Dense output not supported in pointer Map overload");
        } else {
            nnz = sp[cols + 2];
            colind = sp + 2;
            row = sp + cols + 3;
            values = outbuf_[idx].data();
        }
    }


    int set_input(int idx, const std::vector<casadi_real>& value) {
        if (idx < 0 || idx >= n_in_) {
            throw std::out_of_range("Input index out of range");
        }
        if (value.size() != in_size_[idx]) {
            throw std::invalid_argument("Input size mismatch for index " + std::to_string(idx) +
                                        ": expected " + std::to_string(in_size_[idx]) +
                                        ", got " + std::to_string(value.size()));
        }
        inbuf_[idx] = value;
        in_ptr_[idx] = inbuf_[idx].data();
        return 0; // success
    }

private:

    // TSL: the data will be stored in these vectors. 
    // function pointer
    void (*incref_)(void);
    void (*decref_)(void);
    casadi_int (*n_in_func_)(void);
    casadi_int (*n_out_func_)(void);
    const casadi_int* (*sparsity_in_)(casadi_int);
    const casadi_int* (*sparsity_out_)(casadi_int);
    int (*work_)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
    int (*eval_func_)(const casadi_real** , casadi_real** ,
                      casadi_int*, casadi_real*, int);
    const char* (*name_in_func_)(casadi_int);
    const char* (*name_out_func_)(casadi_int);

    int n_in_ = 0, n_out_ = 0;
    std::vector<std::string> in_names_, out_names_;
    std::vector<int> in_size_, out_size_;
    std::vector<std::vector<casadi_real>> inbuf_, outbuf_;
    std::vector<const casadi_real*> in_ptr_;
    std::vector<casadi_real*> out_ptr_;
    std::vector<casadi_int> iw_;
    std::vector<casadi_real> w_;

    std::vector<std::vector<casadi_int>> colptr_vec_;
    std::vector<std::vector<casadi_int>> rowidx_vec_;
};

// CasadiFunctionWrapper::

// CasadiFunctionWrapper::~CasadiFunctionWrapper() {
//     decref_();
// }

// int CasadiFunctionWrapper::eval() 

// SparseMat CasadiFunctionWrapper::

// void CasadiFunctionWrapper::get_output_sparse

// void CasadiFunctionWrapper::