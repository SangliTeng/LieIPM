#pragma once

#include <Eigen/Sparse>
#include <LieIPMUtils.h>

// Enumeration for different solver types
enum class SolverType {
    BASE,
    EIGEN_LU,
    MUMPS
};

// Overload operator<< for SolverType
inline std::ostream& operator<<(std::ostream& os, const SolverType& type) {
    switch (type) {
        case SolverType::BASE:
            os << "BASE";
            break;
        case SolverType::EIGEN_LU:
            os << "EIGEN_LU";
            break;
        case SolverType::MUMPS:
            os << "MUMPS";
            break;
        default:
            os << "UNKNOWN";
            break;
    }
    return os;
}


// MUMPS traits template for handling different precision types
template<typename Scalar>
struct MUMPSTraits;

#include <dmumps_c.h>
#include <smumps_c.h>

// Specialization for double precision
template<>
struct MUMPSTraits<double> {
    using mumps_struct = DMUMPS_STRUC_C;
    static void mumps_call(mumps_struct* mumps) { dmumps_c(mumps); }
};

// Specialization for float precision  
template<>
struct MUMPSTraits<float> {
    using mumps_struct = SMUMPS_STRUC_C;
    static void mumps_call(mumps_struct* mumps) { smumps_c(mumps); }
};

// Generic linear solver wrapper interface (template)
template <typename Scalar = casadi_real, typename Index = long long>
class LinearSolverWrapper {
public:
    using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;
    using Vec       = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    virtual ~LinearSolverWrapper() = default;
    
    // virtual void analyzePattern(const Eigen::Map<const SparseMat>* A_ptr) = 0;
    virtual void analyzePattern(const void *A) = 0;
    // virtual void factorize(const SparseMat& A) = 0;
    virtual void factorize(const void *A) = 0;
    virtual void factorize() = 0;

    virtual Vec solve(const void *b) = 0;

    virtual void getInfo(void *info) = 0; // get what

    virtual SolverType getSolverType() const {
        return solver_type_;
    }
protected:
        SolverType solver_type_ = SolverType::BASE; // Default type
private:
};

// Eigen SparseLU solver wrapper
template <typename Scalar = casadi_real, typename Index = long long>
class EigenLUSolverWrapper : public LinearSolverWrapper<Scalar, Index> {
public:
    using typename LinearSolverWrapper<Scalar, Index>::SparseMat;
    using typename LinearSolverWrapper<Scalar, Index>::Vec;

    EigenLUSolverWrapper() : A_ptr_(nullptr) {this->solver_type_ = SolverType::EIGEN_LU;}

    void analyzePattern(const void *A_ptr) override {
        A_ptr_ = static_cast<const Eigen::Map<const SparseMat>*>(A_ptr);
        solver_.analyzePattern(*A_ptr_);
    }

    void factorize(const void *A) override {
        // solver_.factorize(static_cast<const Eigen::Map<const SparseMat>*>(A));
        solver_.factorize(*static_cast<const Eigen::Map<const SparseMat>*>(A));
    }

    void factorize() override {
        if (A_ptr_) {
            solver_.factorize(*A_ptr_);
        }
    }

    Vec solve(const void *b) override {
        return solver_.solve(*static_cast<const Vec*>(b));
    }

    void getInfo(void *info) override {
    }

private:
    Eigen::SparseLU<SparseMat> solver_;
    const Eigen::Map<const SparseMat>* A_ptr_;
    SolverType solver_type_ = SolverType::EIGEN_LU; // Default type
};

// MUMPS wrapper using traits
template <typename Scalar = casadi_real, typename Index = long long>
class MUMPSSolverWrapper : public LinearSolverWrapper<Scalar, Index> {
public:
    using typename LinearSolverWrapper<Scalar, Index>::SparseMat;
    using typename LinearSolverWrapper<Scalar, Index>::Vec;
    using MUMPSType = typename MUMPSTraits<Scalar>::mumps_struct;

    MUMPSSolverWrapper() {
        mumps_.job = -1;
        mumps_.par = 1;
        mumps_.sym = 2; // 0: 非对称/不定 
        mumps_.comm_fortran = -987654;
        MUMPSTraits<Scalar>::mumps_call(&mumps_);
        // 常用 MUMPS 控制参数（可根据需要调整）
        mumps_.icntl[0] = 6; // 输出级别（0:无输出, 1:错误, 2:警告, 3:全部）
        mumps_.icntl[1] = 2; // 输出流（6=stdout）
        mumps_.icntl[2] = 2; // 输出流（6=stdout, 0=无）
        mumps_.icntl[3] = 0; // 日志流（6=stdout, 0=无）

        // mumps_.icntl[12] = 1; // 
        // mumps_.icntl[13] = 1; // 日志流（6=stdout, 0=无）
        mumps_.icntl[18] = 0; // 矩阵是否分布式（0=否, 1=是）
        // mumps_.icntl[7] = 0.; 
        // mumps_.cntl[0] = 1.0; // 强行使用 static pivoting
        // mumps_.icntl[13] = 1;
        // mumps_.icntl[18] = 0;
        // mumps_.icntl[6] = 1;
        // id.ICNTL(13) = 1;  // Sequential root analysis
        // id.ICNTL(18) = 0;  // Matrix is not distributed
        // id.ICNTL(6)  = 1;  // Moderate verbosity
        this->solver_type_ = SolverType::MUMPS;
        // mumps_.infog = &infog_[0]; // 将 infog_ 的地址传递给 mumps_.infog
        // std::cout<<this->solver_type_<<std::endl;

        // APR. 20 tuning for better robustness
        // Template-A:
        // // Printing level: keep useful diagnostics
        // mumps_.icntl[3] = 2;      // ICNTL(4)=2

        // // Numerical pivoting threshold
        // mumps_.cntl[0] = 0.05;    // CNTL(1)=0.05

        // // Optional main statistics
        // mumps_.icntl[10] = 2;     // ICNTL(11)=2

        // // Null pivot detection
        // mumps_.icntl[23] = 1;     // ICNTL(24)=1

        // // Rank-revealing factorization
        // mumps_.icntl[55] = 1;     // ICNTL(56)=1

        // // IMPORTANT: do NOT enable static pivoting here
        // mumps_.cntl[3] = -1.0;    // CNTL(4)=-1, disabled

        // Template-B:
        // Printing level
        // mumps_.icntl[3] = 2;      // ICNTL(4)=2

        // // Stronger numerical pivoting than default auto
        // mumps_.cntl[0] = 0.05;    // CNTL(1)=0.05
        // // You can try 0.1 if still unstable

        // // Iterative refinement
        // mumps_.icntl[9]  = 2;     // ICNTL(10)=2

        // // Main error statistics
        // mumps_.icntl[10] = 2;     // ICNTL(11)=2

        // // Disable null-pivot detection / rank-revealing
        // mumps_.icntl[23] = 0;     // ICNTL(24)=0
        // mumps_.icntl[55] = 0;     // ICNTL(56)=0

        // // Enable static pivoting with automatic threshold
        // mumps_.cntl[3] = 0.0;     // CNTL(4)=0

        // Template-C:
        // mumps_.icntl[3]  = 2;   // ICNTL(4)=2, print diagnostics

        // // Make preprocessing actually useful for KKT / augmented systems
        // mumps_.icntl[27] = 1;   // ICNTL(28)=1, sequential analysis
        // mumps_.icntl[5]  = 5;   // ICNTL(6)=5, recommended for augmented systems
        // mumps_.icntl[7]  = 77;  // ICNTL(8)=77, automatic scaling

        // // Robust factorization diagnostics
        // mumps_.cntl[0]   = -1.0;  // CNTL(1)=automatic
        // mumps_.icntl[23] = 1;     // ICNTL(24)=1, null pivot detection
        // mumps_.icntl[55] = 1;     // ICNTL(56)=1, rank-revealing
        // mumps_.cntl[3]   = -1.0;  // CNTL(4) off; do not mix with static pivoting

        // mumps_.icntl[10] = 2;     // ICNTL(11)=2, main error stats

    }
    ~MUMPSSolverWrapper() { 
        mumps_.job = -2; 
        MUMPSTraits<Scalar>::mumps_call(&mumps_); 
    }


    void analyzePattern(const void *A_kkt_ptr) override {
        setupMatrix(static_cast<const Eigen::Map<const SparseMat>*>(A_kkt_ptr));
        mumps_.job = 1; // analysis
        MUMPSTraits<Scalar>::mumps_call(&mumps_);
    }

    void factorize(const void *A) override {
        setupMatrix(static_cast<const Eigen::Map<const SparseMat>*>(A));
        mumps_.job = 2;
        MUMPSTraits<Scalar>::mumps_call(&mumps_);
    }

    void factorize() override {
        mumps_.job = 2;

        // auto t0 = std::chrono::high_resolution_clock::now();
        MUMPSTraits<Scalar>::mumps_call(&mumps_);
        // auto t1 = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        // std::cout << "[MUMPS] factorization took " << duration << " µs" << std::endl;
    }

    Vec solve(const void *b) override {
        // Vec x = b;
        Vec x = *static_cast<const Vec*>(b);
        mumps_.rhs = x.data();
        mumps_.job = 3; // solve
        MUMPSTraits<Scalar>::mumps_call(&mumps_);
        return x;
    }

    void getInfo(void *info) override {
        if (info) {
            // mumps_.job = 6;
            // dmumps_c(&mumps_);
            auto* inertia_info = static_cast<InertiaCorrectionUtils*>(info);
            // inertia_info->n_plus  = mumps_.infog[12];
            inertia_info->n_minus    = mumps_.infog[11]; // compatible with IPOPT interface
            inertia_info->flag_zeros  = mumps_.infog[0]; // compatible with IPOPT interface
            // inertia_info->n_zero  = mumps_.infog[28];

            // std::cout << "[MUMPS - info]: "<<std::endl;
            // for (int i = 0; i < 80; ++i) {
            //     std::cout<< mumps_.info[i] << " ";
            // }

            // std::cout << "[MUMPS - infog]: "<<std::endl;
            // for (int i = 0; i < 80; ++i) {
            //     std::cout<< mumps_.infog[i] << " ";
            // }
            // std::cout << std::endl;
        }
    }


    void print_info() const {
        std::cout << "[DEBUG] mumps_.a address: " << static_cast<const void*>(mumps_.a) << std::endl;
    }

private:
    MUMPSType mumps_ = {0};
    std::vector<int> irn_data_;
    std::vector<int> jcn_data_;
    int infog_[80] = {0};

    void setupMatrix(const Eigen::Map<const SparseMat>* A_kkt_ptr) {
        // TODO: use the A_ptr 操作....
        // COO 格式，1-based 行列号
        mumps_.n = (*A_kkt_ptr).rows();
        mumps_.nz = (*A_kkt_ptr).nonZeros();
        irn_data_.resize((*A_kkt_ptr).nonZeros());
        jcn_data_.resize((*A_kkt_ptr).nonZeros());
        int idx = 0;
        for (int k = 0; k < (*A_kkt_ptr).outerSize(); ++k) {
            for (typename Eigen::Map<const SparseMat>::InnerIterator it((*A_kkt_ptr), k); it; ++it) {
                irn_data_[idx] = static_cast<int>(it.row()) + 1; // 1-based
                jcn_data_[idx] = static_cast<int>(it.col()) + 1; // 1-based
                ++idx;
            }
        }
        mumps_.irn = irn_data_.data();
        mumps_.jcn = jcn_data_.data();
        mumps_.a = const_cast<Scalar*>(A_kkt_ptr->valuePtr());
    }
};

// void setupMatrix(const SparseMat& A) {
//     // TODO: use the A_ptr 操作....

//     // COO 格式，1-based 行列号
//     mumps_.n = A.rows();
//     mumps_.nz = A.nonZeros();
//     irn_data_.resize(A.nonZeros());
//     jcn_data_.resize(A.nonZeros());
//     int idx = 0;
//     for (int k = 0; k < A.outerSize(); ++k) {
//         for (typename SparseMat::InnerIterator it(A, k); it; ++it) {
//             irn_data_[idx] = static_cast<int>(it.row()) + 1; // 1-based
//             jcn_data_[idx] = static_cast<int>(it.col()) + 1; // 1-based
//             ++idx;
//         }
//     }
//     mumps_.irn = irn_data_.data();
//     mumps_.jcn = jcn_data_.data();
//     mumps_.a = const_cast<double*>(A.valuePtr());
// }
