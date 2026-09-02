#pragma once

#include <cmath>
#include <sycl/sycl.hpp>

namespace geochem {

constexpr int MAX_ELEMENTS = 16;
constexpr int MAX_SPECIES = 64;
constexpr int MAX_MINERALS = 8;
constexpr int MAX_ALL_PHASES = 64;
constexpr int MAX_KINETICS = 8;
constexpr int MAX_RATE_PARAMS = 8;
constexpr int MAX_SYS_DIM = MAX_ELEMENTS + MAX_MINERALS; // 24

template <typename T, int Rows, int Cols>
struct FixedMatrix {
    T data[Rows * Cols] = {0};

    inline T& operator()(int r, int c) {
        return data[r * Cols + c];
    }

    inline const T& operator()(int r, int c) const {
        return data[r * Cols + c];
    }

    inline void fill(T val) {
        for (int i = 0; i < Rows * Cols; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, int N>
struct FixedVector {
    T data[N] = {0};

    inline T& operator[](int i) {
        return data[i];
    }

    inline const T& operator[](int i) const {
        return data[i];
    }

    inline void fill(T val) {
        for (int i = 0; i < N; ++i) {
            data[i] = val;
        }
    }
};

inline double norm_device(int n, const FixedVector<double, MAX_SYS_DIM>& v) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += v[i] * v[i];
    }
    return sycl::sqrt(s);
}

inline double debye_huckel_gamma_device(double z, double I, double a0, double b0) {
    if (z == 0.0) {
        return 1.0;
    }
    constexpr double A = 0.51;
    constexpr double B = 0.33;
    const double sqrt_I = sycl::sqrt(I);
    double log_gamma = 0.0;

    if (a0 > 0.0) {
        log_gamma = -A * (z * z) * sqrt_I / (1.0 + B * a0 * sqrt_I) + b0 * I;
    } else {
        // Davies equation
        log_gamma = -A * (z * z) * (sqrt_I / (1.0 + sqrt_I) - 0.3 * I);
    }

    if (log_gamma > 2.0) log_gamma = 2.0;
    else if (log_gamma < -2.0) log_gamma = -2.0;

    return sycl::pow(10.0, log_gamma);
}

// Gaussian elimination with partial pivoting in fixed register array
inline bool solve_linear_system_device(int n,
                                       FixedMatrix<double, MAX_SYS_DIM, MAX_SYS_DIM>& A,
                                       FixedVector<double, MAX_SYS_DIM>& b,
                                       FixedVector<double, MAX_SYS_DIM>& x) {
    // Forward elimination with partial row pivoting
    for (int i = 0; i < n; ++i) {
        int max_row = i;
        double max_val = sycl::fabs(A(i, i));
        for (int r = i + 1; r < n; ++r) {
            double val = sycl::fabs(A(r, i));
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        if (max_row != i) {
            for (int c = i; c < n; ++c) {
                double tmp = A(i, c);
                A(i, c) = A(max_row, c);
                A(max_row, c) = tmp;
            }
            double tmp = b[i];
            b[i] = b[max_row];
            b[max_row] = tmp;
        }

        double pivot = A(i, i);
        if (sycl::fabs(pivot) < 1e-18) {
            return false;
        }

        for (int r = i + 1; r < n; ++r) {
            double factor = A(r, i) / pivot;
            for (int c = i; c < n; ++c) {
                A(r, c) -= factor * A(i, c);
            }
            b[r] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        double s = 0.0;
        for (int c = i + 1; c < n; ++c) {
            s += A(i, c) * x[c];
        }
        x[i] = (b[i] - s) / A(i, i);
    }

    return true;
}

} // namespace geochem
