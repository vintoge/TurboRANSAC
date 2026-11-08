#ifndef __LM_UTILS_CUH__
#define __LM_UTILS_CUH__

#include <TbrCudaPnP/config.h>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    // Manual Cholesky for NxN SPD matrix A
    template <typename T, int N>
    __forceinline__ __device__ void choleskyDecompose(
        const Eigen::Matrix<T, N, N> &A,
        Eigen::Matrix<T, N, N> &L)
    {
        L.setZero();
        constexpr T kEpsilon = static_cast<T>(1e-15);

        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            #pragma unroll
            for (int j = 0; j <= i; j++)
            {
                T sum = static_cast<T>(0);
                #pragma unroll
                for (int k = 0; k < j; k++)
                    sum += L(i, k) * L(j, k);

                if (i == j)
                {
                    T val = A(i, i) - sum;
                    if (val <= static_cast<T>(0))
                    {
                        val = kEpsilon;
                    }
                    L(i, j) = sqrt(val);
                }
                else
                {
                    L(i, j) = (A(i, j) - sum) / L(j, j);
                }
            }
        }
    }

    // Forward substitution: L * y = b
    template <typename T, int N>
    __forceinline__ __device__ void forwardSubstitution(
        const Eigen::Matrix<T, N, N> &L,
        const Eigen::Matrix<T, N, 1> &b,
        Eigen::Matrix<T, N, 1> &y)
    {
        #pragma unroll
        for (int i = 0; i < N; i++)
        {
            T sum = static_cast<T>(0);
            #pragma unroll
            for (int j = 0; j < i; j++)
                sum += L(i, j) * y(j);
            y(i) = (b(i) - sum) / L(i, i);
        }
    }

    // Backward substitution: L^T * x = y
    template <typename T, int N>
    __forceinline__ __device__ void backwardSubstitution(
        const Eigen::Matrix<T, N, N> &L,
        const Eigen::Matrix<T, N, 1> &y,
        Eigen::Matrix<T, N, 1> &x)
    {
        #pragma unroll
        for (int i = N - 1; i >= 0; i--)
        {
            T sum = static_cast<T>(0);
            #pragma unroll
            for (int j = i + 1; j < N; j++)
                sum += L(j, i) * x(j); // L^T(i,j) = L(j,i)
            x(i) = (y(i) - sum) / L(i, i);
        }
    }

    // Solve A x = b using Cholesky
    template <typename T, int N>
    __device__ void solve_cholesky(
        const Eigen::Matrix<T, N, N> &A,
        const Eigen::Matrix<T, N, 1> &b,
        Eigen::Matrix<T, N, 1> &x)
    {
        Eigen::Matrix<T, N, N> L;
        Eigen::Matrix<T, N, 1> y;

        // L.setZero();
        choleskyDecompose<T, N>(A, L);
        forwardSubstitution<T, N>(L, b, y);
        backwardSubstitution<T, N>(L, y, x);
    }


 

} // namespace TbrCudaPnP

#endif
