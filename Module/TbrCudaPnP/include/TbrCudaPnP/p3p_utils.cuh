#ifndef __P3P_UTILS_CUH__
#define __P3P_UTILS_CUH__

#include <TbrCudaPnP/config.h>
#include <Eigen/Dense>
#include <cmath>

namespace TbrCudaPnP
{

    template <int M, int N, int P, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, P>
        matmul(
            const Eigen::Matrix<T, M, N> &A,
            const Eigen::Matrix<T, N, P> &B)
    {
        Eigen::Matrix<T, M, P> C;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < P; ++j)
            {
                T sum = T(0);
                for (int k = 0; k < N; ++k)
                    sum += A(i, k) * B(k, j);
                C(i, j) = sum;
            }
        return C;
    }

    template <int M, int N, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, N>
        matmul(
            const T &scalar,
            const Eigen::Matrix<T, M, N> &A)
    {
        Eigen::Matrix<T, M, N> B;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                B(i, j) = scalar * A(i, j);
        return B;
    }

    template <int M, int N, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, N>
        matmul(
            const Eigen::Matrix<T, M, N> &A,
            const T &scalar)
    {
        return matmul<M, N>(scalar, A);
    }

    template <int M, int N, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, 1>
        matmul(
            const Eigen::Matrix<T, M, N> &A,
            const Eigen::Matrix<T, N, 1> &v)
    {
        Eigen::Matrix<T, M, 1> r;
        for (int i = 0; i < M; ++i)
        {
            T sum = T(0);
            for (int k = 0; k < N; ++k)
                sum += A(i, k) * v(k, 0);
            r(i, 0) = sum;
        }
        return r;
    }

    template <int M, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, 1>
        matmul(
            const T &scalar,
            const Eigen::Matrix<T, M, 1> &v)
    {
        Eigen::Matrix<T, M, 1> r;
        for (int i = 0; i < M; ++i)
            r(i, 0) = scalar * v(i, 0);
        return r;
    }

    template <int M, typename T>
    __forceinline__ __host__ __device__
        Eigen::Matrix<T, M, 1>
        matmul(
            const Eigen::Matrix<T, M, 1> &v,
            const T &scalar)
    {
        return matmul<M>(scalar, v);
    }

    //-----------------------------------------------------------------------------------------------------//

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 1> matmul3x1(const T &scalar,
                                                                         const Eigen::Matrix<T, 3, 1> &v)
    {
        Eigen::Matrix<T, 3, 1> result;
        for (int i = 0; i < 3; i++)
        {
            result(i, 0) = scalar * v(i, 0);
        }
        return result;
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 1> matmul3x1(const Eigen::Matrix<T, 3, 1> &v,
                                                                         const T &scalar)
    {
        return matmul3x1(scalar, v);
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 1> matmul3x1(const Eigen::Matrix<T, 3, 3> &A,
                                                                         const Eigen::Matrix<T, 3, 1> &v)
    {
        Eigen::Matrix<T, 3, 1> result;

        for (int i = 0; i < 3; i++)
        {
            T sum = T(0);
            for (int j = 0; j < 3; j++)
            {
                sum += A(i, j) * v(j, 0);
            }
            result(i, 0) = sum;
        }

        return result;
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 3> matmul3x3(const T &scalar,
                                                                         const Eigen::Matrix<T, 3, 3> &A)
    {
        Eigen::Matrix<T, 3, 3> C;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                C(i, j) = scalar * A(i, j);
            }
        }

        return C;
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 3> matmul3x3(const Eigen::Matrix<T, 3, 3> &A,
                                                                         const T &scalar)
    {
        return matmul3x3(scalar, A); // reuse existing function
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 3> matmul3x3(const Eigen::Matrix<T, 3, 3> &A,
                                                                         const Eigen::Matrix<T, 3, 3> &B)
    {
        Eigen::Matrix<T, 3, 3> C;

        for (int i = 0; i < 3; i++)
        { // row of A
            for (int j = 0; j < 3; j++)
            { // col of B
                T sum = T(0);
                for (int k = 0; k < 3; k++)
                {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }

        return C;
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 1> cross3(const Eigen::Matrix<T, 3, 1> &a,
                                                                      const Eigen::Matrix<T, 3, 1> &b)
    {
        Eigen::Matrix<T, 3, 1> c;
        c(0) = a(1) * b(2) - a(2) * b(1);
        c(1) = a(2) * b(0) - a(0) * b(2);
        c(2) = a(0) * b(1) - a(1) * b(0);
        return c;
    }

    template <typename T>
    __forceinline__ __host__ __device__ void compute_pq_cuda(const Eigen::Matrix<T, 3, 3> &C,
                                                             Eigen::Matrix<T, 3, 1> &p,
                                                             Eigen::Matrix<T, 3, 1> &q)
    {
        Eigen::Matrix<T, 3, 3> C_adj;

        C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
        C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
        C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
        C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
        C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
        C_adj(1, 0) = C_adj(0, 1);
        C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
        C_adj(2, 0) = C_adj(0, 2);
        C_adj(2, 1) = C_adj(1, 2);

        Eigen::Matrix<T, 3, 1> v;
        if (C_adj(0, 0) > C_adj(1, 1))
        {
            if (C_adj(0, 0) > C_adj(2, 2))
            {
                v = C_adj.col(0) / sqrt(C_adj(0, 0));
            }
            else
            {
                v = C_adj.col(2) / sqrt(C_adj(2, 2));
            }
        }
        else if (C_adj(1, 1) > C_adj(2, 2))
        {
            v = C_adj.col(1) / sqrt(C_adj(1, 1));
        }
        else
        {
            v = C_adj.col(2) / sqrt(C_adj(2, 2));
        }

        Eigen::Matrix<T, 3, 3> C_mod = C;

        C_mod(0, 1) -= v(2);
        C_mod(0, 2) += v(1);
        C_mod(1, 2) -= v(0);
        C_mod(1, 0) += v(2);
        C_mod(2, 0) -= v(1);
        C_mod(2, 1) += v(0);

        p = C_mod.col(0);
        q = C_mod.row(0).transpose(); // Important!
    }

    template <typename T>
    __forceinline__ __host__ __device__ void swap_vec3(Eigen::Matrix<T, 3, 1> &a, Eigen::Matrix<T, 3, 1> &b)
    {
        Eigen::Matrix<T, 3, 1> tmp = a;
        a = b;
        b = tmp;
    }

    template <typename T>
    __forceinline__ __host__ __device__ void swap_scalar(T &a, T &b)
    {
        T tmp = a;
        a = b;
        b = tmp;
    }

    template <typename T>
    __forceinline__ __host__ __device__ Eigen::Matrix<T, 3, 3> inverse3x3(const Eigen::Matrix<T, 3, 3> &m)
    {
        Eigen::Matrix<T, 3, 3> inv;

        T det =
            m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
            m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
            m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

        // printf("det: %f\n", (double)det);

        if (det == T(0))
        {
            return Eigen::Matrix<T, 3, 3>::Identity();
        }

        // if (fabs(det) < 1e-6)  // <-- threshold can be adjusted
        // {
        //     printf("Degenerate 3x3 matrix, det: %f\n", (float)det); // cast to float if T is float
        //     return Eigen::Matrix<T, 3, 3>::Identity(); // Or Zero() depending on your logic
        // }

        inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1));
        inv(0, 1) = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1));
        inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1));
        inv(1, 0) = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0));
        inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0));
        inv(1, 2) = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0));
        inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        inv(2, 1) = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0));
        inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));

        return inv / det;
    }

    template <typename T>
    __forceinline__ __host__ __device__ bool root2real_cuda(T b, T c, T &r1, T &r2)
    {
        const T THRESHOLD = static_cast<T>(eps_root2real); // In case T != real_t
        T v = b * b - T(4.0) * c;

        if (v < THRESHOLD)
        {
            r1 = r2 = T(-0.5) * b;
            return false;
        }

        if (v > THRESHOLD && v < T(0.0))
        {
            r1 = T(-0.5) * b;
            r2 = T(-2.0); // Sentinel
            return true;
        }

        T y = sqrt(v);
        if (b < T(0.0))
        {
            r1 = T(0.5) * (-b + y);
            r2 = T(0.5) * (-b - y);
        }
        else
        {
            r1 = T(2.0) * c / (-b + y);
            r2 = T(2.0) * c / (-b - y);
        }

        return true;
    }

    template <typename T>
    __forceinline__ __host__ __device__ void refine_lambda_cuda(T &lambda1, T &lambda2, T &lambda3,
                                                                const T a12, const T a13, const T a23,
                                                                const T b12, const T b13, const T b23)
    {
        for (int iter = 0; iter < 5; ++iter)
        {
            T l1 = lambda1;
            T l2 = lambda2;
            T l3 = lambda3;

            T r1 = l1 * l1 - T(2.0) * l1 * l2 * b12 + l2 * l2 - a12;
            T r2 = l1 * l1 - T(2.0) * l1 * l3 * b13 + l3 * l3 - a13;
            T r3 = l2 * l2 - T(2.0) * l2 * l3 * b23 + l3 * l3 - a23;

            if (fabs(r1) + fabs(r2) + fabs(r3) < static_cast<T>(eps_refine_lambda))
                return;

            T x11 = l1 - l2 * b12;
            T x12 = l2 - l1 * b12;
            T x21 = l1 - l3 * b13;
            T x23 = l3 - l1 * b13;
            T x32 = l2 - l3 * b23;
            T x33 = l3 - l2 * b23;

            T detJ = T(0.5) / (x11 * x23 * x32 + x12 * x21 * x33);

            lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
            lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
            lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
        }
    }

    // template <typename T>
    // __forceinline__ __host__ __device__ bool solve_cubic_single_real_cuda(T c2, T c1, T c0, T &root)
    // {
    //     T a = c1 - c2 * c2 / T(3.0);
    //     T b = (T(2.0) * c2 * c2 * c2 - T(9.0) * c2 * c1) / T(27.0) + c0;
    //     T c = b * b / T(4.0) + a * a * a / T(27.0);

    //     if (c != T(0))
    //     {
    //         if (c > T(0))
    //         {
    //             T sqrt_c = sqrt(c);
    //             T half_b = -T(0.5) * b;
    //             root = cbrt(half_b + sqrt_c) + cbrt(half_b - sqrt_c) - c2 / T(3.0);
    //             return true;
    //         }
    //         else
    //         {
    //             T sqrt_term = sqrt(-T(3.0) / a);
    //             T theta = acos(T(3.0) * b / (T(2.0) * a) * sqrt_term);
    //             root = T(2.0) * sqrt(-a / T(3.0)) * cos(theta / T(3.0)) - c2 / T(3.0);
    //         }
    //     }
    //     else
    //     {
    //         root = -c2 / T(3.0) + ((a != T(0)) ? (T(3.0) * b / a) : T(0));
    //     }

    //     return false;
    // }

    template <typename T>
    __forceinline__ __host__ __device__ bool solve_cubic_single_real_cuda(T c2, T c1, T c0, T &root)
    {
        T a = c1 - c2 * c2 / T(3.0);
        T b = (T(2.0) * c2 * c2 * c2 - T(9.0) * c2 * c1) / T(27.0) + c0;
        T c = b * b / T(4.0) + a * a * a / T(27.0);
        // printf("compare c is: %f\n", c);

        if (c != T(0))
        {
            if (c > T(0))
            {
                c = sqrt(c);
                b *= -T(0.5);
                root = cbrt(b + c) + cbrt(b - c) - c2 / T(3.0);
                return true;
            }
            else
            {
                c = T(3.0) * b / (T(2.0) * a) * sqrt(-T(3.0) / a);
                root = T(2.0) * sqrt(-a / T(3.0)) * cos(acos(c) / T(3.0)) - c2 / T(3.0);
            }
        }
        else
        {
            root = -c2 / T(3.0) + ((a != T(0)) ? (T(3.0) * b / a) : T(0));
        }

        return false;
    }



} // namespace TbrCudaPnP

#endif
