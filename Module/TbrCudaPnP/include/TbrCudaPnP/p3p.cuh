#ifndef __P3P_CUH__
#define __P3P_CUH__

#include <TbrCudaPnP/p3p_utils.cuh>

namespace TbrCudaPnP
{

    template <typename T>
    __host__ __device__ int p3p(const Eigen::Matrix<T, 3, 1> *x_copy,
                                const Eigen::Matrix<T, 3, 1> *X_copy,
                                Eigen::Matrix<T, 4, 4> *output)
    {
        if (output == nullptr)
            return 0;

        // output->clear();
        // output->reserve(4);

        // Compute side vectors
        Eigen::Matrix<T, 3, 1> X01 = X_copy[0] - X_copy[1];
        Eigen::Matrix<T, 3, 1> X02 = X_copy[0] - X_copy[2];
        Eigen::Matrix<T, 3, 1> X12 = X_copy[1] - X_copy[2];
        // printf("X01_00 is:%f, X01_01 is:%f, X01_02 is:%f \n", X01(0), X01(1), X01(2));

        T a01 = X01.squaredNorm();
        T a02 = X02.squaredNorm();
        T a12 = X12.squaredNorm();
        // printf("a01 is:%f, a02 is:%f, a12 is:%f \n", a01, a02, a12);

        Eigen::Matrix<T, 3, 1> X[3] = {X_copy[0], X_copy[1], X_copy[2]};
        Eigen::Matrix<T, 3, 1> x[3] = {x_copy[0], x_copy[1], x_copy[2]};

        // Switch X,x so that BC is the largest distance among {X01, X02, X12}
        if (a01 > a02)
        {
            if (a01 > a12)
            {
                swap_vec3(x[0], x[2]);
                swap_vec3(X[0], X[2]);
                swap_scalar(a01, a12);
                X01 = -X12;
                X02 = -X02;
            }
        }
        else if (a02 > a12)
        {
            swap_vec3(x[0], x[1]);
            swap_vec3(X[0], X[1]);
            swap_scalar(a02, a12);
            X01 = -X01;
            X02 = X12;
        }
        // printf("X01_00 is:%f, X01_01 is:%f, X01_02 is:%f \n", X01(0), X01(1), X01(2));

        const T a12_inv = T(1.0) / a12;
        const T a = a01 * a12_inv;
        const T b = a02 * a12_inv;

        const T m01 = x[0].dot(x[1]);
        const T m02 = x[0].dot(x[2]);
        const T m12 = x[1].dot(x[2]);

        const T m12sq = -m12 * m12 + T(1.0);
        const T m02sq = -T(1.0) + m02 * m02;
        const T m01sq = -T(1.0) + m01 * m01;
        const T ab = a * b;
        const T bsq = b * b;
        const T asq = a * a;
        const T m013 = -T(2.0) + T(2.0) * m01 * m02 * m12;
        const T bsqm12sq = bsq * m12sq;
        const T asqm12sq = asq * m12sq;
        const T abm12sq = T(2.0) * ab * m12sq;

        const T k3_inv = T(1.0) / (bsqm12sq + b * m02sq);
        const T k2 = k3_inv * ((-T(1.0) + a) * m02sq + abm12sq + bsqm12sq + b * m013);
        const T k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-T(1.0) + b) * m01sq);
        const T k0 = k3_inv * (asqm12sq + a * m01sq);

        T s;
        // printf("k2 is:%f, k1 is:%f, k0 is:%f \n", k2, k1, k0);
        bool G = solve_cubic_single_real_cuda(k2, k1, k0, s);
        // printf("G: %d\n", G);
        // printf("s is:%f\n", s);

        Eigen::Matrix<T, 3, 3> C;
        C(0, 0) = -a + s * (T(1.0) - b);
        C(0, 1) = -m02 * s;
        C(0, 2) = a * m12 + b * m12 * s;
        C(1, 0) = C(0, 1);
        C(1, 1) = s + T(1.0);
        C(1, 2) = -m01;
        C(2, 0) = C(0, 2);
        C(2, 1) = C(1, 2);
        C(2, 2) = -a - b * s + T(1.0);


        Eigen::Matrix<T, 3, 1> pq[2]; 
        compute_pq_cuda(C, pq[0], pq[1]);

        // printf("pq00: %f, pq01: %f, pq02: %f\n", pq[0](0), pq[0](1), pq[0](2));
        // printf("pq10: %f, pq11: %f, pq12: %f\n", pq[1](0), pq[1](1), pq[1](2));

        T d0, d1, d2;

        Eigen::Matrix<T, 3, 3> XX;
        XX.col(0) = X01;
        XX.col(1) = X02;
        XX.col(2) = cross3(X01, X02);
        XX = inverse3x3(XX);
        // printf("XX00 is:%f, XX01 is:%f, XX02 is:%f \n", XX(0, 0), XX(0, 1), XX(0, 2));

        Eigen::Matrix<T, 3, 1> v1, v2;
        Eigen::Matrix<T, 3, 3> YY;

        int n_sols = 0;

        for (int i = 0; i < 2; ++i)
        {
            // [p0 p1 p2] * [1; x; y] = 0, or [p0 p1 p2] * [d2; d0; d1] = 0
            T p0 = pq[i](0);
            T p1 = pq[i](1);
            T p2 = pq[i](2);
            // here we run into trouble if p0 is zero,
            // so depending on which is larger, we solve for either d0 or d1
            // The case p0 = p1 = 0 is degenerate and can be ignored
            bool switch_12 = fabs(p0) <= fabs(p1);
            // printf("p0: %f, p1: %f, p2: %f\n", p0, p1, p2);
            // printf("switch_12: %d\n", switch_12);
            
            if (switch_12)
            {
                // eliminate d0
                T w0 = -p0 / p1;
                T w1 = -p2 / p1;
                T ca = T(1.0) / (w1 * w1 - b);
                T cb = T(2.0) * (b * m12 - m02 * w1 + w0 * w1) * ca;
                T cc = (w0 * w0 - T(2.0) * m02 * w0 - b + T(1.0)) * ca;
                T taus[2];
                if (!root2real_cuda(cb, cc, taus[0], taus[1]))
                    continue;
                for (T tau : taus)
                {
                    if (tau <= 0)
                        continue;
                    // positive only
                    d2 = sqrt(a12 / (tau * (tau - T(2.0) * m12) + T(1.0)));
                    d1 = tau * d2;
                    d0 = (w0 * d2 + w1 * d1);
                    // printf("d0 is:%f, d1 is:%f, d2 is:%f \n", d0, d1, d2);
                    if (d0 < 0)
                        continue;

                    refine_lambda_cuda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                    v1 = d0 * x[0] - d1 * x[1];
                    v2 = d0 * x[0] - d2 * x[2];

                    YY.col(0) = v1;
                    YY.col(1) = v2;
                    YY.col(2) = cross3(v1, v2);
                    // printf("YY00 is:%f, YY01 is:%f, YY02 is:%f \n", YY(0, 0), YY(0, 1), YY(0, 2));
                    Eigen::Matrix<T, 3, 3> R = matmul3x3(YY, XX);
                    // printf("R00 is:%f, R01 is:%f, R02 is:%f \n", R(1, 0), R(1, 1), R(1, 2));
                    Eigen::Matrix<T, 4, 4> pose = Eigen::Matrix<T, 4, 4>::Identity();

                    pose.template block<3, 3>(0, 0) = R;
                    pose.template block<3, 1>(0, 3) = matmul3x1(d0, x[0]) - matmul3x1(R, X[0]);

                    // pose.block(0, 0, 3, 3) = R;
                    // pose.block(0, 3, 3, 1) = matmul3x1(d0, x[0]) - matmul3x1(R, X[0]);
                    
                    output[n_sols++] = pose;
                }
            }
            else
            {
                T w0 = -p1 / p0;
                T w1 = -p2 / p0;
                T ca = T(1.0) / (-a * w1 * w1 + T(2.0) * a * m12 * w1 - a + T(1.0));
                T cb = T(2.0) * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
                T cc = (T(1.0) - a * w0 * w0) * ca;

                T taus[2];
                if (!root2real_cuda(cb, cc, taus[0], taus[1]))
                    continue;
                for (T tau : taus)
                {
                    if (tau <= T(0.0))
                        continue;
                    d0 = sqrt(a01 / (tau * (tau - T(2.0) * m01) + T(1.0)));
                    d1 = tau * d0;
                    d2 = w0 * d0 + w1 * d1;
                    if (d2 < T(0.0))
                        continue;

                    refine_lambda_cuda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                    v1 = d0 * x[0] - d1 * x[1];
                    v2 = d0 * x[0] - d2 * x[2];
                    YY.col(0) = v1;
                    YY.col(1) = v2;
                    YY.col(2) = cross3(v1, v2);
                    Eigen::Matrix<T, 3, 3> R = matmul3x3(YY, XX);

                    Eigen::Matrix<T, 4, 4> pose = Eigen::Matrix<T, 4, 4>::Identity();

                    pose.template block<3, 3>(0, 0) = R;
                    pose.template block<3, 1>(0, 3) = matmul3x1(d0, x[0]) - matmul3x1(R, X[0]);

                    // pose.block(0, 0, 3, 3) = R;
                    // pose.block(0, 3, 3, 1) = matmul3x1(d0, x[0]) - matmul3x1(R, X[0]);

                    output[n_sols++] = pose;
                }
            }

            if (n_sols > 0 && G)
                break;

        }

        return n_sols;
    }

} // namespace TbrCudaPnP

#endif
