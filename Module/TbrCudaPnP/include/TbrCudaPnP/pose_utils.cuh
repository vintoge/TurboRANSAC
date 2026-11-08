#ifndef __POSE_UTILS_CUH__
#define __POSE_UTILS_CUH__

#include <TbrCudaPnP/config.h>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> rotmat_to_quat(const Eigen::Matrix<double, 3, 3> &R)
    {
        Eigen::Matrix<double, 4, 1> q;

        double trace = R(0, 0) + R(1, 1) + R(2, 2);

        if (trace > 0.0)
        {
            double s = 0.5 / sqrt(trace + 1.0);
            q(0) = 0.25 / s;
            q(1) = (R(2, 1) - R(1, 2)) * s;
            q(2) = (R(0, 2) - R(2, 0)) * s;
            q(3) = (R(1, 0) - R(0, 1)) * s;
        }
        else
        {
            if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2))
            {
                double s = 2.0 * sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
                q(0) = (R(2, 1) - R(1, 2)) / s;
                q(1) = 0.25 * s;
                q(2) = (R(0, 1) + R(1, 0)) / s;
                q(3) = (R(0, 2) + R(2, 0)) / s;
            }
            else if (R(1, 1) > R(2, 2))
            {
                double s = 2.0 * sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
                q(0) = (R(0, 2) - R(2, 0)) / s;
                q(1) = (R(0, 1) + R(1, 0)) / s;
                q(2) = 0.25 * s;
                q(3) = (R(1, 2) + R(2, 1)) / s;
            }
            else
            {
                double s = 2.0 * sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
                q(0) = (R(1, 0) - R(0, 1)) / s;
                q(1) = (R(0, 2) + R(2, 0)) / s;
                q(2) = (R(1, 2) + R(2, 1)) / s;
                q(3) = 0.25 * s;
            }
        }

        // normalize to ensure unit quaternion
        q.normalize();
        return q;
    }

    __host__ __device__ inline Eigen::Matrix<double, 3, 3> quat_to_rotmat(const Eigen::Matrix<double, 4, 1> &q)
    {
        const double w = q(0);
        const double x = q(1);
        const double y = q(2);
        const double z = q(3);

        Eigen::Matrix<double, 3, 3> R;

        R(0, 0) = 1.0 - 2.0 * (y * y + z * z);
        R(0, 1) = 2.0 * (x * y - z * w);
        R(0, 2) = 2.0 * (x * z + y * w);

        R(1, 0) = 2.0 * (x * y + z * w);
        R(1, 1) = 1.0 - 2.0 * (x * x + z * z);
        R(1, 2) = 2.0 * (y * z - x * w);

        R(2, 0) = 2.0 * (x * z - y * w);
        R(2, 1) = 2.0 * (y * z + x * w);
        R(2, 2) = 1.0 - 2.0 * (x * x + y * y);

        return R;
    }

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> quat_multiply(
        const Eigen::Matrix<double, 4, 1> &qa,
        const Eigen::Matrix<double, 4, 1> &qb)
    {
        const double qa1 = qa(0);
        const double qa2 = qa(1);
        const double qa3 = qa(2);
        const double qa4 = qa(3);

        const double qb1 = qb(0);
        const double qb2 = qb(1);
        const double qb3 = qb(2);
        const double qb4 = qb(3);

        Eigen::Matrix<double, 4, 1> result;
        result(0) = qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4;
        result(1) = qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3;
        result(2) = qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2;
        result(3) = qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1;

        return result;
    }

    __host__ __device__ inline Eigen::Matrix<double, 3, 1> quat_rotate(
        const Eigen::Matrix<double, 4, 1> &q,
        const Eigen::Matrix<double, 3, 1> &p)
    {
        const double q1 = q(0);
        const double q2 = q(1);
        const double q3 = q(2);
        const double q4 = q(3);

        const double p1 = p(0);
        const double p2 = p(1);
        const double p3 = p(2);

        const double px1 = -p1 * q2 - p2 * q3 - p3 * q4;
        const double px2 = p1 * q1 - p2 * q4 + p3 * q3;
        const double px3 = p2 * q1 + p1 * q4 - p3 * q2;
        const double px4 = p2 * q2 - p1 * q3 + p3 * q1;

        Eigen::Matrix<double, 3, 1> result;
        result(0) = px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3;
        result(1) = px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2;
        result(2) = px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1;

        return result;
    }

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> quat_conj(const Eigen::Matrix<double, 4, 1> &q)
    {
        return Eigen::Matrix<double, 4, 1>(q(0), -q(1), -q(2), -q(3));
    }

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> quat_exp(
        const Eigen::Matrix<double, 3, 1> &w)
    {
        const double theta2 = w.squaredNorm();
        const double theta = sqrt(theta2);
        const double theta_half = 0.5 * theta;

        double re, im;

        if (theta > 1e-6)
        {
            re = cos(theta_half);
            im = sin(theta_half) / theta;
        }
        else
        {
            // we are close to zero, use taylor expansion to avoid problems
            // with zero divisors in sin(theta/2)/theta
            const double theta4 = theta2 * theta2;
            re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
            im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

            // for the linearized part we re-normalize to ensure unit length
            // here s should be roughly 1.0 anyways, so no problem with zero div
            const double s = sqrt(re * re + im * im * theta2);
            re /= s;
            im /= s;
        }

        Eigen::Matrix<double, 4, 1> result;
        result(0) = re;
        result(1) = im * w(0);
        result(2) = im * w(1);
        result(3) = im * w(2);

        return result;
    }

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> quat_step_pre(
        const Eigen::Matrix<double, 4, 1> &q,
        const Eigen::Matrix<double, 3, 1> &w_delta)
    {
        return quat_multiply(quat_exp(w_delta), q);
    }

    __host__ __device__ inline Eigen::Matrix<double, 4, 1> quat_step_post(
        const Eigen::Matrix<double, 4, 1> &q,
        const Eigen::Matrix<double, 3, 1> &w_delta)
    {
        return quat_multiply(q, quat_exp(w_delta));
    }

} // namespace TbrCudaPnP

#endif
