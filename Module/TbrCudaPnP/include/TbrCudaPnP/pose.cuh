#ifndef __POSE_CUH__
#define __POSE_CUH__

#include <TbrCudaPnP/pose_utils.cuh>
#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    // alignas(32)
    struct CameraPose_cuda
    {
        // Unit quaternion [w, x, y, z]
        Eigen::Matrix<double, 4, 1> q;
        Eigen::Matrix<double, 3, 1> t;

        // Default: identity
        __host__ __device__ CameraPose_cuda()
            : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}

        __host__ __device__ CameraPose_cuda(const Eigen::Matrix<double, 4, 1> &qq,
                                            const Eigen::Matrix<double, 3, 1> &tt)
            : q(qq), t(tt) {}

        __host__ __device__ CameraPose_cuda(const Eigen::Matrix<double, 3, 3> &R,
                                            const Eigen::Matrix<double, 3, 1> &tt)
            : q(rotmat_to_quat(R)), t(tt) {}

        __host__ __device__ inline Eigen::Matrix<double, 3, 3> R() const
        {
            return quat_to_rotmat(q);
        }

        __host__ __device__ inline Eigen::Matrix<double, 3, 4> Rt() const
        {
            // Eigen::Matrix<double, 3, 4> tmp;
            // tmp.template block<3, 3>(0, 0) = quat_to_rotmat(q);
            // tmp.col(3) = t;

            Eigen::Matrix<double, 3, 4> tmp;
            tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
            tmp.col(3) = t;

            return tmp;
        }

        __host__ __device__ inline Eigen::Matrix<double, 4, 4> Rt_homo() const
        {
            // Eigen::Matrix<double, 4, 4> tmp = Eigen::Matrix<double, 4, 4>::Identity();
            // tmp.template block<3, 3>(0, 0) = quat_to_rotmat(q);
            // tmp.template block<3, 1>(0, 3) = t;

            Eigen::Matrix<double, 4, 4> tmp = Eigen::Matrix<double, 4, 4>::Identity();
            tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
            tmp.block<3, 1>(0, 3) = t;

            return tmp;
        }

        __host__ __device__ inline Eigen::Matrix<double, 3, 1> rotate(const Eigen::Matrix<double, 3, 1> &p) const
        {
            return quat_rotate(q, p);
        }

        __host__ __device__ inline Eigen::Matrix<double, 3, 1> derotate(const Eigen::Matrix<double, 3, 1> &p) const
        {
            return quat_rotate(quat_conj(q), p);
        }

        __host__ __device__ inline Eigen::Matrix<double, 3, 1> apply(const Eigen::Matrix<double, 3, 1> &p) const
        {
            return rotate(p) + t;
        }

        __host__ __device__ inline Eigen::Matrix<double, 3, 1> center() const
        {
            return -derotate(t);
        }
    };


} // namespace TbrCudaPnP

#endif
