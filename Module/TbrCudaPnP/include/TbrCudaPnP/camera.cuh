#ifndef __CAMERA_CUH__
#define __CAMERA_CUH__

#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    class Camera_cuda
    {
    public:
        int model_id;
        int width;
        int height;
        double *params;

        // Default constructor
        __host__ __device__ Camera_cuda()
            : model_id(1), width(-1), height(-1), params(nullptr) {} // "model_id(1)" for PINHOLE

        // Device-safe projection
        __host__ __device__ void project(
            const Eigen::Matrix<double, 2, 1> &x,
            Eigen::Matrix<double, 2, 1> *xp) const
        {
            // Example: trivial pinhole projection
            double fx = params[0];
            double fy = params[1];
            double cx = params[2];
            double cy = params[3];

            (*xp)(0) = fx * x(0) + cx;
            (*xp)(1) = fy * x(1) + cy;
        }

        __host__ __device__ void project_with_jac(
            const Eigen::Matrix<double, 2, 1> &x,
            Eigen::Matrix<double, 2, 1> *xp,
            Eigen::Matrix<double, 2, 2> *jac) const
        {
            // Example: same simple pinhole model
            project(x, xp);

            (*jac)(0, 0) = params[0];
            (*jac)(0, 1) = 0.0;
            (*jac)(1, 0) = 0.0;
            (*jac)(1, 1) = params[1];
        }

        __host__ __device__ void unproject(
            const Eigen::Matrix<double, 2, 1> &xp,
            Eigen::Matrix<double, 2, 1> *x) const
        {
            double fx = params[0];
            double fy = params[1];
            double cx = params[2];
            double cy = params[3];

            (*x)(0) = (xp(0) - cx) / fx;
            (*x)(1) = (xp(1) - cy) / fy;
        }

        __host__ __device__ double focal() const
        {
            return 0.5 * (params[0] + params[1]);
        }

        __host__ __device__ double focal_x() const { return params[0]; }
        __host__ __device__ double focal_y() const { return params[1]; }

        __host__ __device__ Eigen::Matrix<double, 2, 1> principal_point() const
        {
            return Eigen::Matrix<double, 2, 1>(params[2], params[3]);
        }

        // Host-only rescale
        void rescale(double scale)
        {
            params[0] *= scale; // fx
            params[1] *= scale; // fy
            params[2] *= scale; // cx
            params[3] *= scale; // cy
        }
    };

} // namespace TbrCudaPnP

#endif