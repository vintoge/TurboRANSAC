#ifndef __JACO_CUH__
#define __JACO_CUH__

#include <TbrCudaPnP/robust_loss.cuh>
#include <TbrCudaPnP/jaco_utils.cuh>
#include <TbrCudaPnP/camera.cuh>
#include <TbrCudaPnP/pose.cuh>
#include <TbrCudaPnP/p3p_utils.cuh>

#include <TbrCudaPnP/msac.cuh>

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    class CameraJacobianAccumulator_cuda
    {
    public:
        using param_t = CameraPose_cuda;
        static constexpr size_t num_params = 6;

        const double *x;
        const double *X;
        const double *cam_k;
        int n;

        // no wrap divergence
        __device__ void residual(const BundleOptions_cuda &opt, const param_t &pose, double &cost_last) const
        {
            int tid = threadIdx.x;
            int stride = blockDim.x;

            UniformWeightVector_cuda weights;
            CauchyLoss_cuda loss_fn(opt.loss_scale);

            const double fx = cam_k[0];
            const double fy = cam_k[1];
            const double cx = cam_k[2];
            const double cy = cam_k[3];

            double local_cost = 0.0;

            for (size_t i = tid; i < n; i += stride)
            {
                Eigen::Matrix<double, 3, 1> X_cuda(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
                Eigen::Matrix<double, 2, 1> x_cuda(x[i * 2 + 0], x[i * 2 + 1]);

                const Eigen::Matrix<double, 3, 1> Z = pose.apply(X_cuda);
                const bool valid = Z(2) >= 0.0;
                const double inv_z = valid ? 1.0 / Z(2) : 0.0;
                Eigen::Matrix<double, 2, 1> p(Z(0) * inv_z, Z(1) * inv_z);

                p(0) = fx * p(0) + cx;
                p(1) = fy * p(1) + cy;

                const double r0 = p(0) - x_cuda(0);
                const double r1 = p(1) - x_cuda(1);
                const double r_squared = r0 * r0 + r1 * r1;

                const double contrib = valid ? weights[i] * loss_fn.loss(r_squared) : 0.0;

                local_cost += contrib;
            }

            // Block-wide reduction using warp shuffle
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                local_cost += __shfl_down_sync(0xffffffff, local_cost, offset);
            }

            // Warp reduction
            __shared__ double shared_cost[32]; // Max wraps for 1024 threads

            int lane = tid % warpSize;
            int warp_id = tid / warpSize;

            if (lane == 0)
                shared_cost[warp_id] = local_cost;

            __syncthreads();

            if (warp_id == 0)
            {
                int n_warps = (blockDim.x + warpSize - 1) / warpSize;
                double block_sum = (lane < n_warps) ? shared_cost[lane] : 0.0;

                for (int offset = warpSize / 2; offset > 0; offset /= 2)
                {
                    block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
                }

                if (lane == 0)
                {
                    cost_last = block_sum;
                    // printf("cost_last: %f\n", cost_last);
                }
            }

            __syncthreads();

        }

        __device__ void accumulate(const BundleOptions_cuda &opt,
                                   const param_t &pose,
                                   Eigen::Matrix<double, num_params, num_params> &JtJ,
                                   Eigen::Matrix<double, num_params, 1> &Jtr) const
        {
            int tid = threadIdx.x;
            int stride = blockDim.x;

            Eigen::Matrix<double, 3, 3> R = pose.R();
            Eigen::Matrix<double, 2, 2> Jcam;
            Jcam.setIdentity();

            const double fx = cam_k[0];
            const double fy = cam_k[1];
            const double cx = cam_k[2];
            const double cy = cam_k[3];

            UniformWeightVector_cuda weights;
            CauchyLoss_cuda loss_fn(opt.loss_scale);

            Eigen::Matrix<double, num_params, num_params> local_JtJ;
            Eigen::Matrix<double, num_params, 1> local_Jtr;

            local_JtJ.setZero();  // very import!!!
            local_Jtr.setZero();  // very import!!!

            for (size_t i = tid; i < n; i += stride)
            {
                Eigen::Matrix<double, 3, 1> X_cuda(X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]);
                Eigen::Matrix<double, 2, 1> x_cuda(x[i * 2 + 0], x[i * 2 + 1]);
                const Eigen::Matrix<double, 3, 1> Z = matmul3x1(R, X_cuda) + pose.t;
                const Eigen::Matrix<double, 2, 1> z = Z.hnormalized();

                // Project with intrinsics
                Eigen::Matrix<double, 2, 1> zp = z;
                zp(0) = fx * z(0) + cx;
                zp(1) = fy * z(1) + cy;

                Jcam(0, 0) = fx;
                Jcam(0, 1) = 0.0;
                Jcam(1, 0) = 0.0;
                Jcam(1, 1) = fy;

                Eigen::Matrix<double, 2, 1> r = zp - x_cuda;
                const double r_squared = r.squaredNorm();
                const double weight = weights[i] * loss_fn.weight(r_squared);

                double Z_valid = (Z(2) >= 0.0) ? 1.0 : 0.0;
                double W_valid = (weight != 0.0) ? 1.0 : 0.0;
                // double W_valid = (weight == 0) ? 0 : 1;
                double valid = Z_valid * W_valid;  // weight mask, Z > 0 mask
                const double w = valid * weight;

                // const double w = (Z(2) >= 0.0 && weight > 0.0) * weight;

                //--------------------------------------------------------------------------------//

                Eigen::Matrix<double, 2, 3> dZ;
                dZ.block<2, 2>(0, 0) = Jcam;
                dZ.col(2) = -matmul(Jcam, z);
                dZ = matmul((1.0 / Z(2)), dZ);
                dZ = matmul(dZ, R);

                const double X0 = X[i * 3 + 0];
                const double X1 = X[i * 3 + 1];
                const double X2 = X[i * 3 + 2];

                const double dZtdZ_0_0 = w * dZ.col(0).dot(dZ.col(0));
                const double dZtdZ_1_0 = w * dZ.col(1).dot(dZ.col(0));
                const double dZtdZ_1_1 = w * dZ.col(1).dot(dZ.col(1));
                const double dZtdZ_2_0 = w * dZ.col(2).dot(dZ.col(0));
                const double dZtdZ_2_1 = w * dZ.col(2).dot(dZ.col(1));
                const double dZtdZ_2_2 = w * dZ.col(2).dot(dZ.col(2));

                local_JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
                local_JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
                local_JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
                local_JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
                local_JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
                local_JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
                local_JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
                local_JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
                local_JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
                local_JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
                local_JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
                local_JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
                local_JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
                local_JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
                local_JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
                local_JtJ(3, 3) += dZtdZ_0_0;
                local_JtJ(4, 3) += dZtdZ_1_0;
                local_JtJ(5, 3) += dZtdZ_2_0;
                local_JtJ(4, 4) += dZtdZ_1_1;
                local_JtJ(5, 4) += dZtdZ_2_1;
                local_JtJ(5, 5) += dZtdZ_2_2;

                r *= w;

                local_Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
                local_Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
                local_Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
                local_Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
                local_Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
                local_Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
            }

            // Block-wide reduction using warp shuffle
            #pragma unroll
            for (int i = 0; i < num_params; ++i)
                #pragma unroll
                for (int j = 0; j < num_params; ++j)
                    for (int offset = warpSize/2; offset > 0; offset /= 2)
                        local_JtJ(i,j) += __shfl_down_sync(0xffffffff, local_JtJ(i,j), offset);

            #pragma unroll
            for (int i = 0; i < num_params; ++i)
                for (int offset = warpSize/2; offset > 0; offset /= 2)
                    local_Jtr(i) += __shfl_down_sync(0xffffffff, local_Jtr(i), offset);

            __shared__ double shared_JtJ[32][num_params][num_params];  // Max wraps for 1024 threads
            __shared__ double shared_Jtr[32][num_params];   // Max wraps for 1024 threads

            int lane = tid % warpSize;
            int warp_id = tid / warpSize;

            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < num_params; ++i)
                    #pragma unroll
                    for (int j = 0; j < num_params; ++j)
                        shared_JtJ[warp_id][i][j] = local_JtJ(i,j);

                #pragma unroll
                for (int i = 0; i < num_params; ++i)
                    shared_Jtr[warp_id][i] = local_Jtr(i);
            }

            __syncthreads();

            if (warp_id == 0) {
                int n_warps = (blockDim.x + warpSize - 1) / warpSize;

                Eigen::Matrix<double, num_params, num_params> block_JtJ;
                Eigen::Matrix<double, num_params, 1> block_Jtr;

                block_JtJ.setZero();
                block_Jtr.setZero();

                if (lane < n_warps) {
                    #pragma unroll
                    for (int i = 0; i < num_params; ++i)
                        #pragma unroll
                        for (int j = 0; j < num_params; ++j)
                            block_JtJ(i,j) = shared_JtJ[lane][i][j];

                    #pragma unroll
                    for (int i = 0; i < num_params; ++i)
                        block_Jtr(i) = shared_Jtr[lane][i];
                }

                #pragma unroll
                for (int i = 0; i < num_params; ++i)
                    #pragma unroll
                    for (int j = 0; j < num_params; ++j)
                        for (int offset = warpSize/2; offset > 0; offset /= 2)
                            block_JtJ(i,j) += __shfl_down_sync(0xffffffff, block_JtJ(i,j), offset);

                #pragma unroll
                for (int i = 0; i < num_params; ++i)
                    for (int offset = warpSize/2; offset > 0; offset /= 2)
                        block_Jtr(i) += __shfl_down_sync(0xffffffff, block_Jtr(i), offset);

                if (lane == 0) {
                    JtJ = block_JtJ;
                    Jtr = block_Jtr;
                }
            }

            __syncthreads();

        }

        __device__ param_t step(Eigen::Matrix<double, num_params, 1> dp, const param_t &pose) const
        {
            param_t pose_new;
            // The rotation is parameterized via the lie-rep. and post-multiplication
            //   i.e. R(delta) = R * expm([delta]_x)
            pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));

            // Translation is parameterized as (negative) shift in position
            //  i.e. t(delta) = t + R*delta
            pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
            return pose_new;
        }
    };

} // namespace TbrCudaPnP

#endif
