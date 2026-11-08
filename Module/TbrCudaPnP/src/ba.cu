#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_string.h>

#include <TbrCudaPnP/lm.cuh>
#include <TbrCudaPnP/jaco.cuh>
#include <TbrCudaPnP/ba.cuh>
#include <TbrCudaPnP/utils.h>
#include <TbrCudaPnP/robust_loss.cuh>

#include <TbrCudaPnP/turbo_ransac.cuh>
#include <TbrCudaPnP/msac.cuh>
#include <TbrCudaPnP/p3p.cuh>

#include <TbrCudaPnP/ticktock.h>
#include <TbrCudaPnP/cudaCheck.h>

#include <limits>
#include <cfloat>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <Eigen/Dense>

namespace TbrCudaPnP
{

    template <typename WeightType, typename LossFunction>
    BundleStats_cuda bundle_adjust_cuda(const int points_num,
                                        const Eigen::Matrix<double, 2, 1> *x,
                                        const Eigen::Matrix<double, 3, 1> *X,
                                        const Camera_cuda &camera,
                                        CameraPose_cuda *pose,
                                        const int poses_num,
                                        const BundleOptions_cuda &opt,
                                        const WeightType &weights)
    {

        double *p2d_norm_cuda = nullptr;
        CHECK(cudaMalloc(&p2d_norm_cuda, points_num * sizeof(double) * 2));
        CHECK(cudaMemcpy(p2d_norm_cuda, reinterpret_cast<const double *>(x),
                         points_num * sizeof(double) * 2, cudaMemcpyHostToDevice));

        double *p3d_cuda = nullptr;
        CHECK(cudaMalloc(&p3d_cuda, points_num * sizeof(double) * 3));
        CHECK(cudaMemcpy(p3d_cuda, reinterpret_cast<const double *>(X),
                         points_num * sizeof(double) * 3, cudaMemcpyHostToDevice));

        double *cam_k_cuda = nullptr;
        CHECK(cudaMalloc(&cam_k_cuda, sizeof(double) * 4));
        CHECK(cudaMemcpy(cam_k_cuda, reinterpret_cast<const double *>(camera.params),
                         sizeof(double) * 4, cudaMemcpyHostToDevice));

        CameraJacobianAccumulator_cuda::param_t pose_init;
        pose_init.q = pose[0].q;
        pose_init.t = pose[0].t;

        CameraJacobianAccumulator_cuda::param_t *refine_pose;
        CHECK(cudaMalloc(&refine_pose, sizeof(CameraJacobianAccumulator_cuda::param_t)));
        CHECK(cudaMemcpy(refine_pose, &pose_init, sizeof(CameraJacobianAccumulator_cuda::param_t), cudaMemcpyHostToDevice));

        CameraJacobianAccumulator_cuda accum;
        accum.x = p2d_norm_cuda;
        accum.X = p3d_cuda;
        accum.cam_k = cam_k_cuda;
        accum.n = points_num;
        // std::cout << points_num << std::endl;

        BundleOptions_cuda opt_cuda;  // Passed by value
        opt_cuda = opt;

        //----------------------------------------------------------------------------//

        // Timing
        float milliseconds = 0;
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        for (size_t i = 0; i < 1; i++)
        {
            lm_impl<decltype(accum)><<<1, 256>>>(accum, refine_pose, opt_cuda);     

            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Bundle Adjust %d iterations in %.3f ms\n", static_cast<int>(opt.max_iterations), milliseconds);

        //----------------------------------------------------------------------------//

        CHECK(cudaMemcpy(&pose_init, refine_pose, sizeof(CameraJacobianAccumulator_cuda::param_t), cudaMemcpyDeviceToHost));
        (*pose) = pose_init;

        CHECK(cudaFree(p2d_norm_cuda));
        CHECK(cudaFree(p3d_cuda));
        CHECK(cudaFree(cam_k_cuda));
        CHECK(cudaFree(refine_pose));

        BundleStats_cuda final_stats;
        return final_stats;
    }

    // Entry point for PnP refinement
    BundleStats_cuda bundle_adjust_cuda(const int points_num,
                                        const Eigen::Matrix<double, 2, 1> *x,
                                        const Eigen::Matrix<double, 3, 1> *X,
                                        const Camera_cuda &camera,
                                        CameraPose_cuda *pose,
                                        const int poses_num,
                                        const BundleOptions_cuda &opt,
                                        const double *weights)
    {

        if (weights == nullptr)
        {
            return bundle_adjust_cuda<UniformWeightVector_cuda, CauchyLoss_cuda>(points_num, x, X, camera, pose, poses_num, opt, UniformWeightVector_cuda());
        }
        else
        {
            RealWeightVector_cuda real_weight(weights);
            return bundle_adjust_cuda<RealWeightVector_cuda, CauchyLoss_cuda>(points_num, x, X, camera, pose, poses_num, opt, real_weight);
        }
    }

}
