#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <helper_cuda.h>
#include <helper_string.h>

#include <TbrCudaPnP/turbo_ransac.cuh>
#include <TbrCudaPnP/ransac.cuh>
#include <TbrCudaPnP/msac.cuh>

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

    void turbo_ransac_pnp(const int points_num,
                         const Eigen::Matrix<double, 2, 1> *x,
                         const Eigen::Matrix<double, 3, 1> *X,
                         const RansacOptions_cuda &opt,
                         RansacStats_cuda &stats_cuda,
                         Eigen::Matrix<double, 4, 4> &result_pose,
                         int *inliers_bitmap)
    {

        double max_sq_error = opt.max_reproj_error * opt.max_reproj_error;

        double *p2d_norm_cuda = nullptr;
        CHECK(cudaMalloc(&p2d_norm_cuda, points_num * sizeof(double) * 2));
        CHECK(cudaMemcpy(p2d_norm_cuda, reinterpret_cast<const double *>(x),
                         points_num * sizeof(double) * 2, cudaMemcpyHostToDevice));

        double *p3d_cuda = nullptr;
        CHECK(cudaMalloc(&p3d_cuda, points_num * sizeof(double) * 3));
        CHECK(cudaMemcpy(p3d_cuda, reinterpret_cast<const double *>(X),
                         points_num * sizeof(double) * 3, cudaMemcpyHostToDevice));

        double *pose_host = new double[opt.max_iterations * 16];
        double *pose_cuda = nullptr;
        CHECK(cudaMalloc(&pose_cuda, 2 * opt.max_iterations * sizeof(double) * 16));

        int *inlier_count_vector_host = new int[opt.max_iterations];
        int *inlier_count_vector_cuda = nullptr;
        CHECK(cudaMalloc(&inlier_count_vector_cuda, 2 * opt.max_iterations * sizeof(int)));

        double *score_vector_host = new double[opt.max_iterations];
        double *score_vector_cuda = nullptr;
        CHECK(cudaMalloc(&score_vector_cuda, 2 * opt.max_iterations * sizeof(double)));

        //----------------------------------------------------------------------------//

        // Double buffering
        double* bufA_pose_cuda = pose_cuda;
        double* bufB_pose_cuda = pose_cuda + opt.max_iterations * 16;

        int* bufA_inlier_cuda = inlier_count_vector_cuda;
        int* bufB_inlier_cuda = inlier_count_vector_cuda + opt.max_iterations;

        double* bufA_score_cuda = score_vector_cuda;
        double* bufB_score_cuda = score_vector_cuda + opt.max_iterations;

        //----------------------------------------------------------------------------//

        // CUDA kernel setup
        int device;
        CHECK(cudaGetDevice(&device));

        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, device));
        ASSERT_WITH_MSG(prop.major >= 7u, "Compute capability must be >= 7.0");

        int max_shared_mem;
        CHECK(cudaDeviceGetAttribute(&max_shared_mem,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device));

        // Clamp shared memory usage to 64KB (65536 bytes)
        int shared_mem_size = (max_shared_mem > 65536) ? 65536 : max_shared_mem;

        // Set shared memory attribute for kernel
        CHECK(cudaFuncSetAttribute(turbo_ransac_pnp_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shared_mem_size));

        // Compute tile size based on shared memory available and size of struct
        int tile_size = shared_mem_size / sizeof(SharedPointData);
        tile_size = (tile_size > 384) ? 384 : tile_size; // 1024 doesn't work
        // tile_size = tile_size > prop.maxThreadsPerBlock ? prop.maxThreadsPerBlock : tile_size;

        // Start the random sampling
        curandState *rand_state;
        CHECK(cudaMalloc(&rand_state, sizeof(curandState) * opt.max_iterations));

        //----------------------------------------------------------------------------//

        double *d_min_score = nullptr;
        int *d_min_index = nullptr;
        CHECK(cudaMalloc(&d_min_score, sizeof(double)));
        CHECK(cudaMalloc(&d_min_index, sizeof(int)));

        int *h_min_index = nullptr;
        double *h_min_score = nullptr;
        CHECK(cudaMallocHost(&h_min_score, sizeof(double)));
        CHECK(cudaMallocHost(&h_min_index, sizeof(int)));

        //----------------------------------------------------------------------------//

        // Timing
        float milliseconds = 0;
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));

        //----------------------------------------------------------------------------//

        // TICK(INIT)
        int threads_init = 1024;
        int blocks_pose = (2 * opt.max_iterations + threads_init - 1) / threads_init;
        int blocks_score = (2 * opt.max_iterations + threads_init - 1) / threads_init;

        CHECK(cudaMemset(inlier_count_vector_cuda, 0, 2 * opt.max_iterations * sizeof(int)));

        init_identity_poses<<<blocks_pose, threads_init>>>(pose_cuda, 2 * opt.max_iterations);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        init_scores<<<blocks_score, threads_init>>>(score_vector_cuda, 2 * opt.max_iterations, DBL_MAX);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        // TOCK(INIT);

        //----------------------------------------------------------------------------//

        bool stop_iter = false;
        int iter = 0;
        const int max_iters = 100000;
        double best_score = DBL_MAX;
        // int h_min_index = 0;
        // double h_min_score = 0;
        int buffer_id = 0;

        while (!stop_iter && iter < max_iters)
        {

        // for (size_t i = 0; i < 1; i++)
        // {

            //----------------------------------------------------------------------------//

            setup_rand_kernel<<<opt.max_iterations, tile_size>>>(rand_state, iter + 1234UL); // 420UL
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            turbo_ransac_pnp_kernel<<<opt.max_iterations, tile_size, shared_mem_size>>>(rand_state,
                                                                                       tile_size,
                                                                                       points_num,
                                                                                       p2d_norm_cuda,
                                                                                       p3d_cuda,
                                                                                       bufA_pose_cuda,
                                                                                       max_sq_error,
                                                                                       bufA_score_cuda,
                                                                                       bufA_inlier_cuda);

            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            buffer_id = (iter / opt.max_iterations) % 2;
            printf("buffer_id: %d\n", buffer_id);

            /* Ablation */

            // buffer_id = 0; // only used in ablation

            int threadsPerBlock = 256;
            int blocks = (opt.max_iterations + threadsPerBlock - 1) / threadsPerBlock;
            merge_top_half_kernel_pairwise<<<blocks, threadsPerBlock>>>(pose_cuda,
                                                                        score_vector_cuda,
                                                                        inlier_count_vector_cuda,
                                                                        opt.max_iterations,
                                                                        buffer_id);

            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            std::swap(bufA_pose_cuda, bufB_pose_cuda);
            std::swap(bufA_score_cuda, bufB_score_cuda);
            std::swap(bufA_inlier_cuda, bufB_inlier_cuda);

            /* Ablation */

            // std::cout << bufA_pose_cuda << std::endl;
            // std::cout << bufA_score_cuda << std::endl;
            // std::cout << bufA_inlier_cuda << std::endl;

            //----------------------------------------------------------------------------//

            TICK(find_min_score_kernel)
            int threads = prop.maxThreadsPerBlock;
            int shared_mem = (threads / 32) * (sizeof(double) + sizeof(int));
            find_min_score_kernel<<<1, threads, shared_mem>>>(score_vector_cuda + buffer_id * opt.max_iterations, opt.max_iterations, d_min_score, d_min_index);

            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            // Copy result back to host
            CHECK(cudaMemcpy(h_min_score, d_min_score, sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_min_index, d_min_index, sizeof(int), cudaMemcpyDeviceToHost));
            TOCK(find_min_score_kernel)

            std::cout << "Minimum score: " << *h_min_score << " at index: " << *h_min_index << std::endl;

            CHECK(cudaMemcpy(inlier_count_vector_host, inlier_count_vector_cuda + buffer_id * opt.max_iterations + *h_min_index, sizeof(int), cudaMemcpyDeviceToHost));

            int inliner_nums = 0;
            int needed_iter = 0;
            inliner_nums = inlier_count_vector_host[0];

            double log_prob_missing_model = std::log(1.0 - 0.9999);
            double inlier_ratio = static_cast<double>(inliner_nums) / static_cast<double>(points_num);
            const double prob_outlier = 1.0 - std::pow(inlier_ratio, 3);

            needed_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * 3);

            std::cout << "Max inliers: " << inliner_nums << ", needed iter: " << needed_iter << std::endl;

            if (*h_min_score < best_score)
            {
                CHECK(cudaMemcpy(pose_host, pose_cuda + (buffer_id * opt.max_iterations + *h_min_index) * 16, sizeof(double) * 16, cudaMemcpyDeviceToHost));

                // int base = *h_min_index * 16; // 4x4 mat
                for (int row = 0; row < 4; row++)
                {
                    for (int col = 0; col < 4; col++)
                    {
                        result_pose(row, col) = static_cast<double>(pose_host[row * 4 + col]);
                    }
                }

                // std::cout << "Estimated Pose: \n " << result_pose << std::endl;
            }

            best_score = *h_min_score;

            //----------------------------------------------------------------------------//

            iter += opt.max_iterations;
            std::cout << iter << std::endl;
            if (iter > needed_iter) stop_iter = true;

        }

        //----------------------------------------------------------------------------//

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("Generated and processed %d iterations in %.3f ms\n", static_cast<int>(opt.max_iterations), milliseconds);

        //----------------------------------------------------------------------------//

        //----------------------------------------------------------------------------//

        CHECK(cudaFree(p2d_norm_cuda));
        CHECK(cudaFree(p3d_cuda));
        CHECK(cudaFree(pose_cuda));
        CHECK(cudaFree(inlier_count_vector_cuda));
        CHECK(cudaFree(score_vector_cuda));
        CHECK(cudaFree(rand_state));

        CHECK(cudaFree(d_min_score));
        CHECK(cudaFree(d_min_index));
        CHECK(cudaFreeHost(h_min_score));
        CHECK(cudaFreeHost(h_min_index));

        delete[] pose_host;
        delete[] inlier_count_vector_host;
        delete[] score_vector_host;

    }

}
