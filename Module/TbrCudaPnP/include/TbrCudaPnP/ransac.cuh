#ifndef __RANSAC_CUH__
#define __RANSAC_CUH__

#include <TbrCudaPnP/ransac_utils.cuh>
#include <TbrCudaPnP/msac.cuh>
#include <TbrCudaPnP/p3p.cuh>
#include <TbrCudaPnP/shared_storage.cuh>

#include <TbrCudaPnP/utils.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

namespace TbrCudaPnP
{

    __global__ void init_scores(double* scores, int total, double val)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total) scores[idx] = val;
    }

    __global__ void init_identity_poses(double* poses, int num_poses)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_poses)
        {
            // Each pose has 16 elements (4x4 matrix)
            double* pose = poses + idx * 16;

            // Set identity matrix
            for (int i = 0; i < 16; ++i) pose[i] = 0.0;
            pose[0]  = 1.0;
            pose[5]  = 1.0;
            pose[10] = 1.0;
            pose[15] = 1.0;
        }
    }

    __global__ void merge_top_half_kernel_pairwise(double * __restrict__ pose_base,
                                                   double * __restrict__ score_base,
                                                   int * __restrict__ inlier_base,
                                                   int buffer_size,
                                                   int buffer_id)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= buffer_size) return;

        int curr_idx, prev_idx;
        if (buffer_id == 0) {
            curr_idx = idx;
            prev_idx = idx + buffer_size;
        } else {
            curr_idx = idx + buffer_size;
            prev_idx = idx;
        }

        double score_curr = score_base[curr_idx];
        double score_prev = score_base[prev_idx];

        int best_idx = (score_curr <= score_prev) ? curr_idx : prev_idx;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            pose_base[curr_idx * 16 + j] = pose_base[best_idx * 16 + j];
        }

        score_base[curr_idx] = score_base[best_idx];
        inlier_base[curr_idx] = inlier_base[best_idx];
    }

    __device__ void warp_min_reduce(double &val, int &idx)
    {
        for (int offset = 16; offset > 0; offset /= 2)
        {
            double other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val < val)
            {
                val = other_val;
                idx = other_idx;
            }
        }
    }

    __global__ void find_min_score_kernel(const double* data, int n, double* d_min_val, int* d_min_idx)
    {
        extern __shared__ double sdata[]; // shared memory for warp mins and idxs
        double* svals = sdata;
        int* sidxs = (int*)&svals[blockDim.x / 32];

        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp_id = tid / 32;

        // Each thread processes multiple elements, stride by blockDim.x
        double local_min_val = DBL_MAX;
        int local_min_idx = -1;

        for (int i = tid; i < n; i += blockDim.x)
        {
            double val = data[i];
            if (val < local_min_val)
            {
                local_min_val = val;
                local_min_idx = i;
            }
        }

        // Warp-level reduction with shuffle
        warp_min_reduce(local_min_val, local_min_idx);

        // Lane 0 in each warp writes warp result to shared memory
        if (lane == 0)
        {
            svals[warp_id] = local_min_val;
            sidxs[warp_id] = local_min_idx;
        }

        __syncthreads();

        // First warp reduces all warp results
        if (warp_id == 0)
        {
            double val = (tid < (blockDim.x / 32)) ? svals[lane] : DBL_MAX;
            int idx = (tid < (blockDim.x / 32)) ? sidxs[lane] : -1;

            warp_min_reduce(val, idx);

            if (lane == 0)
            {
                *d_min_val = val;
                *d_min_idx = idx;
            }
        }
    }

    // Not take shared memory size with points number into account
    __global__ void turbo_ransac_pnp_kernel(curandState *rand_state,
                                           int tile_size,
                                           int points_num,
                                           const double *p2d_norm_cuda,
                                           const double *p3d_cuda,
                                           double *pose_cuda,
                                           double max_sq_error,
                                           double *score_vector_cuda,
                                           int *inlier_count_vector_cuda)
    {

        int bid = blockIdx.x;
        int tid = threadIdx.x;

        // extern __shared__ SharedPointData shared_mem[];
        // __shared__ uint8_t inlier_temp[1024];
        // __shared__ double score_temp[1024];

        struct SharedState
        {
            Eigen::Matrix<double, 4, 4> pose_candidate[4];
            Eigen::Matrix<double, 3, 1> p2d_sample[3]; // x, y, 1
            Eigen::Matrix<double, 3, 1> p3d_sample[3]; // x, y, z
        };
        __shared__ SharedStorage<SharedState> storage;
        if (tid == 0)
        {
            storage.construct();
        }
        __syncthreads();

        auto &pose_candidate = storage.get().pose_candidate;
        auto &p2d_sample = storage.get().p2d_sample;
        auto &p3d_sample = storage.get().p3d_sample;

        // First thread per block generates 3 unique samples
        if (tid == 0) // " && bid == 0" just for test, should not used in normal operation
        {
            curandState localState = rand_state[bid];
            int s[3];
            bool valid = false;

            while (!valid)
            {
                for (int i = 0; i < 3; i++)
                {
                    float r = curand_uniform(&localState);
                    s[i] = static_cast<int>(r * points_num);
                    if (s[i] >= points_num)
                        s[i] = (points_num - 1);
                }
                valid = (s[0] != s[1] && s[0] != s[2] && s[1] != s[2]);
            }

            for (int i = 0; i < 3; i++)
            {
                p2d_sample[i](0) = p2d_norm_cuda[2 * s[i] + 0];
                p2d_sample[i](1) = p2d_norm_cuda[2 * s[i] + 1];
                p2d_sample[i](2) = 1;
                // p2d_sample[i] = p2d_sample[i].normalized();
                p2d_sample[i].normalize();

                p3d_sample[i](0) = p3d_cuda[3 * s[i] + 0];
                p3d_sample[i](1) = p3d_cuda[3 * s[i] + 1];
                p3d_sample[i](2) = p3d_cuda[3 * s[i] + 2];
            }

            rand_state[bid] = localState;
        }

        __syncthreads(); // Sync all threads before p3p process

        __shared__ int valid_result;
        __shared__ bool early_exit;

        if (tid == 0) // just for test " && bid == 0 "
        {
            valid_result = p3p<double>(p2d_sample, p3d_sample, pose_candidate);
            early_exit = (valid_result == 0);

            if (early_exit)
            {
                score_vector_cuda[bid] = DBL_MAX;
                inlier_count_vector_cuda[bid] = 0;

                #pragma unroll
                for (int row = 0; row < 4; row++)
                {
                    #pragma unroll
                    for (int col = 0; col < 4; col++)
                    {
                        pose_cuda[bid * 16 + row * 4 + col] = (row == col) ? 1.0 : 0.0;
                    }
                }
            }

        }

        __syncthreads();

        if (early_exit)
            return;

        extern __shared__ SharedPointData shared_mem[]; // Dynamic shared memory
        __shared__ double score_shm[384 / 32];
        __shared__ int inlier_shm[384 / 32]; // 12 warps

        __shared__ double best_score;
        __shared__ int best_pose_idx;
        __shared__ int best_inliers_num;

        if (tid == 0)
        {
            best_score = DBL_MAX;
            best_pose_idx = -1;
            best_inliers_num = 0;
        }
        __syncthreads();

        for (size_t pose_idx = 0; pose_idx < valid_result; pose_idx++)
        {
            //--------------------------------------------------------------------------------------------//

            double thread_score = 0.0;
            int thread_inlier = 0;

            for (int i = tid; i < points_num; i += blockDim.x)
            {
                SharedPointData pt;

                pt.p2d(0) = p2d_norm_cuda[2 * i + 0];
                pt.p2d(1) = p2d_norm_cuda[2 * i + 1];
                pt.p3d(0) = p3d_cuda[3 * i + 0];
                pt.p3d(1) = p3d_cuda[3 * i + 1];
                pt.p3d(2) = p3d_cuda[3 * i + 2];
                pt.is_inlier = 0;
                pt.point_score = 0.0;

                compute_msac_scores(max_sq_error, pose_candidate[pose_idx], pt);

                thread_score += pt.point_score;
                thread_inlier += pt.is_inlier;
            }

            __syncthreads();

            //--------------------------------------------------------------------------------------------//

            //--------------------------------------------------------------------------------------------//

            // Warp reduction for score and inliers
            double warp_score = thread_score;
            int warp_inlier = thread_inlier;
            // uint mask = __activemask();

            for (int offset = 16; offset > 0; offset /= 2)
            {
                warp_score += __shfl_down_sync(0xffffffff, warp_score, offset);
                warp_inlier += __shfl_down_sync(0xffffffff, warp_inlier, offset);
            }

            if ((tid & 31) == 0)
            {
                score_shm[tid / 32] = warp_score;
                inlier_shm[tid / 32] = warp_inlier;
            }
            __syncthreads();

            // Final reduction among warps
            if (tid < 32)
            {
                double block_score = (tid < (blockDim.x / 32)) ? score_shm[tid] : 0;
                int block_inlier = (tid < (blockDim.x / 32)) ? inlier_shm[tid] : 0;

                for (int offset = 16; offset > 0; offset /= 2)
                {
                    block_score += __shfl_down_sync(0xffffffff, block_score, offset);
                    block_inlier += __shfl_down_sync(0xffffffff, block_inlier, offset);
                }

                if (tid == 0)
                {
                    double score = block_score + (points_num - block_inlier) * max_sq_error;
                    if (score < best_score)
                    {
                        best_score = score;
                        best_pose_idx = pose_idx;
                        best_inliers_num = block_inlier;
                    }
                }

            }
            __syncthreads();

            //--------------------------------------------------------------------------------------------//
        }

        // Save best result
        if (tid == 0)
        {
            score_vector_cuda[bid] = (best_pose_idx >= 0) ? best_score : DBL_MAX;
            inlier_count_vector_cuda[bid] = best_inliers_num;

            #pragma unroll
            for (int row = 0; row < 4; row++)
            {
                #pragma unroll
                for (int col = 0; col < 4; col++)
                {
                    // if (bid == 7381)  // 7381, 22016
                    // {
                    // printf("final_pose: %f\n", pose_candidate[best_pose_idx](row, col));
                    pose_cuda[bid * 16 + row * 4 + col] = pose_candidate[best_pose_idx](row, col);
                    // }
                }
            }
        }

        // __syncthreads();

    }

} // namespace TbrCudaPnP

#endif
