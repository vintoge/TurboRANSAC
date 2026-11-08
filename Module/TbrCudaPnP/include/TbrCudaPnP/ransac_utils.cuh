#ifndef __RANSAC_UTILS_CUH__
#define __RANSAC_UTILS_CUH__

#include <TbrCudaPnP/config.h>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    // Initialize one RNG state per block
    __global__ void setup_rand_kernel(curandState *state, unsigned long seed)
    {
        int block_id = blockIdx.x;
        if (threadIdx.x == 0)
            curand_init(seed, block_id, 0, &state[block_id]);
    }

} // namespace TbrCudaPnP

#endif
