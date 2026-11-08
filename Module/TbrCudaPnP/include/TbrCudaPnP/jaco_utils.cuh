#ifndef __JACO_UTILS_CUH__
#define __JACO_UTILS_CUH__

#include <TbrCudaPnP/config.h>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{


    // === Uniform weight vector: always returns 1.0 ===
    class UniformWeightVector_cuda
    {
    public:
        __forceinline__ __host__ __device__
        UniformWeightVector_cuda() {}

        __forceinline__ __device__ constexpr double operator[](std::size_t idx) const
        {
            return 1.0;
        }
    };

    // === Real weight vector: uses pointer to global memory ===
    // template <typename T>
    class RealWeightVector_cuda
    {
    public:
        const double *data;

        __forceinline__ __host__ __device__
        RealWeightVector_cuda(const double *data_) : data(data_) {}

        __forceinline__ __device__ double operator[](std::size_t idx) const
        {
            return data[idx];
        }
    };


    

} // namespace TbrCudaPnP

#endif
