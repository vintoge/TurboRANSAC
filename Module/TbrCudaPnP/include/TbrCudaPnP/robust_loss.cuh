#ifndef __ROBUST_LOSS_CUH__
#define __ROBUST_LOSS_CUH__

#include <cmath>
#include <cfloat>

namespace TbrCudaPnP
{

    class TruncatedLoss_cuda
    {
    public:
        __host__ __device__ TruncatedLoss_cuda(double threshold) : squared_thr(threshold * threshold) {}

        __host__ __device__ double loss(double r2) const { return fmin(r2, squared_thr); }

        __host__ __device__ double weight(double r2) const { return (r2 < squared_thr) ? 1.0 : 0.0; }

    public:
        const double squared_thr;
    };


    class CauchyLoss_cuda
    {
    public:
        __host__ __device__ CauchyLoss_cuda(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}

        __host__ __device__ double loss(double r2) const { return log1p(r2 * inv_sq_thr); }

        __host__ __device__ double weight(double r2) const
        {
            return fmax(DBL_MIN, inv_sq_thr / (1.0 + r2 * inv_sq_thr));
        }

    public:
        const double inv_sq_thr;
    };

} 

#endif