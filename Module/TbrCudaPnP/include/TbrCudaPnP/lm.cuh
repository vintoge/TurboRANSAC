#ifndef __LM_CUH__
#define __LM_CUH__

#include <TbrCudaPnP/lm_utils.cuh>
#include <TbrCudaPnP/msac.cuh>
#include <TbrCudaPnP/shared_storage.cuh>
#include <TbrCudaPnP/utils.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace TbrCudaPnP
{

    template <typename Problem, typename Param = typename Problem::param_t>
    __global__ void lm_impl(Problem problem, Param *parameters, const BundleOptions_cuda opt) // BundleStats_cuda &stats_in
    {
        int tid = threadIdx.x;

        constexpr int n_params = Problem::num_params;
        struct SharedState
        {
            Eigen::Matrix<double, n_params, n_params> JtJ;
            Eigen::Matrix<double, n_params, 1> Jtr;
            Param parameters_new;
            BundleStats_cuda stats;
        };
        __shared__ SharedStorage<SharedState> storage;
        if (tid == 0)
        {
            storage.construct();
        }
        __syncthreads();

        auto &JtJ = storage.get().JtJ;
        auto &Jtr = storage.get().Jtr;
        auto &parameters_new = storage.get().parameters_new;
        auto &stats = storage.get().stats;
        __shared__ double cost_new;
        __shared__ bool should_break;
        __shared__ bool recompute_jac;

        // Initialize
        problem.residual(opt, *parameters, stats.cost);


        if (tid == 0)
        {
            stats.initial_cost = stats.cost;
            stats.grad_norm = -1;
            stats.step_norm = -1;
            stats.invalid_steps = 0;
            stats.lambda = opt.initial_lambda;
            stats.iterations = 0;

            recompute_jac = true;
        }
        __syncthreads();

        // Each thread advances its own counter; only thread 0 writes shared stats.
        for (size_t iteration = 0; iteration < opt.max_iterations; ++iteration)
        {

            if (tid == 0)
            {
                should_break = false; // Reset flag
            }
            __syncthreads(); // Make sure all threads reset


            if (recompute_jac)
            {
                if (tid == 0)
                {
                    JtJ.setZero();
                    Jtr.setZero();
                }
                __syncthreads();

                problem.accumulate(opt, *parameters, JtJ, Jtr);

                if (tid == 0)
                {
                    stats.grad_norm = Jtr.norm();
                    // printf("stats.grad_norm: %f\n", stats.grad_norm);
                    if (stats.grad_norm < opt.gradient_tol)
                    {
                        should_break = true;
                    }
                }
                __syncthreads();
            }

            // if (tid == 0)
            // {
            //     if (recompute_jac)
            //     {
            //         JtJ.setZero();
            //         Jtr.setZero();

            //         problem.accumulate(opt, *parameters, JtJ, Jtr);

            //         stats.grad_norm = Jtr.norm();
            //         if (stats.grad_norm < opt.gradient_tol)
            //         {
            //             should_break = true;
            //         }
            //     }
            // }
            // __syncthreads();


            if (tid == 0)
            {
                #pragma unroll
                for (size_t k = 0; k < n_params; k++)
                {
                    JtJ(k, k) += stats.lambda;
                }

                Eigen::Matrix<double, n_params, 1> sol;
                solve_cholesky(JtJ, Jtr, sol);
                sol = -1 * sol;

                stats.step_norm = sol.norm();
                if (stats.step_norm < opt.step_tol)
                {
                    should_break = true;
                }

                parameters_new = problem.step(sol, *parameters);
            }
            __syncthreads();

            if (should_break)
                break;

            //----------------------------------------------------------------------------------------------------//

            problem.residual(opt, parameters_new, cost_new);

            if (tid == 0)
            {

                if (cost_new < stats.cost)
                {
                    *parameters = parameters_new;
                    // printf("parameters(): %f\n", parameters->t(0));
                    stats.lambda = std::max(opt.min_lambda, stats.lambda / 10);
                    stats.cost = cost_new;
                    recompute_jac = true;
                }
                else
                {
                    stats.invalid_steps++;
                    
                    #pragma unroll
                    for (size_t k = 0; k < n_params; k++)
                    {
                        JtJ(k, k) -= stats.lambda;  // Remove dampening
                    }
                    stats.lambda = std::min(opt.max_lambda, stats.lambda * 10);
                    recompute_jac = false;
                }

                ++stats.iterations;
                // printf("stats.iterations: %d\n", stats.iterations);
            }
            __syncthreads();
        }
    }

 

} // namespace TbrCudaPnP

#endif
