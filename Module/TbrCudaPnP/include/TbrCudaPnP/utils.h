#ifndef __UTILS_H__
#define __UTILS_H__

#include <cfloat>
#include <cstddef>
#include <iostream>

#include <TbrCudaPnP/config.h>

namespace TbrCudaPnP
{


    struct RansacOptions_cuda
    {
        size_t max_iterations = 100000;
        size_t min_iterations = 1000;
        double dyn_num_trials_mult = 3.0;
        double success_prob = 0.9999;
        double max_reproj_error = 12.0;  // used for 2D-3D matches
        double max_epipolar_error = 1.0; // used for 2D-2D matches
        unsigned long seed = 0;
        // If we should use PROSAC sampling. Assumes data is sorted
        bool progressive_sampling = false;
        size_t max_prosac_iterations = 100000;
        // Whether we should use real focal length checking: https://arxiv.org/abs/2311.16304
        // Assumes that principal points of both cameras are at origin.
        bool real_focal_check = false;
        // Whether to treat the input 'best_model' as an initial model and score it before running the main RANSAC loop
        bool score_initial_model = false;
    };

    struct RansacStats_cuda
    {
        size_t refinements = 0;
        size_t iterations = 0;
        size_t num_inliers = 0;
        double inlier_ratio = 0;
        double model_score = DBL_MAX;
    };

    struct BundleOptions_cuda
    {
        size_t max_iterations = 100;
        enum LossType {
            TRIVIAL,
            TRUNCATED,
            HUBER,
            CAUCHY,
            // This is the TR-IRLS scheme from Le and Zach, 3DV 2021
            TRUNCATED_LE_ZACH
        } loss_type = LossType::CAUCHY;
        double loss_scale = 1.0;
        double gradient_tol = 1e-10;
        double step_tol = 1e-8;
        double initial_lambda = 1e-3;
        double min_lambda = 1e-10;
        double max_lambda = 1e10;
        bool verbose = false;
    };

    struct BundleStats_cuda
    {
        size_t iterations = 0;
        // size_t iterations;
        double initial_cost;
        double cost;
        double lambda;
        size_t invalid_steps;
        double step_norm;
        double grad_norm;
    };


}

#endif