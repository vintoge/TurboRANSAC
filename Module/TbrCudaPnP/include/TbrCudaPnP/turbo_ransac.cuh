#ifndef __TURBO_RANSAC_CUH__
#define __TURBO_RANSAC_CUH__

#include <iostream>
#include <TbrCudaPnP/config.h>
#include <TbrCudaPnP/utils.h>

#include <Eigen/Dense>

namespace TbrCudaPnP
{

    void turbo_ransac_pnp(const int points_num,
                         const Eigen::Matrix<double, 2, 1> *x,
                         const Eigen::Matrix<double, 3, 1> *X,
                         const RansacOptions_cuda &opt,
                         RansacStats_cuda &stats_cuda,
                         Eigen::Matrix<double, 4, 4> &result_pose,
                         int *inliers_bitmap);

}

#endif
