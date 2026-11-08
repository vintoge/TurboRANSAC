#ifndef __BA_CUH__
#define __BA_CUH__

#include <iostream>
#include <TbrCudaPnP/config.h>
#include <TbrCudaPnP/utils.h>

#include <TbrCudaPnP/camera.cuh>
#include <TbrCudaPnP/pose.cuh>

#include <Eigen/Dense>

namespace TbrCudaPnP
{

    // Entry point for PnP refinement
    BundleStats_cuda bundle_adjust_cuda(const int points_num,
                                        const Eigen::Matrix<double, 2, 1> *x,
                                        const Eigen::Matrix<double, 3, 1> *X,
                                        const Camera_cuda &camera,
                                        CameraPose_cuda *pose,
                                        const int poses_num,
                                        const BundleOptions_cuda &opt,
                                        const double *weights = nullptr);

}

#endif