#ifndef __ENTRY_H__
#define __ENTRY_H__

#include <iostream>

#include <TbrCudaPnP/pose.cuh>
#include <TbrCudaPnP/utils.h>
#include <TbrCudaPnP/camera.cuh>

namespace TbrCudaPnP
{

    RansacStats_cuda estimate_absolute_pose_cuda(const std::vector<Eigen::Vector2d> &points2D,
                                                 const std::vector<Eigen::Vector3d> &points3D,
                                                 const Camera_cuda &camera,
                                                 const RansacOptions_cuda &ransac_opt,
                                                 const BundleOptions_cuda &bundle_opt,
                                                 CameraPose_cuda *pose,
                                                 std::vector<char> *inliers);

    // Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
    void get_inliers_mx(const CameraPose_cuda &pose,
                        const std::vector<Eigen::Vector2d> &x,
                        const std::vector<Eigen::Vector3d> &X,
                        double sq_threshold, std::vector<char> *inliers);

}

#endif