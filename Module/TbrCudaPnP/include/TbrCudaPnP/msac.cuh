#ifndef __MSAC_CUH__
#define __MSAC_CUH__

#include <cstdint>
#include <Eigen/Dense>

namespace TbrCudaPnP
{

    typedef struct __align__(32) SharedPointData_t
    {
        Eigen::Matrix<double, 2, 1> p2d; // 2D point
        Eigen::Matrix<double, 3, 1> p3d; // 3D point
        uint8_t is_inlier;
        double point_score;
    } SharedPointData;

    __forceinline__ __device__ void compute_msac_scores(const double sq_threshold,
                                                        const Eigen::Matrix<double, 4, 4> &pose,
                                                        SharedPointData &point_shared)
    {

        const double P0_0 = pose(0, 0), P0_1 = pose(0, 1), P0_2 = pose(0, 2), P0_3 = pose(0, 3);
        const double P1_0 = pose(1, 0), P1_1 = pose(1, 1), P1_2 = pose(1, 2), P1_3 = pose(1, 3);
        const double P2_0 = pose(2, 0), P2_1 = pose(2, 1), P2_2 = pose(2, 2), P2_3 = pose(2, 3);

        const double x0 = point_shared.p2d(0);
        const double x1 = point_shared.p2d(1);

        const double X0 = point_shared.p3d(0);
        const double X1 = point_shared.p3d(1);
        const double X2 = point_shared.p3d(2);

        // Project 3D point
        const double z0 = P0_0 * X0 + P0_1 * X1 + P0_2 * X2 + P0_3;
        const double z1 = P1_0 * X0 + P1_1 * X1 + P1_2 * X2 + P1_3;
        const double z2 = P2_0 * X0 + P2_1 * X1 + P2_2 * X2 + P2_3;
        const double inv_z2 = 1.0 / z2;

        // Compute residual
        const double r0 = z0 * inv_z2 - x0;
        const double r1 = z1 * inv_z2 - x1;
        const double r_sq = r0 * r0 + r1 * r1;

        // Store result
        if (r_sq < sq_threshold && z2 > 0.0)
        {
            point_shared.is_inlier = 1;
            point_shared.point_score = r_sq;
        }
        else
        {
            point_shared.is_inlier = 0;
            point_shared.point_score = 0.0;
        }
    }



} // namespace TbrCudaPnP

#endif
