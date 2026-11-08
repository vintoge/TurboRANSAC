#include <TbrCudaPnP/entry.h>
#include <TbrCudaPnP/config.h>
#include <TbrCudaPnP/utils.h>

#include <TbrCudaPnP/turbo_ransac.cuh>
#include <TbrCudaPnP/ba.cuh>

namespace TbrCudaPnP
{

    // Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
    void get_inliers_mx(const CameraPose_cuda &pose,
                        const std::vector<Eigen::Vector2d> &x,
                        const std::vector<Eigen::Vector3d> &X,
                        double sq_threshold, std::vector<char> *inliers)
    {
        inliers->resize(x.size());
        const Eigen::Matrix3d R = pose.R();

        for (size_t k = 0; k < x.size(); ++k)
        {
            Eigen::Vector3d Z = (R * X[k] + pose.t);
            double r2 = (Z.hnormalized() - x[k]).squaredNorm();
            (*inliers)[k] = (r2 < sq_threshold && Z(2) > 0.0);
        }
    }

    RansacStats_cuda estimate_absolute_pose_cuda(const std::vector<Eigen::Vector2d> &points2D,
                                                 const std::vector<Eigen::Vector3d> &points3D,
                                                 const Camera_cuda &camera,
                                                 const RansacOptions_cuda &ransac_opt,
                                                 const BundleOptions_cuda &bundle_opt,
                                                 CameraPose_cuda *pose,
                                                 std::vector<char> *inliers)
    {

        assert(points2D.size() == points3D.size() && "2D and 3D point counts must match.");
        int points_num = points2D.size();

        //----------------------------------------------------------------------------------//

        std::vector<Eigen::Vector2d> points2D_calib(points2D.size());
        for (size_t k = 0; k < points2D.size(); ++k)
        {
            camera.unproject(points2D[k], &points2D_calib[k]);
        }

        RansacOptions_cuda ransac_opt_scaled_cuda = ransac_opt;
        ransac_opt_scaled_cuda.max_reproj_error /= camera.focal();

        //----------------------------------------------------------------------------------//

        Eigen::Matrix<double, 2, 1> p2d_norm[points_num];
        Eigen::Matrix<double, 3, 1> p3d[points_num];
        int inliers_bitmap[points_num];

        for (size_t i = 0; i < points_num; i++)
        {
            p2d_norm[i](0) = static_cast<double>(points2D_calib[i].x());
            p2d_norm[i](1) = static_cast<double>(points2D_calib[i].y());

            p3d[i](0) = static_cast<double>(points3D[i].x());
            p3d[i](1) = static_cast<double>(points3D[i].y());
            p3d[i](2) = static_cast<double>(points3D[i].z());

            inliers_bitmap[i] = 0;
        }

        RansacStats_cuda stats_cuda;
        Eigen::Matrix<double, 4, 4> result_pose = Eigen::Matrix<double, 4, 4>::Identity();
        // std::cout << result_pose << std::endl;

        // TICK(RANSAC_CUDA)
        turbo_ransac_pnp(points_num, p2d_norm, p3d, ransac_opt_scaled_cuda, stats_cuda, result_pose, inliers_bitmap);
        // TOCK(RANSAC_CUDA)

        //----------------------------------------------------------------------------//

        Eigen::Matrix3d R = result_pose.block<3, 3>(0, 0);
        Eigen::Vector3d t = result_pose.block<3, 1>(0, 3);

        CameraPose_cuda test_pose(R, t);
        get_inliers_mx(test_pose, points2D_calib, points3D, ransac_opt_scaled_cuda.max_reproj_error * ransac_opt_scaled_cuda.max_reproj_error, inliers);

        //---------------------------------------------------------------------------------------------//

        const double scale = 1.0 / camera.focal();
        size_t count = 0;

        Eigen::Matrix<double, 2, 1> p2d_inliers[points2D.size()];
        Eigen::Matrix<double, 3, 1> p3d_inliers[points2D.size()];

        for (size_t k = 0; k < points2D.size(); ++k)
        {
            if (!(*inliers)[k])
                continue;

            p2d_inliers[count] = points2D[k] * scale;
            p3d_inliers[count] = points3D[k];
            count += 1;
        }

        int final_inliers_num = count;

        //---------------------------------------------------------------------------------------------//

        Camera_cuda norm_camera_cuda;
        norm_camera_cuda.model_id = 1; // "PINHOLE"
        norm_camera_cuda.width = camera.width;
        norm_camera_cuda.height = camera.height;

        double cam_k[4] = {camera.params[0], camera.params[1], camera.params[2], camera.params[3]};
        norm_camera_cuda.params = cam_k;
        norm_camera_cuda.rescale(scale);

        //----------------------------------------------------------------------------//

        CameraPose_cuda test_pose_cuda(R, t);

        // Ablation

        BundleOptions_cuda bundle_opt_scaled_cuda;
        bundle_opt_scaled_cuda.loss_scale *= scale;
        // bundle_opt_scaled_cuda.max_iterations = 5;

        bundle_adjust_cuda(final_inliers_num,
                           p2d_inliers,
                           p3d_inliers,
                           norm_camera_cuda,
                           &test_pose_cuda,
                           1,
                           bundle_opt_scaled_cuda);

        // Ablation

        pose->q = test_pose_cuda.q;
        pose->t = test_pose_cuda.t;

        //----------------------------------------------------------------------------//

        stats_cuda.model_score = 0;

        return stats_cuda;
    }

}
