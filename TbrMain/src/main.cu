#include <TbrCudaPnP/header.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>
using json = nlohmann::json;


struct FrameData
{
    Eigen::Matrix4d gt_w2c;
    Eigen::Matrix3d intrinsic_K;
    std::vector<Eigen::Vector2d> p2d;
    std::vector<Eigen::Vector3d> p3d;
};

bool load_json(const std::string &path, std::map<std::string, FrameData> &frames)
{
    std::ifstream file(path);
    if (!file.is_open())
        return false;

    json j;
    file >> j;

    const int total = j.size();
    int current = 0;
    const int bar_width = 50;

    std::cout << "Loading frames from: " << path << "\n";

    for (auto &[frame_name, content] : j.items())
    {
        // Show progress bar
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(bar_width * progress);

        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";
        std::cout.flush();

        current++;

        if (!content.contains("gt_w2c") || content["gt_w2c"].is_null() ||
            !content.contains("intrinsic K") || content["intrinsic K"].is_null() ||
            !content.contains("point pair") || content["point pair"].is_null())
        {
            continue;
        }

        FrameData frame;

        try
        {
            for (int i = 0; i < 4; ++i)
                for (int j_ = 0; j_ < 4; ++j_)
                    frame.gt_w2c(i, j_) = content["gt_w2c"].at(i).at(j_);

            for (int i = 0; i < 3; ++i)
                for (int j_ = 0; j_ < 3; ++j_)
                    frame.intrinsic_K(i, j_) = content["intrinsic K"].at(i).at(j_);

            for (const auto &pair : content["point pair"])
            {
                if (!pair.is_array() || pair.size() != 5)
                    continue;

                frame.p2d.emplace_back(pair[0].get<double>(), pair[1].get<double>());
                frame.p3d.emplace_back(pair[2].get<double>(), pair[3].get<double>(), pair[4].get<double>());
            }

            frames[frame_name] = std::move(frame);
        }
        catch (const json::exception &e)
        {
            std::cerr << "\nError parsing frame '" << frame_name << "': " << e.what() << "\n";
            continue;
        }
    }

    std::cout << "\r[" << std::string(bar_width, '=') << "] 100%\n";
    std::cout << "Loaded " << frames.size() << " valid frames from file.\n";
    return true;
}

bool write_predicted_poses_with_progress(const std::string &output_path,
                                         const json &pred_poses_json)
{
    const int bar_width = 50;
    int total = static_cast<int>(pred_poses_json.size());
    int current = 0;

    std::ofstream out_file(output_path);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open file for writing predicted poses: " << output_path << "\n";
        return false;
    }

    out_file << "{\n";

    for (auto it = pred_poses_json.begin(); it != pred_poses_json.end(); ++it, ++current)
    {
        out_file << "  \"" << it.key() << "\": " << it.value().dump(4);
        if (current + 1 < total)
            out_file << ",";
        out_file << "\n";

        // Show progress bar
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(bar_width * progress);

        std::cout << "\rWriting predicted poses: [";
        for (int i = 0; i < bar_width; ++i)
            std::cout << (i < pos ? '=' : ' ');
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";
        std::cout.flush();
    }

    out_file << "}\n";

    std::cout << "\rWriting predicted poses: [" << std::string(bar_width, '=') << "] 100%\n";
    std::cout << "Saved predicted poses to " << output_path << "\n";

    return true;
}

std::pair<double, double> cal_pose_error(const Eigen::Matrix4d &pred_w2c, const Eigen::Matrix4d &gt_w2c)
{
    Eigen::Matrix3d pred_R = pred_w2c.block<3, 3>(0, 0);
    Eigen::Matrix3d gt_R = gt_w2c.block<3, 3>(0, 0);

    Eigen::Vector3d pred_t = (-pred_R.transpose()) * pred_w2c.block<3, 1>(0, 3);
    Eigen::Vector3d gt_t = (-gt_R.transpose()) * gt_w2c.block<3, 1>(0, 3);

    Eigen::Matrix3d r_err = gt_R * pred_R.transpose();

    const Eigen::Quaterniond q_err(r_err);
    const double sin_half_angle = std::hypot(q_err.x(), q_err.y(), q_err.z());
    double ae = 2.0 * std::atan2(sin_half_angle, std::abs(q_err.w())) * 180.0 / M_PI;

    double te = (pred_t - gt_t).norm() * 100.0;

    return {ae, te};
}

std::tuple<double, double, double, Eigen::Matrix4d> evaluate_pose(
    const Eigen::Matrix4d &gt_pose,
    const Eigen::Matrix3d &cam_k,
    const std::vector<Eigen::Vector2d> &p2d,
    const std::vector<Eigen::Vector3d> &p3d)
{
    std::vector<double> cam_params(4);
    cam_params[0] = cam_k(0, 0); // fx
    cam_params[1] = cam_k(1, 1); // fy
    cam_params[2] = cam_k(0, 2); // cx
    cam_params[3] = cam_k(1, 2); // cy

    TbrCudaPnP::Camera_cuda cam;
    cam.width = static_cast<int>(cam_params[2]) * 2;
    cam.height = static_cast<int>(cam_params[3]) * 2;
    cam.params = cam_params.data();

    TbrCudaPnP::RansacOptions_cuda ransac_opt;
    ransac_opt.max_iterations = 1024;
    ransac_opt.success_prob = 0.9999;

    TbrCudaPnP::BundleOptions_cuda ba_opt;
    TbrCudaPnP::CameraPose_cuda result_pose;
    std::vector<char> result_inliers(p2d.size(), 0);

    auto start = std::chrono::high_resolution_clock::now();
    TbrCudaPnP::estimate_absolute_pose_cuda(p2d, p3d, cam,
                                                    ransac_opt, ba_opt,
                                                    &result_pose, &result_inliers);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    Eigen::Matrix4d pred_pose = result_pose.Rt_homo();

    auto [ae_gpu, te_gpu] = cal_pose_error(pred_pose, gt_pose);

    std::cout << "(GPU) Estimated Pose: \n" << pred_pose << std::endl;
    std::cout << "(GPU) Angle Error (deg): " << ae_gpu << "\n";
    std::cout << "(GPU) Translation Error (cm): " << te_gpu << "\n";
    std::cout << "(GPU) Runtime (ms): " << elapsed_ms << "\n";

    return {ae_gpu, te_gpu, elapsed_ms, pred_pose};
}

struct ResultEntry
{
    std::string scene;
    double avg_ae;
    double avg_te;
    double avg_time_ms;
    int frame_count;
};

int main()
{
    const std::string data_dir = "../data/7scenes";
    std::vector<std::string> scenes = {
        "chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"
    };

    std::vector<ResultEntry> results;
    const bool save_predicted_poses = false; // Set false to skip writing

    for (const auto &scene : scenes)
    {
        std::string json_path = data_dir + "/" + scene + ".json";
        std::map<std::string, FrameData> frames;

        if (!load_json(json_path, frames))
        {
            std::cerr << "Failed to load JSON: " << json_path << "\n";
            continue;
        }

        double total_ae = 0.0;
        double total_te = 0.0;
        double total_time = 0.0;
        int count = 0;

        json pred_poses_json;

        for (const auto &[name, frame] : frames)
        {
            auto [ae, te, time_ms, pred_pose] = evaluate_pose(frame.gt_w2c, frame.intrinsic_K, frame.p2d, frame.p3d);

            total_ae += ae;
            total_te += te;
            total_time += time_ms;
            count++;

            json mat_json = json::array();
            for (int i = 0; i < 4; ++i)
            {
                json row = json::array();
                for (int j = 0; j < 4; ++j)
                    row.push_back(pred_pose(i, j));
                mat_json.push_back(row);
            }
            pred_poses_json[name]["pred_w2c"] = mat_json;
        }

        if (count > 0)
        {
            results.push_back({scene,
                               total_ae / count,
                               total_te / count,
                               total_time / count,
                               count});

            if (save_predicted_poses)
            {
                std::string output_path = data_dir + "/" + scene + "_pred.json";
                if (!write_predicted_poses_with_progress(output_path, pred_poses_json))
                {
                    std::cerr << "Error writing predicted poses for scene " << scene << "\n";
                }
            }
            else
            {
                std::cout << "Skipping saving predicted poses to file.\n";
            }
        }
    }

    std::cout << "\n=== Evaluation Results ===\n";
    for (const auto &res : results)
    {
        std::printf("Scene: %-12s | Avg AE = %8.6f°, Avg TE = %8.6f cm, Avg Time = %8.3f ms (%4d frames)\n",
                    res.scene.c_str(), res.avg_ae, res.avg_te, res.avg_time_ms, res.frame_count);
    }

    return 0;
}

