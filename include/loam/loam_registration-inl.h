/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <iomanip>  // TODO remove

#include "loam/geometry.h"
#include "loam/kdtree.h"
#include "loam/loam_loss_functions.h"
#include "loam/loam_registration.h"

namespace loam {

template <typename PointType, template <typename> class Accessor = FieldAccessor>
Pose3d registerFeatures(const LoamFeatures<PointType>& source, const LoamFeatures<PointType>& target,
                        const Pose3d& target_T_source_init, const RegistrationParams& params) {
  // Compute a KDtree for the target features: 20 leaf nodes is approx optimal given nanoflann's documentation

  // TODO (dan) Convert pointclouds to eigen before constructing trees so we only need convert once not multiple times
  KDTreeDataAdaptor<PointType, Accessor> target_edge_adaptor(target.edge_points);
  KDTree<PointType, Accessor> target_edge_kdtree(3, target_edge_adaptor, KDTreeParams(20));
  KDTreeDataAdaptor<PointType, Accessor> target_plane_adaptor(target.planar_points);
  KDTree<PointType, Accessor> target_plane_kdtree(3, target_plane_adaptor, KDTreeParams(20));

  // Setup the parameters of the optimization (the relative pose)
  Pose3d prev_est(target_T_source_init);
  Pose3d target_T_source_est(target_T_source_init);
  target_T_source_est.print();
  std::cout << std::endl;
  for (size_t iter_count = 0; iter_count < params.max_iterations; iter_count++) {
    target_T_source_est.print();
    // Define the Ceres Problem
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // Define the parameters of the system and their respective manifolds
    Pose3d rel_est; // TODO (dan) cleanup comments, finalize the use of rel_est, add summary returns
    problem.AddParameterBlock(rel_est.rotation.coeffs().data(), 4, new ceres::QuaternionManifold());
    problem.AddParameterBlock(rel_est.translation.data(), 3, new ceres::EuclideanManifold<3>());

    // Compute Edge Associations
    size_t pt_idx = 0;
    std::cout << "edge_point_correspondences = [";
    for (const PointType& point_src : source.edge_points) {
      // TODO (convert to eigen once to avoid doing it many times here)
      const Eigen::Vector3d eig_point_src = pointToEigen<PointType, Accessor>(point_src);
      const Eigen::Vector3d eig_point_tgt = target_T_source_est.act(eig_point_src);

      // Associate the query point with target points
      std::vector<size_t> neighbor_edge_idxes =
          knnSearch(target_edge_kdtree, eig_point_tgt, params.num_edge_neighbors, params.max_edge_neighbor_dist);
      if (neighbor_edge_idxes.size() < params.min_line_fit_points) continue;  // GUARD: Insufficient Matches

      // Accumulate the points into a matrix
      Eigen::MatrixXd neighbor_edge_points = Eigen::MatrixXd::Zero(neighbor_edge_idxes.size(), 3);
      for (size_t i = 0; i < neighbor_edge_idxes.size(); i++) {
        neighbor_edge_points.row(i) = pointToEigen<PointType, Accessor>(target.edge_points.at(neighbor_edge_idxes[i]));
      }

      // Fit the Line to these points
      auto [line, condition_number] = fitLine(neighbor_edge_points);
      if (condition_number < params.min_line_condition_number) continue;  // GUARD: Edge points not co-linear

      // std::cout << "np.array([" << pt_idx++ << "," << neighbor_edge_idxes[0] << "]),";

      // Construct the cost function and add it to the problem
      problem.AddResidualBlock(EdgeCostFunction::Create(eig_point_tgt, line), new ceres::HuberLoss(1.0),
                               rel_est.rotation.coeffs().data(), rel_est.translation.data());
    }
    std::cout << "]" << std::endl << std::endl;

    // Compute Edge Associations
    pt_idx = 0;
    std::cout << "plane_point_correspondences = [";
    for (const PointType& point_src : source.planar_points) {
      const Eigen::Vector3d eig_point_src = pointToEigen<PointType, Accessor>(point_src);
      const Eigen::Vector3d eig_point_tgt = target_T_source_est.act(eig_point_src);

      // Associate the query point with target points
      std::vector<size_t> neighbor_plane_idxes =
          knnSearch(target_plane_kdtree, eig_point_tgt, params.num_plane_neighbors, params.max_plane_neighbor_dist);
      if (neighbor_plane_idxes.size() < params.min_plane_fit_points) continue;  // GUARD: Insufficient Matches

      // Accumulate the points into a matrix
      Eigen::MatrixXd neighbor_plane_points = Eigen::MatrixXd::Zero(neighbor_plane_idxes.size(), 3);
      for (size_t i = 0; i < neighbor_plane_idxes.size(); i++) {
        neighbor_plane_points.row(i) =
            pointToEigen<PointType, Accessor>(target.planar_points.at(neighbor_plane_idxes[i]));
      }

      // Fit the Plane to these points
      auto [plane, avg_dist] = fitPlane(neighbor_plane_points);
      // std::cout << std::fixed << std::setprecision(2) << "Plane: " << plane.normal(0) << ", " << plane.normal(1) <<
      // ", "
      //           << plane.normal(2) << std::endl;
      if (avg_dist > params.max_avg_point_plane_dist) continue;  // GUARD: Plane points not co-planar
      // std::cout << "np.array([" << pt_idx++ << "," << neighbor_plane_idxes[0] << "]),";

      // Construct the cost function and add it to the problem
      problem.AddResidualBlock(PlaneCostFunction::Create(eig_point_tgt, plane), new ceres::HuberLoss(1.0),
                               rel_est.rotation.coeffs().data(), rel_est.translation.data());
    }
    std::cout << "]" << std::endl << std::endl;
    // Solve the problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 2;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    target_T_source_est = target_T_source_est.compose(rel_est);
    const double angle_change = rel_est.rotation.angularDistance(Eigen::Quaterniond::Identity());
    const double position_change = rel_est.translation.norm();
    if (angle_change < params.rotation_convergence_thresh && position_change < params.position_convergence_thresh) {
      break;
    } else {
      prev_est = Pose3d(target_T_source_est);
    }
  }

  return target_T_source_est;
}

}  // namespace loam