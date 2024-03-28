/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once

#include "loam/loam_registration.h"

namespace loam {

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Accessor = FieldAccessor>
Pose3d registerFeatures(const LoamFeatures<PointType>& source, const LoamFeatures<PointType>& target,
                        const Pose3d& target_T_source_init, const RegistrationParams& params,
                        std::shared_ptr<RegistrationDetail> detail) {
  // Convert features to eigen once here to avoid repeated conversions later
  LoamFeatures<Eigen::Vector3d> source_eig = featuresToEigen<PointType, Accessor>(source);
  LoamFeatures<Eigen::Vector3d> target_eig = featuresToEigen<PointType, Accessor>(target);

  // Compute a KDtree for the target features: 20 leaf nodes is approx optimal given nanoflann's documentation
  KDTreeDataAdaptor target_edge_adaptor(target_eig.edge_points);
  KDTree target_edge_kdtree(3, target_edge_adaptor, KDTreeParams(20));
  KDTreeDataAdaptor target_plane_adaptor(target_eig.planar_points);
  KDTree target_plane_kdtree(3, target_plane_adaptor, KDTreeParams(20));

  // Setup the parameters of the optimization (the relative pose)
  Pose3d target_T_source_est(target_T_source_init);
  RegistrationDetail::TerminationType termination_type = RegistrationDetail::TerminationType::MAX_ITER;
  for (size_t iter_count = 0; iter_count < params.max_iterations; iter_count++) {
    // Define the Ceres Problem
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // Define the parameters of the system and their respective manifolds
    Pose3d estimate_update;
    problem.AddParameterBlock(estimate_update.rotation.coeffs().data(), 4, new ceres::QuaternionManifold());
    problem.AddParameterBlock(estimate_update.translation.data(), 3, new ceres::EuclideanManifold<3>());

    // Compute Associations and accumulate the optimization problem [WARN: Mutates "problem"]
    auto edge_assoc = registration_internal::associateEdges(params, source_eig, target_eig, target_edge_kdtree,
                                                            target_T_source_est, estimate_update, problem);
    auto plane_assoc = registration_internal::associatePlanes(params, source_eig, target_eig, target_plane_kdtree,
                                                              target_T_source_est, estimate_update, problem);

    if (edge_assoc.size() + plane_assoc.size() < params.min_associations) {
      termination_type = RegistrationDetail::TerminationType::INSUFFICIENT_ASSOCIATIONS;
      break;
    }

    // Solve the problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // If configured Fill out the detail
    if (detail) {
      detail->iteration_info.emplace_back(target_T_source_est, edge_assoc, plane_assoc, estimate_update);
    }

    // Update the solution
    target_T_source_est = target_T_source_est.compose(estimate_update);

    // Check for convergence - defined as when the computed update is sufficiently small
    const double angle_change = estimate_update.rotation.angularDistance(Eigen::Quaterniond::Identity());
    const double position_change = estimate_update.translation.norm();
    if (angle_change < params.rotation_convergence_thresh && position_change < params.position_convergence_thresh) {
      termination_type = RegistrationDetail::TerminationType::CONVERGED;
      break;
    }
  }
  // Fill out termination reason in detail if it exists
  if (detail) detail->termination_type = termination_type;
  return target_T_source_est;
}

/**
 * ##     ## ######## ##       ########  ######## ########   ######
 * ##     ## ##       ##       ##     ## ##       ##     ## ##    ##
 * ##     ## ##       ##       ##     ## ##       ##     ## ##
 * ######### ######   ##       ########  ######   ########   ######
 * ##     ## ##       ##       ##        ##       ##   ##         ##
 * ##     ## ##       ##       ##        ##       ##    ##  ##    ##
 * ##     ## ######## ######## ##        ######## ##     ##  ######
 */
/*********************************************************************************************************************/
namespace registration_internal {
std::vector<std::pair<size_t, size_t>> associateEdges(const RegistrationParams& params,
                                                      const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                      const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                      const KDTree& target_edge_kdtree,
                                                      const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                      ceres::Problem& problem) {
  std::vector<std::pair<size_t, size_t>> edge_associations;
  edge_associations.reserve(source_eig.edge_points.size());
  // Compute Edge Associations
  for (size_t source_idx = 0; source_idx < source_eig.edge_points.size(); source_idx++) {
    // Transform the point into the target frame using the current solution
    const Eigen::Vector3d point_tgt = target_T_source_est.act(source_eig.edge_points.at(source_idx));

    // Associate the query point with target points
    std::vector<size_t> neighbor_edge_idxes =
        knnSearch(target_edge_kdtree, point_tgt, params.num_edge_neighbors, params.max_edge_neighbor_dist);
    if (neighbor_edge_idxes.size() < params.min_line_fit_points) continue;  // GUARD: Insufficient Matches

    // Accumulate the points into a matrix
    Eigen::MatrixXd neighbor_edge_points = Eigen::MatrixXd::Zero(neighbor_edge_idxes.size(), 3);
    for (size_t i = 0; i < neighbor_edge_idxes.size(); i++) {
      neighbor_edge_points.row(i) = target_eig.edge_points.at(neighbor_edge_idxes[i]);
    }

    // Fit the Line to these points
    auto [line, condition_number] = fitLine(neighbor_edge_points);
    if (condition_number < params.min_line_condition_number) continue;  // GUARD: Edge points not co-linear

    // Construct the cost function and add it to the problem
    problem.AddResidualBlock(EdgeCostFunction::Create(point_tgt, line), new ceres::HuberLoss(1.0),
                             estimate_update.rotation.coeffs().data(), estimate_update.translation.data());
    // Accumulate the association
    edge_associations.emplace_back(source_idx, neighbor_edge_idxes.front());
  }
  return edge_associations;
}

/*********************************************************************************************************************/
std::vector<std::pair<size_t, size_t>> associatePlanes(const RegistrationParams& params,
                                                       const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                       const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                       const KDTree& target_plane_kdtree,
                                                       const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                       ceres::Problem& problem) {
  std::vector<std::pair<size_t, size_t>> plane_associations;
  plane_associations.reserve(source_eig.planar_points.size());
  for (size_t source_idx = 0; source_idx < source_eig.planar_points.size(); source_idx++) {
    // Transform the point into the target frame using the current solution
    const Eigen::Vector3d point_tgt = target_T_source_est.act(source_eig.planar_points.at(source_idx));

    // Associate the query point with target points
    std::vector<size_t> neighbor_plane_idxes =
        knnSearch(target_plane_kdtree, point_tgt, params.num_plane_neighbors, params.max_plane_neighbor_dist);
    if (neighbor_plane_idxes.size() < params.min_plane_fit_points) continue;  // GUARD: Insufficient Matches

    // Accumulate the points into a matrix
    Eigen::MatrixXd neighbor_plane_points = Eigen::MatrixXd::Zero(neighbor_plane_idxes.size(), 3);
    for (size_t i = 0; i < neighbor_plane_idxes.size(); i++) {
      neighbor_plane_points.row(i) = target_eig.planar_points.at(neighbor_plane_idxes[i]);
    }

    // Fit the Plane to these points
    auto [plane, avg_dist] = fitPlane(neighbor_plane_points);
    if (avg_dist > params.max_avg_point_plane_dist) continue;  // GUARD: Plane points not co-planar

    // Construct the cost function and add it to the problem
    problem.AddResidualBlock(PlaneCostFunction::Create(point_tgt, plane), new ceres::HuberLoss(1.0),
                             estimate_update.rotation.coeffs().data(), estimate_update.translation.data());
    // Accumulate the association
    plane_associations.emplace_back(source_idx, neighbor_plane_idxes.front());
  }
  return plane_associations;
}

}  // namespace registration_internal
}  // namespace loam