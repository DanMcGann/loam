/** @brief This module provides all functionality to match two sets of LOAM features and register them using ICF.
 * Given a target set of LOAM features (e.g. a previous lidar scan, a local map, or a pre-computed map), a source set of
 * LOAM features, and an initial guess for the relative transform between the source and target frames the goal is to
 * compute a refined estimate of the relative pose between source and target.
 *
 * This is done with Iterative Closest Feature* (ICF) that accounts for the geometric ambiguities of the LOAM features
 * (planar points and edge points).
 *
 * Note: most sources simply refer to the LOAM optimization as Iterative Closest Point which is a misnomer given LOAM
 * performs point to feature matching NOT point to point matching.
 *
 * [1] Ji Zhang and Sanjiv Singh, "LOAM: Lidar Odometry and Mapping in Real-time,"
 *     in Proceedings of Robotics: Science and Systems, 2014.
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <nanoflann.hpp>

#include "loam/common.h"
#include "loam/features.h"
#include "loam/geometry.h"
#include "loam/kdtree.h"

namespace loam {

/**
 * ######## ##    ## ########  ########  ######
 *    ##     ##  ##  ##     ## ##       ##    ##
 *    ##      ####   ##     ## ##       ##
 *    ##       ##    ########  ######    ######
 *    ##       ##    ##        ##             ##
 *    ##       ##    ##        ##       ##    ##
 *    ##       ##    ##        ########  ######
 */
struct RegistrationParams {
  /// @brief The number of edge points to search for in the target when associating a source edge point
  // Must be >= 2, Reasonable numbers are just greater than 2
  size_t num_edge_neighbors{5};
  /// @brief The max distance to target edge points when associating a source edge point
  /// If zero no max range is used, reasonable values are ~1m for most robotic applications
  double max_edge_neighbor_dist{1.0};
  /// @brief The min number of target edge points from which we can construct a line
  /// Must be >= 2 and <= num_edge_neighbors, Reasonable numbers are just greater than 2
  size_t min_line_fit_points{3};
  /// @brief The minimum condition number of a line for the line to be considered valid
  double min_line_condition_number{10};

  /// @brief The number of planar points to search for in the target when associating a source planar point
  // Must be >= 3, Reasonable numbers are just greater than 3
  size_t num_plane_neighbors{5};
  /// @brief The max distance to target planar points when associating a source planar point
  /// If zero no max range is used, reasonable values are ~1m for most robotic applications
  double max_plane_neighbor_dist{2.0};
  /// @brief The min number of target planar points from which we can construct a plane
  /// Must be >= 3 and <= num_plane_neighbors, Reasonable numbers are just greater than 3
  size_t min_plane_fit_points{4};
  /// @brief The max average distance from component points for the plane to be considered valid
  double max_avg_point_plane_dist{0.1};

  /// @brief The maximum number of ICF iterations permitted
  /// Reasonable numbers depend on real-time and accuracy requirements of the system
  size_t max_iterations{10};
  /// @brief The convergence threshold for angular change of the pose we are estimating (radians)
  double rotation_convergence_thresh{1e-3};
  /// @brief The convergence threshold for position change of the pose we are estimating (point units)
  double position_convergence_thresh{1e-2};

  /// @brief The minimum required number of associations to attempt ICF
  size_t min_associations{100};
};

/// @brief Structure for storing detailed information about the ICF registration
/// Filled out if provided to registerFeatures
struct RegistrationDetail {
  /** TYPES* */
  /// @brief The reason that ICF terminated.
  /// ICF either converges, hits maximum iterations, or does not have enough associations to estimate the transform
  enum TerminationType { CONVERGED, MAX_ITER, INSUFFICIENT_ASSOCIATIONS };
  /// @brief Information for a single iteration of ICF
  struct IterationInfo {
    /// @brief The transform estimate at the start of the iteration
    Pose3d target_T_source_init;
    /// @brief The associations computed at this iteration for edge point as pairs of [source idx, target idx] where
    /// target idx is the index of only the closest point in the target frame
    std::vector<std::pair<size_t, size_t>> edge_associations;
    /// @brief The associations computed at this iteration for planar point as pairs of [source idx, target idx] where
    /// target idx is the index of only the closest point in the target frame
    std::vector<std::pair<size_t, size_t>> plane_associations;
    /// @brief The update to the transform estimate computed at this iteration
    Pose3d estimate_update;

    // Explicit Constructor
    IterationInfo(const Pose3d target_T_source_init, const std::vector<std::pair<size_t, size_t>> edge_associations,
                  const std::vector<std::pair<size_t, size_t>> plane_associations, const Pose3d estimate_update)
        : target_T_source_init(target_T_source_init),
          edge_associations(edge_associations),
          plane_associations(plane_associations),
          estimate_update(estimate_update) {}
  };
  /// @brief Information for each ICF iteration executed
  std::vector<IterationInfo> iteration_info;
  /// @brief The reason that ICF terminated
  TerminationType termination_type;
};

/**
 * #### ##    ## ######## ######## ########  ########    ###     ######  ########
 *  ##  ###   ##    ##    ##       ##     ## ##         ## ##   ##    ## ##
 *  ##  ####  ##    ##    ##       ##     ## ##        ##   ##  ##       ##
 *  ##  ## ## ##    ##    ######   ########  ######   ##     ## ##       ######
 *  ##  ##  ####    ##    ##       ##   ##   ##       ######### ##       ##
 *  ##  ##   ###    ##    ##       ##    ##  ##       ##     ## ##    ## ##
 * #### ##    ##    ##    ######## ##     ## ##       ##     ##  ######  ########
 */

/** @brief Registers source to target, computing the transform from source to target (target_T_source)
 * @param source: The source loam features
 * @param target: The target loam features
 * @param target_T_source_init: The initial estimate of the transform from the source frame to the target frame
 * @returns The transform from source to target (target_T_source)
 * WARN: Mutates detail if provided
 */
template <template <typename> class Accessor = FieldAccessor, typename PointType, template <typename> class  Alloc>
Pose3d registerFeatures(const LoamFeatures<PointType, Alloc>& source, const LoamFeatures<PointType, Alloc>& target,
                        const Pose3d& target_T_source_init, const RegistrationParams& params = RegistrationParams(),
                        std::shared_ptr<RegistrationDetail> detail = nullptr);

/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */
namespace registration_internal {

/// @brief Defines the cost function between a point and a line [1] Eq.(2) using ceres autodiff
class EdgeCostFunction {
  /** FIELDS */
 private:
  /// @brief The edge point in the source frame
  const Eigen::Vector3d source_pt_;
  /// @brief The line the source point is matched to in the target frame, represented as two points
  const geometry_internal::Line line_;

  /** Interface */
 public:
  /** @brief Constructor
   * @param source_pt: The point in the source frame matched with the line
   * @param tgt_line_1: The first point defining the line in the target frame
   * @param tgt_line_2: The second point defining the line in the target frame
   **/
  EdgeCostFunction(Eigen::Vector3d source_pt, geometry_internal::Line line) : source_pt_(source_pt), line_(line) {}

  /** @brief Computes the loss as the point-to-line distance
   * @param t_R_s_ptr: The current relative rotation solution target_R_source
   * @param t_p_s_ptr: The current relative position solution target_p_source
   * @param residuals_ptr: Container for the error of this loss
   */
  template <typename T>
  bool operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const;

  /// @brief Helper to return the cost function as a ceres auto diff cost function
  static ceres::CostFunction* Create(Eigen::Vector3d source_pt, geometry_internal::Line line) {
    return new ceres::AutoDiffCostFunction<EdgeCostFunction, 1, 4, 3>(new EdgeCostFunction(source_pt, line));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Defines the cost function between a point and a plane. NOTE: we do not use [1] Eq.(2)
class PlaneCostFunction {
  /** FIELDS */
 private:
  /// @brief The planar point in the source frame
  const Eigen::Matrix<double, 3, 1> source_pt_;
  /// @brief The plane the source point has been matched to
  const geometry_internal::Plane plane_;

  /** Interface */
 public:
  /** @brief Constructor
   * @param source_pt: The point in the source frame matched with the plane
   * @param origin: The origin of the plane in the target frame
   * @param normal: The normal of the plan in the target frame
   **/
  PlaneCostFunction(Eigen::Vector3d source_pt, geometry_internal::Plane plane) : source_pt_(source_pt), plane_(plane) {}

  /** @brief Computes the loss as the point-to-plane distance
   * @param t_R_s_ptr: The current relative rotation solution target_R_source
   * @param t_p_s_ptr: The current relative position solution target_p_source
   * @param residuals_ptr: Container for the error of this loss
   */
  template <typename T>
  bool operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const;

  /// @brief Helper to return the cost function as a ceres auto diff cost function
  static ceres::CostFunction* Create(Eigen::Vector3d source_pt, geometry_internal::Plane plane) {
    return new ceres::AutoDiffCostFunction<PlaneCostFunction, 1, 4, 3>(new PlaneCostFunction(source_pt, plane));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** @brief Computes associations between edge points in source and edges in target, validates the associations, and adds
 * the appropriate cost function to the ceres problem.
 * NOTE: Params as named identical to their name in the body of registerFeatures
 * WARN: Mutates problem (mutation used to avoid an extra iteration over all cost functions)
 * @returns The associations as a list of pair [src_idx, target_idx] where target_idx is the closest neighbor
 */
std::vector<std::pair<size_t, size_t>> associateEdges(const RegistrationParams& params,
                                                      const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                      const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                      const kdtree_internal::KDTree& target_edge_kdtree,
                                                      const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                      ceres::Problem& problem);

/** @brief Computes associations between planar points in source and planes in target, validates the associations, and
 * adds the appropriate cost function to the ceres problem.
 * NOTE: Params as named identical to their name in the body of registerFeatures
 * WARN: Mutates problem (mutation used to avoid an extra iteration over all cost functions)
 * @returns The associations as a list of pair [src_idx, target_idx] where target_idx is the closest neighbor
 */
std::vector<std::pair<size_t, size_t>> associatePlanes(const RegistrationParams& params,
                                                       const LoamFeatures<Eigen::Vector3d>& source_eig,
                                                       const LoamFeatures<Eigen::Vector3d>& target_eig,
                                                       const kdtree_internal::KDTree& target_plane_kdtree,
                                                       const Pose3d& target_T_source_est, Pose3d& estimate_update,
                                                       ceres::Problem& problem);
};  // namespace registration_internal

}  // namespace loam

// Include the actual implementation of this module
#include "loam/registration-inl.h"