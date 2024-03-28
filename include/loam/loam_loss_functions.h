/** @brief This module defines the loss functions used in ICF registration of LOAM feature points.
 *
 * We use ceres AutoDiffCostFunctions to implement the ICF optimization problem
 *
 * [1] Ji Zhang and Sanjiv Singh, "LOAM: Lidar Odometry and Mapping in Real-time,"
 *     in Proceedings of Robotics: Science and Systems, 2014.
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <ceres/autodiff_cost_function.h>

#include <Eigen/Dense>

namespace loam {

/// @brief Defines the cost function between a point and a line [1] Eq.(2) using ceres autodiff
class EdgeCostFunction {
  /** FIELDS */
 private:
  /// @brief The edge point in the source frame
  const Eigen::Vector3d source_pt_;
  /// @brief The line the source point is matched to in the target frame, represented as two points
  const Line line_;

  /** Interface */
 public:
  /** @brief Constructor
   * @param source_pt: The point in the source frame matched with the line
   * @param tgt_line_1: The first point defining the line in the target frame
   * @param tgt_line_2: The second point defining the line in the target frame
   **/
  EdgeCostFunction(Eigen::Vector3d source_pt, Line line) : source_pt_(source_pt), line_(line) {}

  /** @brief Computes the loss as the point-to-line distance
   * @param t_R_s_ptr: The current relative rotation solution target_R_source
   * @param t_p_s_ptr: The current relative position solution target_p_source
   * @param residuals_ptr: Container for the error of this loss
   */
  template <typename T>
  bool operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

    // Transform the source point into the target frame given the current estimate
    const Eigen::Matrix<T, 3, 1> target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;

    // Compute the loss
    residuals_ptr[0] = pointToLineDistance<T>(target_pt_, line_.a.cast<T>(), line_.b.cast<T>());
    return true;
  }

  /// @brief Helper to return the cost function as a ceres auto diff cost function
  static ceres::CostFunction* Create(Eigen::Vector3d source_pt, Line line) {
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
  const Plane plane_;

  /** Interface */
 public:
  /** @brief Constructor
   * @param source_pt: The point in the source frame matched with the plane
   * @param origin: The origin of the plane in the target frame
   * @param normal: The normal of the plan in the target frame
   **/
  PlaneCostFunction(Eigen::Vector3d source_pt, Plane plane)
      : source_pt_(source_pt), plane_(plane) {}

  /** @brief Computes the loss as the point-to-plane distance
   * @param t_R_s_ptr: The current relative rotation solution target_R_source
   * @param t_p_s_ptr: The current relative position solution target_p_source
   * @param residuals_ptr: Container for the error of this loss
   */
  template <typename T>
  bool operator()(const T* const t_R_s_ptr, const T* const t_p_s_ptr, T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> t_R_s(t_R_s_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_p_s(t_p_s_ptr);

    // Transform the source point into the target frame given the current estimate
    const Eigen::Matrix<T, 3, 1> target_pt_ = t_R_s * source_pt_.cast<T>() + t_p_s;

    // Compute the loss
    residuals_ptr[0] = pointToPlaneDistance<T>(target_pt_, plane_.normal.cast<T>(), T(plane_.d));
    return true;
  }

  /// @brief Helper to return the cost function as a ceres auto diff cost function
  static ceres::CostFunction* Create(Eigen::Vector3d source_pt, Plane plane) {
    return new ceres::AutoDiffCostFunction<PlaneCostFunction, 1, 4, 3>(new PlaneCostFunction(source_pt, plane));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace loam