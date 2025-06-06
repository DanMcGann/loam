/** @brief This module implements the geometry concepts used in LOAM.
 * Namely, Line and plane fitting which are used during registration.
 *
 *
 * [1] Ji Zhang and Sanjiv Singh, "LOAM: Lidar Odometry and Mapping in Real-time,"
 *     in Proceedings of Robotics: Science and Systems, 2014.
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <Eigen/Dense>
#include <iostream>
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

/// @brief Represents a pose in 3d space
struct Pose3d {
  /** FIELDS **/
  Eigen::Quaterniond rotation;
  Eigen::Vector3d translation;

  /** Interface **/
  /// @brief Explicit parameter constructor
  Pose3d(Eigen::Quaterniond rot, Eigen::Vector3d trans) : rotation(rot), translation(trans) {}
  /// @brief Default Constructor
  Pose3d() : rotation(Eigen::Quaterniond::Identity()), translation(Eigen::Vector3d::Zero()) {}
  /// @brief Static Identity Constructor
  static Pose3d Identity() { return Pose3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()); }

  /// @brief Returns in inverse of the pose [P^{-1}]
  Pose3d inverse() const;
  /// @brief Composes two poses [P \oplus other]
  Pose3d compose(const Pose3d &other) const;
  /// @brief A pose e_T_s acts on a point p_s according to [p_e = e_T_s * p_s]
  Eigen::Vector3d act(const Eigen::Vector3d &p) const;
  /// @brief Converts the pose into a 4x4 transformation matrix
  Eigen::Matrix4d matrix() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */
namespace geometry_internal {
/** @brief A Line in 3D space.
 * We represent lines by two points on that line as it makes it easier when computing point-to-line distances.
 * These points cannot be the same, and for numeric stability should be reasonable far away >0.1
 */
struct Line {
  /// @brief Point a that falls on the line
  const Eigen::Vector3d a;
  /// @brief Point b that falls on the line
  const Eigen::Vector3d b;

  ///@brief Explicit parameter constructor
  Line(Eigen::Vector3d a, Eigen::Vector3d b) : a(a), b(b) {}
};

/** @brief A Plane in 3d Space
 * We represent planes by parameters [n, d] such that for a point [p] on the plane n.dot(p) - d = 0
 * n is the normal of the plane and d represents the distance of the plane to the origin
 */
struct Plane {
  /// @brief The normal of the plane
  const Eigen::Vector3d normal;
  /// @brief The distance between the origin and the plane
  const double d;

  ///@brief Explicit parameter constructor
  Plane(Eigen::Vector3d normal, double d) : normal(normal), d(d) {}
};

/** @brief Fits a line to a set of points
 * To do so we perform principal component analysis on the points.
 * First we normalize the points.
 * Next we compute the covariance of points: sum x.T * x forall x in points.
 * Then we perform eigenvalue decomposition on this covariance matrix.
 * The eigen vector corresponding to the largest eigenvalue is the dominant direction of the line.
 *
 * The condition number (largest eigenvalue / smallest eigenvalue)
 * provides a measure of how co-linear these points are with larger being more co-linear
 * @param points: Matrix of points in shape (K, 3) where K must be >= 2
 * @returns The line and the condition number
 */
std::pair<Line, double> fitLine(Eigen::MatrixXd points);

///@brief Computes the distance between a point and a line defined by two points(a and b)
/// Templated, and not using Line class to work with ceres and its autodiff functionality
template <typename T>
T pointToLineDistance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &line_a,
                      const Eigen::Matrix<T, 3, 1> &line_b);

/** @brief Fits a Plane to a set of points
 * We do so by solving a least squares problem of the form
 *            points * [a, b, c].T = [1]
 * This problem assumes that the plane does not pass through the origin.
 * This assumption is okay in LOAM as any plane that passes through origin will be parallel to the LiDAR beams.
 * Such planes are avoided due to inaccurate point returns, and explicitly avoided in feature extraction.
 *
 * We also compute the average distance between the plane and the provided points to quantify how planar the input
 * points are.
 *
 * @param points: Matrix of points in shape (K, 3) where K must be >= 3
 * @returns The Plane and the average point distance
 */
std::pair<Plane, double> fitPlane(Eigen::MatrixXd points);

///@brief Computes the distance between a point and a plane defined by a normal and a distance
/// Templated, and not using Plane class to work with ceres and its autodiff functionality
template <typename T>
T pointToPlaneDistance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &normal, const T distance);

}  // namespace geometry_internal

}  // namespace loam

// Include the actual implementation
#include "loam/geometry-inl.h"