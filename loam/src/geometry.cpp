/** @brief Implementation of LOAM Geometry Module
 * @author Dan McGann
 * @date Mar 2024
 */
#include "loam/geometry.h"

namespace loam {

/*********************************************************************************************************************/
Pose3d Pose3d::inverse() const {
  Eigen::Quaterniond inv_rot = rotation.inverse();
  return Pose3d(inv_rot, inv_rot * -translation);
}

/*********************************************************************************************************************/
Pose3d Pose3d::compose(const Pose3d &other) const {
  return Pose3d(rotation * other.rotation, translation + (rotation * other.translation));
}

/*********************************************************************************************************************/
Eigen::Vector3d Pose3d::act(const Eigen::Vector3d &p) const { return rotation * p + translation; }

/*********************************************************************************************************************/
Eigen::Matrix4d Pose3d::matrix() const {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
  mat.block<3, 3>(0, 0) = rotation.toRotationMatrix();
  mat.block<3, 1>(0, 3) = translation;
  return mat;
}

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
/*********************************************************************************************************************/
std::pair<Line, double> fitLine(Eigen::MatrixXd points) {
  assert(points.rows >= 2 && points.cols == 3);
  // Compute the mean of the points
  Eigen::Vector3d center = points.colwise().mean();
  // Normalize the points around the center
  Eigen::MatrixXd centered_points = points.rowwise() - center.transpose();
  // Run PCA on the line points
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> pca(centered_points.transpose() * centered_points);
  // Get line direction
  Eigen::Vector3d line_direction = pca.eigenvectors().col(2);
  // Construct the Line
  Line line(center + 0.1 * line_direction, center - 0.1 * line_direction);
  // Compute the condition number being careful to avoid a division by zero
  double condition_number = std::numeric_limits<double>::max();
  if (pca.eigenvalues()(2) > 1e-12) pca.eigenvalues()(2) / pca.eigenvalues()(0);

  return std::make_pair(line, condition_number);
}

/*********************************************************************************************************************/
std::pair<Plane, double> fitPlane(Eigen::MatrixXd points) {
  assert(points.rows() >= 3 && points.cols() == 3);
  // Compose the ones vector
  Eigen::VectorXd ones_vec = Eigen::VectorXd::Ones(points.rows());
  // Solve the least squares problem
  Eigen::Vector3d abc = points.colPivHouseholderQr().solve(ones_vec);
  // Convert the abc parameterization to normal, d representation
  Plane plane(abc / abc.norm(), 1.0 / abc.norm());
  // Compute the average distance to the plane
  double avg_dist = (points * plane.normal - (ones_vec * plane.d)).mean();
  return std::make_pair(plane, avg_dist);
}

}  // namespace geometry_internal

}  // namespace loam