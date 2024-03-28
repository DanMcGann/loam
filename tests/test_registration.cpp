#include <gtest/gtest.h>

#include "Eigen/Dense"
#include "loam/loam.h"

using namespace loam;

LoamFeatures<Eigen::Vector3d> constructSimpleScene() {
  /** SimpleTest Environment [Top Down]
   *      X    ────────────
   *   |                    X
   *   |            █████
   *   |            █████
   *   |
   *
   *  y          ^Lidar
   *  L x
   *
   * X = Column of corners
   * | + ── = Vertical Plane
   * █ = Horizontal plane
   */
  LoamFeatures<Eigen::Vector3d> result;

  // Plane 1 (along yz axes at x=-3)
  for (double y = 3; y < 6; y += 0.05) {
    for (double z = -1; z < 2; z += 0.05) {
      result.planar_points.push_back(Eigen::Vector3d(-3, y, z));
    }
  }

  // Plane 2 (along xz axis at y=5)
  for (double x = -1; x < 2; x += 0.05) {
    for (double z = -1; z < 2; z += 0.05) {
      result.planar_points.push_back(Eigen::Vector3d(x, 5, z));
    }
  }

  // Plane 2 (along xy axis at z=-1)
  for (double x = 1; x < 3; x += 0.05) {
    for (double y = 1; y < 3; y += 0.05) {
      result.planar_points.push_back(Eigen::Vector3d(x, y, -1));
    }
  }

  // Edge 1 vertical at xy=[-1, 4]
  for (double z = -1; z < 3; z += 0.05) {
    result.edge_points.push_back(Eigen::Vector3d(-1, 4, z));
  }

  // Edge 2 vertical at xy=[-3, 2]
  for (double z = -1; z < 3; z += 0.05) {
    result.edge_points.push_back(Eigen::Vector3d(3, 2, z));
  }

  std::cout << "edge_points=[";
  for (auto p : result.edge_points) std::cout << "np.array([" << p(0) << "," << p(1) << "," << p(2) << "]),";
  std::cout << "]" << std::endl << std::endl;

  std::cout << "planar_points=[";
  for (auto p : result.planar_points) std::cout << "np.array([" << p(0) << "," << p(1) << "," << p(2) << "]),";
  std::cout << "]" << std::endl << std::endl;

  return result;
}

LoamFeatures<Eigen::Vector3d> transformFeatures(LoamFeatures<Eigen::Vector3d> in_features, Pose3d transform) {
  LoamFeatures<Eigen::Vector3d> result;
  for (auto pp : in_features.planar_points) {
    result.planar_points.push_back(transform.rotation * pp + transform.translation);
  }
  for (auto pp : in_features.edge_points) {
    result.edge_points.push_back(transform.rotation * pp + transform.translation);
  }
  return result;
}
/**
TEST(TestLoamRegistration, TestSimpleCase) {
  Pose3d source_T_target(
      Eigen::Quaterniond(0.9993921140970299, 0.014692022378442412, 0.030140550562090015, 0.009544316157523478),
      Eigen::Vector3d(0.01, 0.03, -0.01));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<Eigen::Vector3d, ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-4);
}
**/

TEST(TestLoamRegistration, TestSimpleLargeTranslation) {
  Pose3d source_T_target(
      Eigen::Quaterniond(0.9993921140970299, 0.014692022378442412, 0.030140550562090015, 0.009544316157523478),
      Eigen::Vector3d(-0.1, 0.1, 0.0));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<Eigen::Vector3d, ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-4);
}