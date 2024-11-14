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

TEST(TestLoamRegistration, TestSimpleCase) {
  Pose3d source_T_target(
      Eigen::Quaterniond(0.9993921140970299, 0.014692022378442412, 0.030140550562090015, 0.009544316157523478),
      Eigen::Vector3d(0.01, 0.03, -0.01));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-4);
}

TEST(TestLoamRegistration, TestSimpleLargeTranslation) {
  Pose3d source_T_target(
      Eigen::Quaterniond(0.9993921140970299, 0.014692022378442412, 0.030140550562090015, 0.009544316157523478),
      Eigen::Vector3d(-0.1, 0.1, 0.0));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-3);
}

TEST(TestLoamRegistration, TestSimpleEvenLargerTranslation) {
  Pose3d source_T_target(
      Eigen::Quaterniond(0.9993921140970299, 0.014692022378442412, 0.030140550562090015, 0.009544316157523478),
      Eigen::Vector3d(-0.3, 0.2, 0.1));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-3);
}

TEST(TestLoamRegistration, TestSimpleLargeRotation) {
  Eigen::Vector3d axis(1, 3, 1);
  Pose3d source_T_target(Eigen::Quaterniond(Eigen::AngleAxisd(0.2, axis / axis.norm())),
                         Eigen::Vector3d(-0.01, 0.02, 0.1));
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Run the registration
  Pose3d target_T_source = registerFeatures<ParenAccessor>(source_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-3);
}

TEST(TestLoamRegistration, TestCompositionDirection) {
  // This test was developed to ensure that the relative transform computed in each iteration of registerFeatures
  // is composed correctly with the current estimate.

  Eigen::Vector3d z_axis(0, 0, 1);
  Pose3d source_T_target(Eigen::Quaterniond(Eigen::AngleAxisd(0.1, z_axis)), Eigen::Vector3d::Zero());
  LoamFeatures<Eigen::Vector3d> target_features = constructSimpleScene();
  LoamFeatures<Eigen::Vector3d> source_features = transformFeatures(target_features, source_T_target);

  // Setup Registration Params
  Pose3d tgt_T_src_init_est(Eigen::Quaterniond(Eigen::AngleAxisd(-0.1, z_axis)), Eigen::Vector3d(0.1, 0, 0));
  RegistrationParams params;
  params.max_iterations = 1;

  // Run the registration
  Pose3d target_T_source =
      registerFeatures<ParenAccessor>(source_features, target_features, tgt_T_src_init_est, params);

  // Compute the error
  Eigen::Quaterniond err_rot = source_T_target.rotation * target_T_source.rotation;
  Eigen::Vector3d err_trans = source_T_target.rotation * target_T_source.translation + source_T_target.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-3);
}

TEST(TestLoamRegistration, NonStandardAllocator) {
  // Mainly a compile time test
  Eigen::Vector3d axis(1, 3, 1);
  LoamFeatures<Eigen::Vector3d, Eigen::aligned_allocator> target_features;
  // Plane 1 (along yz axes at x=-3)
  for (double y = 3; y < 6; y += 0.05) {
    for (double z = -1; z < 2; z += 0.05) {
      target_features.planar_points.push_back(Eigen::Vector3d(-3, y, z));
    }
  }

  // Run the registration
  Pose3d target_T_source = registerFeatures<ParenAccessor>(target_features, target_features, Pose3d());

  // Compute the error
  Eigen::Quaterniond err_rot = target_T_source.rotation;
  Eigen::Vector3d err_trans = target_T_source.translation;

  ASSERT_NEAR(err_rot.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-4);
  ASSERT_NEAR(err_trans(0), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(1), 0.0, 1e-3);
  ASSERT_NEAR(err_trans(2), 0.0, 1e-3);
}