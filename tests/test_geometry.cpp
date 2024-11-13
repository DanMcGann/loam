#include <gtest/gtest.h>

#include "Eigen/Dense"
#include "loam/loam.h"

using namespace loam;

/**
 * ########   #######   ######  ########
 * ##     ## ##     ## ##    ## ##
 * ##     ## ##     ## ##       ##
 * ########  ##     ##  ######  ######
 * ##        ##     ##       ## ##
 * ##        ##     ## ##    ## ##
 * ##         #######   ######  ########
 */
TEST(TestPose3d, TestCopyConstructor) {
  Pose3d pa;
  Pose3d pb(pa);

  pa.rotation.x() = 1;
  pa.translation(0) = 1;

  EXPECT_NEAR(pb.rotation.x(), 0.0, 1e-12);
  EXPECT_NEAR(pb.translation(0), 0.0, 1e-12);

  EXPECT_NEAR(pa.rotation.x(), 1.0, 1e-12);
  EXPECT_NEAR(pa.translation(0), 1.0, 1e-12);
}

TEST(TestPose3d, TestCompose) {
  // Expected result (generated using GTSAM's geometry library)
  Eigen::Vector3d expected_t(8.495192564434786 , -1.0236436475358897, -6.2354567213734695);
  Eigen::Quaterniond expected_q(0.7567645973045605, 0.5634200098313201, -0.3314580747140943, 0.0008840220400626);

  // Setup
  Eigen::Quaterniond q1(0.7473257838894183, 0.38405116269438366, -0.17015746936361906, -0.5148352287741462);
  Eigen::Quaterniond q2(0.8378767472656409, -0.040374739652255895, -0.40934599608063865, 0.3588429911288663);
  Eigen::Vector3d t1(-0.4, 3., -8.9);
  Eigen::Vector3d t2(4, -5, 1);

  Pose3d p1(q1, t1);
  Pose3d p2(q2, t2);

  Pose3d comp = p1.compose(p2);
  EXPECT_TRUE(comp.rotation.isApprox(expected_q, 1e-8));
  EXPECT_TRUE(comp.translation.isApprox(expected_t, 1e-8));
}

TEST(TestPose3d, TestInverse) {
  // Expected result (generated using GTSAM's geometry library)
  Eigen::Vector3d expected_t(1.60941772, 6.39896027, 6.69575105);
  Eigen::Quaterniond expected_q(0.7473257838894183, -0.38405116269438366, 0.17015746936361906, 0.5148352287741462);

  // Setup
  Eigen::Quaterniond q1(0.7473257838894183, 0.38405116269438366, -0.17015746936361906, -0.5148352287741462);
  Eigen::Vector3d t1(-0.4, 3., -8.9);

  Pose3d p1(q1, t1);
  Pose3d p1_inv = p1.inverse();
  EXPECT_TRUE(p1_inv.rotation.isApprox(expected_q, 1e-8));
  EXPECT_TRUE(p1_inv.translation.isApprox(expected_t, 1e-8));
}

TEST(TestPose3d, TestMatrix) {
  // Expected result (generated using GTSAM's geometry library)
  Eigen::Matrix4d expected;
  expected << 0.87992318, -0.360299, 0.30970927, 1.,  //
      0.37202555, 0.92794845, 0.0225534, -5.,         //
      -0.29552021, 0.09537451, 0.95056379, 2.,        //
      0., 0., 0., 1.;
  // Setup
  Eigen::Quaterniond q1(0.9693342323515085, 0.018781217536151106, 0.15609411554196426, 0.18887307630401792);
  Eigen::Vector3d t1(1., -5., 2.);

  Pose3d p1(q1, t1);
  Eigen::Matrix4d mat = p1.matrix();
  EXPECT_TRUE(mat.isApprox(expected, 1e-6));
}

/**
 * ########  ####  ######  ########    ###    ##    ##  ######  ########  ######
 * ##     ##  ##  ##    ##    ##      ## ##   ###   ## ##    ## ##       ##    ##
 * ##     ##  ##  ##          ##     ##   ##  ####  ## ##       ##       ##
 * ##     ##  ##   ######     ##    ##     ## ## ## ## ##       ######    ######
 * ##     ##  ##        ##    ##    ######### ##  #### ##       ##             ##
 * ##     ##  ##  ##    ##    ##    ##     ## ##   ### ##    ## ##       ##    ##
 * ########  ####  ######     ##    ##     ## ##    ##  ######  ########  ######
 */

TEST(TestDistances, TestPoint2Line) {
  Eigen::Vector3d la(0, 0, 0);
  Eigen::Vector3d lb(0, 0, 1);

  for (double x = -5; x < 5; x += 0.5) {
    for (double y = -5; y < 5; y += 0.5) {
      Eigen::Vector3d p(x, y, x + y);
      EXPECT_NEAR(geometry_internal::pointToLineDistance(p, la, lb), sqrt(x * x + y * y), 1e-8);
    }
  }
}

TEST(TestDistances, TestPoint2Plane) {
  // y, z plane +1m from origin
  Eigen::Vector3d normal(1, 0, 0);
  double distance = 2.25;

  for (double x = -5; x < 5; x += 0.5) {
    for (double y = -5; y < 5; y += 0.5) {
      Eigen::Vector3d p(x, y, x + y);
      EXPECT_NEAR(geometry_internal::pointToPlaneDistance(p, normal, distance), abs(x - 2.25), 1e-8);
    }
  }
}