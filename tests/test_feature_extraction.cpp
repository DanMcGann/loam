#include <gtest/gtest.h>

#include "loam/common.h"
#include "loam/features.h"

using namespace loam;

// Setup type system for points
struct Point {
  double x;
  double y;
  double z;
  Point(double x, double y, double z) : x(x), y(y), z(z) {}
};

/**
 *  ######  ##     ## ########  ##     ##    ###    ######## ##     ## ########  ########
 * ##    ## ##     ## ##     ## ##     ##   ## ##      ##    ##     ## ##     ## ##
 * ##       ##     ## ##     ## ##     ##  ##   ##     ##    ##     ## ##     ## ##
 * ##       ##     ## ########  ##     ## ##     ##    ##    ##     ## ########  ######
 * ##       ##     ## ##   ##    ##   ##  #########    ##    ##     ## ##   ##   ##
 * ##    ## ##     ## ##    ##    ## ##   ##     ##    ##    ##     ## ##    ##  ##
 *  ######   #######  ##     ##    ###    ##     ##    ##     #######  ##     ## ########
 */
TEST(TestLoamFeatureExtraction, TestCurvaturePlane) {
  /** Test Environment
   *    *  *  *  *  *  *  *
   *             |
   *             | 1m
   *             |
   *             ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -5; i <= 5; i++) {
    pcd.push_back(Point(i, 1, 0.0));
  }
  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 11, /* min range */ 0.1, /* max range */ 10);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  auto curv = computeCurvature<Point>(pcd, lidar_params, params);

  // Ensure we computed curvature for all points
  ASSERT_EQ(curv.size(), 11);
  // All edge points will have invalid curvature
  for (size_t i = 0; i < 5; i++) {
    EXPECT_NEAR(curv[i].curvature, -1, 1e-9);
    EXPECT_NEAR(curv[10 - i].curvature, -1, 1e-9);
  }
  // For the single non-edge point we have a zero curvature
  EXPECT_NEAR(curv[5].curvature, 0.0, 1e-9);
}

TEST(TestLoamFeatureExtraction, TestCurvatureCorner) {
  /** Test Environment
   *    *                 *
   *       *           *
   *          *     *
   *             *
   *             |
   *             | 1m
   *             |
   *             ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -5; i <= 5; i++) {
    pcd.push_back(Point(i, abs(i) + 1, 0.0));
  }
  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 11, /* min range */ 0.1, /* max range */ 50);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<PointCurvature> curv = computeCurvature<Point>(pcd, lidar_params, params);

  // Ensure we computed curvature for all points
  ASSERT_EQ(curv.size(), 11);
  // All edge points will have zero curvature
  for (size_t i = 0; i < 5; i++) {
    EXPECT_NEAR(curv[i].curvature, -1, 1e-9);
    EXPECT_NEAR(curv[10 - i].curvature, -1, 1e-9);
  }
  // For the single non-edge point we have a large curvature
  EXPECT_NEAR(curv[5].curvature, 900.0, 1e-9);
}

/**
 * ##     ##    ###    ##       #### ########     ########   #######  #### ##    ## ########  ######
 * ##     ##   ## ##   ##        ##  ##     ##    ##     ## ##     ##  ##  ###   ##    ##    ##    ##
 * ##     ##  ##   ##  ##        ##  ##     ##    ##     ## ##     ##  ##  ####  ##    ##    ##
 * ##     ## ##     ## ##        ##  ##     ##    ########  ##     ##  ##  ## ## ##    ##     ######
 *  ##   ##  ######### ##        ##  ##     ##    ##        ##     ##  ##  ##  ####    ##          ##
 *   ## ##   ##     ## ##        ##  ##     ##    ##        ##     ##  ##  ##   ###    ##    ##    ##
 *    ###    ##     ## ######## #### ########     ##         #######  #### ##    ##    ##     ######
 */

TEST(TestValidPoints, TestInvalidEdges) {
  /** Test Environment
   *   * * * * * * * * * * *
   *             |
   *             | 1m
   *             |
   *             ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -5; i <= 5; i++) {
    pcd.push_back(Point(i * 0.1, 1, 0.0));
  }
  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 11, /* min range */ 0.1, /* max range */ 50);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 11);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[10 - i]);
  }
  // Non edge point is valid
  EXPECT_TRUE(valid_mask[5]);
}

TEST(TestValidPoints, TestInvalidRanges) {
  /** Test Environment
   *         *
   *
   *   *****        *****
   *
   *
   *             *
   *             ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -5; i < 0; i++) pcd.push_back(Point(i, 1, 0.0));
  pcd.push_back(Point(-0.5, 20.0, 0.0));  // TOO FAR
  pcd.push_back(Point(0.0, 0.2, 0.0));    // TOO CLOSE
  for (int i = 1; i <= 5; i++) pcd.push_back(Point(i, 1, 0.0));

  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 12, /* min range */ 0.5, /* max range */ 6.0);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 12);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[10 - i]);
  }
  // Check test candidates
  EXPECT_FALSE(valid_mask[5]);  // TOO FAR
  EXPECT_FALSE(valid_mask[6]);  // TOO CLOSE
}

TEST(TestValidPoints, TestOcclusionCase1) {
  /** Test Environment
   * -> Scan Dir
   *         ********
   *
   *  *******
   *
   *
   *         ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -15; i < 0; i++) pcd.push_back(Point(i * 0.1, 4.0, 0.0));
  for (int i = 0; i < 15; i++) pcd.push_back(Point(i * 0.1, 6.0, 0.0));

  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 30, /* min range */ 0.1, /* max range */ 100);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 30);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[29 - i]);
  }

  // Close Points are all valid
  for (size_t i = 5; i < 15; i++) EXPECT_TRUE(valid_mask[i]);
  // Occluded Points are not valid
  for (size_t i = 15; i < 20; i++) EXPECT_FALSE(valid_mask[i]);
  // Planar points far enough away from occlusion are valid
  for (size_t i = 20; i < 25; i++) EXPECT_TRUE(valid_mask[i]);
}

TEST(TestValidPoints, TestOcclusionCase2) {
  /** Test Environment
   * -> Scan Dir
   *  *******
   *
   *         ********
   *
   *
   *         ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -15; i < 0; i++) pcd.push_back(Point(i * 0.1, 6.0, 0.0));
  for (int i = 0; i < 15; i++) pcd.push_back(Point(i * 0.1, 4.0, 0.0));

  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 30, /* min range */ 0.1, /* max range */ 100);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 30);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[29 - i]);
  }

  // Planar points far enough away from occlusion are valid
  for (size_t i = 5; i < 10; i++) EXPECT_TRUE(valid_mask[i]);
  // Occluded Points are not valid
  for (size_t i = 10; i < 15; i++) EXPECT_FALSE(valid_mask[i]);
  // Far Points are all valid
  for (size_t i = 15; i < 25; i++) EXPECT_TRUE(valid_mask[i]);
}

TEST(TestValidPoints, TestParallelPlaneCase1) {
  /** Test Environment
   * -> Scan Dir
   *          ********
   *         *
   *  *******
   *
   *
   *         ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -15; i < 0; i++) pcd.push_back(Point(i * 0.1, 2.0, 0.0));
  pcd.push_back(Point(0, 0, 2.05));
  for (int i = 1; i <= 15; i++) pcd.push_back(Point(i * 0.1, 2.1, 0.0));

  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 31, /* min range */ 0.1, /* max range */ 100);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 31);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[30 - i]);
  }

  // Planar left points are all good
  for (size_t i = 5; i < 15; i++) EXPECT_TRUE(valid_mask[i]);
  // Planar right points are all good
  for (size_t i = 16; i < 26; i++) EXPECT_TRUE(valid_mask[i]);

  // Near parallel point is not good
  EXPECT_FALSE(valid_mask[15]);
}

TEST(TestValidPoints, TestParallelPlaneCase2) {
  /** Test Environment
   * -> Scan Dir
   *  *******
   *         *
   *          ********
   *
   *
   *         ^Lidar
   */
  std::vector<Point> pcd;
  for (int i = -15; i < 0; i++) pcd.push_back(Point(i * 0.1, 2.1, 0.0));
  pcd.push_back(Point(0, 0, 2.05));
  for (int i = 1; i <= 15; i++) pcd.push_back(Point(i * 0.1, 2.0, 0.0));

  LidarParams lidar_params(/* scan_lines */ 1, /* pts/line */ 31, /* min range */ 0.1, /* max range */ 100);
  FeatureExtractionParams params{5, 6, 5, 5, 100, 0.1, 0.25, 0.02};

  std::vector<bool> valid_mask = computeValidPoints<Point>(pcd, lidar_params, params);

  // Ensure we computed mask for all points
  ASSERT_EQ(valid_mask.size(), 31);
  // All edge points are invalid
  for (size_t i = 0; i < 5; i++) {
    EXPECT_FALSE(valid_mask[i]);
    EXPECT_FALSE(valid_mask[30 - i]);
  }

  // Planar left points are all good
  for (size_t i = 5; i < 15; i++) EXPECT_TRUE(valid_mask[i]);
  // Planar right points are all good
  for (size_t i = 16; i < 26; i++) EXPECT_TRUE(valid_mask[i]);

  // Near parallel point is not good
  EXPECT_FALSE(valid_mask[15]);
}