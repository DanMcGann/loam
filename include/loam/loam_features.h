/** @brief This module provides all functionality to extract LOAM features.
 * LOAM utilizes two types of features for data association and scan registration
 * 1) Planar Features
 * 2) Edge Features
 * Features are identified based on "curvature" [1] Eq.(1) accounting for unreliable edge cases like planes parallel to
 * the LiDAR beam and points bordering potentially occluded regions  [1] Fig. 4
 *
 * [1] Ji Zhang and Sanjiv Singh, "LOAM: Lidar Odometry and Mapping in Real-time,"
 *     in Proceedings of Robotics: Science and Systems, 2014.
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <vector>

#include "loam/loam_common.h"

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

/// @brief Structure for storing feature extraction parameters
struct FeatureExtractionParams {
  /// @brief The number of neighbor points (on either size) to use when computing curvature [1] Eq. (1)
  /// A reasonable number is between 3 and 6, less and curvature is too noisy, more and points cover too large an area
  size_t neighbor_points{5};
  /// @brief The number of sectors to break each scan line into when detecting feature points
  /// A reasonable number is between 4 and 8
  // If the number of points per line is not divisible by number_sectors, remainder points are added to the last sector
  size_t number_sectors{6};
  /// @brief The maximum number of edge features to detect in each sector
  /// Reasonable numbers depends on compute power available for registration, with more points registration is expensive
  size_t max_edge_feats_per_sector{5};
  /// @brief The maximum number of planar features to detect in each sector
  /// Reasonable numbers depends on compute power available for registration, with more points registration is expensive
  size_t max_planar_feats_per_sector{5};
  /// @brief Threshold for edge feature curvature.
  /// The UNNORMALIZED curvature must be greater than this thresh to be considered an edge feature.
  /// WARN: This is an unintuitive param manual tuning and plotting results is recommended
  double edge_feat_threshold{100.0};
  /// @brief Threshold for planar feature curvature.
  /// The UNNORMALIZED curvature must be less than this thresh to be considered a planar feature.
  /// WARN: This is an unintuitive param manual tuning and plotting results is recommended
  double planar_feat_threshold{0.1};
  /// @brief This distance in point units (e.x. meters) between neighboring points to be considered for occlusion
  /// Reasonable values are on the order of 1m for most robotics applications
  double occlusion_thresh{0.25};
  /// @brief The range difference (as proportion of range) between consecutive points to be considered too parallel
  /// See computeValidPoints::Check 4 for details
  /// WARN: This is an unintuitive param manual tuning and plotting results is recommended
  double parallel_thresh{0.002};
};

/// @brief Structure for storing edge and planar feature points from a scan together
/// @tparam PointType Template for point type see README.md
template <typename PointType>
struct LoamFeatures {
  /// @brief A pointcloud of edge feature points
  std::vector<PointType> edge_points;
  /// @brief A pointcloud of planar feature points
  std::vector<PointType> planar_points;
};

/// @brief Structure for storing curvature information for points
struct PointCurvature {
  /// @brief The index of the point
  size_t index;
  /// @brief The curvature of the point
  double curvature;
  /// @brief Explicit parameterized constructor
  PointCurvature(size_t index, double curvature) : index(index), curvature(curvature) {}
  /// @brief Default constructor
  PointCurvature() = default;
};

/// @brief Comparator for PointCurvature used in std algorithms (like sort)
bool curvatureComparator(const PointCurvature& lhs, const PointCurvature& rhs) { return lhs.curvature < rhs.curvature; }

/**
 * #### ##    ## ######## ######## ########  ########    ###     ######  ########
 *  ##  ###   ##    ##    ##       ##     ## ##         ## ##   ##    ## ##
 *  ##  ####  ##    ##    ##       ##     ## ##        ##   ##  ##       ##
 *  ##  ## ## ##    ##    ######   ########  ######   ##     ## ##       ######
 *  ##  ##  ####    ##    ##       ##   ##   ##       ######### ##       ##
 *  ##  ##   ###    ##    ##       ##    ##  ##       ##     ## ##    ## ##
 * #### ##    ##    ##    ######## ##     ## ##       ##     ##  ######  ########
 */

/** @brief Extracts and returns LOAM features from a LiDAR scan. Main entry point for using this module.
 * @param input_scan: The LiDAR Scan
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 */
template <typename PointType, template <typename> class Accessor = FieldAccessor>
LoamFeatures<PointType> extractFeatures(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                        const FeatureExtractionParams& params = FeatureExtractionParams());

/** @brief Computes the curvature [1] Eq. (1) of each point in the given LiDAR scan
 * @param input_scan: The LiDAR scan
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 * @WARN Assumes that │S│ = 10 in Eq. (1) (i.e. search 5 points on either side)
 * @WARN To match the published LOAM implementation we do not normalize curvature
 */
template <typename PointType, template <typename> class Accessor = FieldAccessor>
std::vector<PointCurvature> computeCurvature(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                             const FeatureExtractionParams& params = FeatureExtractionParams());

/** @brief Computes all valid points in the LiDAR scan [1] Sec. V.A
 * A point can be deemed invalid for 4 reasons
 * 1. The point is at the edge of a scan line
 *    - These points have invalid curvature of 0 - see compute curvature
 * 2. The point is outside the valid range of the lidar
 *    - These points also invalidate their neighbors as their neighbors curvature will be invalid
 * 3. The point is part of a probably occluded object: There are two cases for this
 *    - Case 1: The current point is a corner occluding something behind
 *        -> i+1 and neighbors to right are invalid
 *        -> The current point is valid since it is probably a corner
 *      ┌─────────────────────────────┐
 *      │───────► Scan Direction      │
 *      │                             │
 *      │  00000000000************    │
 *      │             i+1             │
 *      │                             │
 *      │            i     *=Scan     │
 *      │ ************     0=Occluded │
 *      │                             │
 *      │                             │
 *      │              ^Lidar         │
 *      └─────────────────────────────┘
 *    - Case 2: The current point is part of an occluded object
 *        -> i and neighbors to left are invalid
 *        -> i+1 is valid since it is probably a corner
 *      ┌─────────────────────────────┐
 *      │───────► Scan Direction      │
 *      │                             │
 *      │   ************00000000000   │
 *      │              i              │
 *      │                             │
 *      │ *=Scan        ************  │
 *      │ 0=Occluded    i+1           │
 *      │                             │
 *      │                             │
 *      │              ^Lidar         │
 *      └─────────────────────────────┘
 * 4. The point is on a plane nearly parallel to the LiDAR Beam
 * @param input_scan: The LiDAR scan
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 */
template <typename PointType, template <typename> class Accessor = FieldAccessor>
std::vector<bool> computeValidPoints(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                     const FeatureExtractionParams& params = FeatureExtractionParams());

/// @brief Converts Loam Features to Eigen [used in registration for efficiency]
template <typename PointType, template <typename> class Accessor = FieldAccessor>
LoamFeatures<Eigen::Vector3d> featuresToEigen(const LoamFeatures<PointType>& in_features) {
  LoamFeatures<Eigen::Vector3d> result;
  for (const PointType pt : in_features.edge_points) {
    result.edge_points.push_back(pointToEigen<PointType, Accessor>(pt));
  }
  for (const PointType pt : in_features.planar_points) {
    result.planar_points.push_back(pointToEigen<PointType, Accessor>(pt));
  }
  return result;
}

}  // namespace loam

// Include the actual implementation of this module
#include "loam/loam_features-inl.h"