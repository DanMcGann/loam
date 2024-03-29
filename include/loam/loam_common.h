/** @brief The purpose of this file is to define common types and functions used throughout the loam implementation
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <exception>
#include <sstream>
#include <vector>

namespace loam {

/**
 * ##       #### ########     ###    ########
 * ##        ##  ##     ##   ## ##   ##     ##
 * ##        ##  ##     ##  ##   ##  ##     ##
 * ##        ##  ##     ## ##     ## ########
 * ##        ##  ##     ## ######### ##   ##
 * ##        ##  ##     ## ##     ## ##    ##
 * ######## #### ########  ##     ## ##     ##
 */

/** @brief Structure to store intrinsic LiDAR parameters
 * LOAM uses the structure of how LiDARs function to help in feature extraction (searching along scan lines)
 * so we must define the intrinsic parameters for the lidar used to obtain scans.
 */
struct LidarParams {
  /// @brief The number of scan lines measured by the LiDAR (e.x. Velodyne VLP-16 has 16 scan lines)
  const size_t scan_lines;
  /// @brief The number of points per scan line measured by the LiDAR
  const size_t points_per_line;
  /// @brief The min valid range of the LiDAR
  const double min_range;
  /// @brief The max valid range of the LiDAR
  const double max_range;
  /// @brief Explicit parameter constructor
  LidarParams(size_t scan_lines, size_t points_per_line, double min_range, double max_range)
      : scan_lines(scan_lines), points_per_line(points_per_line), min_range(min_range), max_range(max_range) {}
};

/**
 * ########   #######  #### ##    ## ########    ######## ##    ## ########  ########
 * ##     ## ##     ##  ##  ###   ##    ##          ##     ##  ##  ##     ## ##
 * ##     ## ##     ##  ##  ####  ##    ##          ##      ####   ##     ## ##
 * ########  ##     ##  ##  ## ## ##    ##          ##       ##    ########  ######
 * ##        ##     ##  ##  ##  ####    ##          ##       ##    ##        ##
 * ##        ##     ##  ##  ##   ###    ##          ##       ##    ##        ##
 * ##         #######  #### ##    ##    ##          ##       ##    ##        ########
 */

/// @brief Accessor struct for points that store their position as public fields
// Assumes that fields are named x, y, and z, example: PCL Point Type
template <typename PointType>
struct FieldAccessor {
  static double x(PointType pt) { return pt.x; }
  static double y(PointType pt) { return pt.y; }
  static double z(PointType pt) { return pt.z; }
};

/// @brief Accessor struct for points that store their positions in a vector/array/matrix accessed with brackets
/// Assumes that order is [x, y, z] and zero indexed, example: Eigen Matrix
template <typename PointType>
struct ParenAccessor {
  static double x(PointType pt) { return pt(0); }
  static double y(PointType pt) { return pt(1); }
  static double z(PointType pt) { return pt(2); }
};

/// @brief Accessor struct for points that store their positions in a vector/array/matrix accessed with an `at` func
/// Assumes that order is [x, y, z] and zero indexed, example: std::vector
template <typename PointType>
struct AtAccessor {
  static double x(PointType pt) { return pt.at(0); }
  static double y(PointType pt) { return pt.at(1); }
  static double z(PointType pt) { return pt.at(2); }
};

/// @brief Computes the range from the LiDAR to the point
template <typename PointType, template <typename> class Accessor = FieldAccessor>
double pointRange(const PointType& pt) {
  return std::sqrt(Accessor<PointType>::x(pt) * Accessor<PointType>::x(pt)    //
                   + Accessor<PointType>::y(pt) * Accessor<PointType>::y(pt)  //
                   + Accessor<PointType>::z(pt) * Accessor<PointType>::z(pt));
}

/// @brief Converts a point type into an Eigen 3d vector
template <typename PointType, template <typename> class Accessor = FieldAccessor>
Eigen::Vector3d pointToEigen(const PointType& pt) {
  return (Eigen::Vector3d() << Accessor<PointType>::x(pt), Accessor<PointType>::y(pt), Accessor<PointType>::z(pt))
      .finished();
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
template <typename PointType>
void validateLidarScan(const std::vector<PointType>& input_scan, const LidarParams& lidar_params) {
  if (input_scan.size() != lidar_params.scan_lines * lidar_params.points_per_line) {
    std::stringstream msg_stream;
    msg_stream << "LOAM: provided lidar scan size ( " << input_scan.size()
               << ")  does not match provided lidar parameters (" << lidar_params.scan_lines << " x "
               << lidar_params.points_per_line << ")";
    throw std::runtime_error(msg_stream.str());
  }
}

}  // namespace loam