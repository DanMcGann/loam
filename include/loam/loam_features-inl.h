/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once

#include "loam/loam_features.h"

namespace loam {

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Accessor = FieldAccessor>
LoamFeatures<PointType> extractFeatures(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                        const FeatureExtractionParams& params) {
  validateLidarScan(input_scan, lidar_params);
  // Initialize Results
  LoamFeatures<PointType> result;
  // Compute the number of points in each sector given the parameters
  const size_t points_per_sector = lidar_params.points_per_line / params.number_sectors;

  // Step 1: Compute Curvature for all points this is used in the extraction of both planar and edge feature points
  std::vector<PointCurvature> curvature = computeCurvature<PointType, Accessor>(input_scan, lidar_params, params);

  // Step 2: Compute mask of valid points
  std::vector<bool> valid_mask = computeValidPoints<PointType, Accessor>(input_scan, lidar_params, params);

  /// Step 3: Detect features in each sector of each scan line
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    // Independently detect features in each sector of this scan_line
    for (size_t sector_idx = 0; sector_idx < params.number_sectors; sector_idx++) {
      // Get the point index of the sector start and sector end
      const size_t sector_start_point =
          (scan_line_idx * lidar_params.points_per_line) + (sector_idx * points_per_sector);
      // Special case for end point as we add any reminder points to the last sector
      const size_t sector_end_point = (sector_idx == params.number_sectors - 1)
                                          ? ((scan_line_idx + 1) * lidar_params.points_per_line)
                                          : sector_start_point + points_per_sector;

      // Sort the points based on curvature
      std::sort(curvature.begin() + sector_start_point, curvature.begin() + sector_end_point, curvatureComparator);

      size_t num_sector_edge_features = 0;
      size_t num_sector_planar_features = 0;
      // Search largest to smallest curvature [i.e. edge features]
      for (size_t sorted_curv_idx_p1 = sector_end_point; sorted_curv_idx_p1 > sector_start_point;
           sorted_curv_idx_p1--) {
        const PointCurvature curv = curvature[sorted_curv_idx_p1 - 1];  // subtraction as loop cannot go negative
        if (valid_mask[curv.index] && curv.curvature > params.edge_feat_threshold) {
          result.edge_points.push_back(input_scan.at(curv.index));  // Add to edge points
          for (size_t n = 0; n < params.neighbor_points; n++) {     // update mask
            valid_mask[curv.index + n] = false;
            valid_mask[curv.index - n] = false;
          }
          num_sector_edge_features++;
        }
        // Early exit if we have found enough features
        if (num_sector_edge_features > params.max_edge_feats_per_sector) break;
      }

      // Search smallest to largest [i.e. planar features]
      for (size_t sorted_curv_idx = sector_start_point; sorted_curv_idx < sector_end_point; sorted_curv_idx++) {
        const PointCurvature curv = curvature[sorted_curv_idx];
        if (valid_mask[curv.index] && curv.curvature < params.planar_feat_threshold) {
          result.planar_points.push_back(input_scan.at(curv.index));  // Add to edge points
          for (size_t n = 0; n < params.neighbor_points; n++) {       // update mask
            valid_mask[curv.index + n] = false;
            valid_mask[curv.index - n] = false;
          }
          num_sector_planar_features++;
        }
        // Early exit if we have found enough features
        if (num_sector_planar_features > params.max_planar_feats_per_sector) break;

      }  // end feature search in sector
    }    // end sector search
  }      // end scan line search
  return result;
}

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Accessor = FieldAccessor>
std::vector<PointCurvature> computeCurvature(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                             const FeatureExtractionParams& params) {
  validateLidarScan(input_scan, lidar_params);
  // Allocate vector (with zeros) to store curvature
  std::vector<PointCurvature> curvature;

  // Structured search (search over each scan line individually over all points [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < lidar_params.points_per_line; line_pt_idx++) {
      const size_t idx = (scan_line_idx * lidar_params.points_per_line) + line_pt_idx;
      // If point is on the edge of the scan line record invalid curvature [-1]
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= lidar_params.points_per_line - params.neighbor_points) {
        curvature.push_back(PointCurvature(idx, -1));
      }
      // If not an edge point compute the curvature
      else {
        // Initialize with the difference term
        double dx = -(2.0 * params.neighbor_points) * Accessor<PointType>::x(input_scan[idx]);
        double dy = -(2.0 * params.neighbor_points) * Accessor<PointType>::y(input_scan[idx]);
        double dz = -(2.0 * params.neighbor_points) * Accessor<PointType>::z(input_scan[idx]);
        // Iterate over neighbors and accumulate
        for (size_t n = 1; n <= params.neighbor_points; n++) {
          dx = dx + Accessor<PointType>::x(input_scan[idx - n]) + Accessor<PointType>::x(input_scan[idx + n]);
          dy = dy + Accessor<PointType>::y(input_scan[idx - n]) + Accessor<PointType>::y(input_scan[idx + n]);
          dz = dz + Accessor<PointType>::z(input_scan[idx - n]) + Accessor<PointType>::z(input_scan[idx + n]);
        }
        curvature.push_back(PointCurvature(idx, dx * dx + dy * dy + dz * dz));
      }
    }
  }
  return curvature;
}

/*********************************************************************************************************************/
template <typename PointType, template <typename> class Accessor = FieldAccessor>
std::vector<bool> computeValidPoints(const std::vector<PointType>& input_scan, const LidarParams& lidar_params,
                                     const FeatureExtractionParams& params) {
  validateLidarScan(input_scan, lidar_params);
  // Allocate mask (with valid flags)
  std::vector<bool> mask(input_scan.size(), true);

  // Structured search (search over each scan line individually over all points [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.scan_lines; scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < lidar_params.points_per_line; line_pt_idx++) {
      const size_t idx = (scan_line_idx * lidar_params.points_per_line) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points of each scan line are invalid
      if (line_pt_idx < params.neighbor_points ||
          line_pt_idx >= lidar_params.points_per_line - params.neighbor_points) {
        mask[idx] = false;
        continue;
      }

      // Get the current point and its two neighbors
      const PointType prev_point = input_scan.at(idx - 1);
      const PointType point = input_scan.at(idx);
      const PointType next_point = input_scan[idx + 1];

      // Compute the range of each point
      const double point_range = pointRange<PointType, Accessor>(point);
      const double next_point_range = pointRange<PointType, Accessor>(next_point);
      const double prev_point_range = pointRange<PointType, Accessor>(prev_point);

      // CHECK 2: Is the point in the valid range of the LiDAR
      if (point_range < lidar_params.min_range || point_range > lidar_params.max_range) {
        mask[idx] = false;
        for (size_t n = 1; n <= params.neighbor_points; n++) {
          mask[idx + n] = false;
          mask[idx - n] = false;
        }
        continue;  // continue with loop
      }

      // CHECK 3: Occlusions
      if (next_point_range - point_range > params.occlusion_thresh) {  // Case 1
        for (size_t n = 1; n <= params.neighbor_points; n++) mask[idx + n] = false;
        continue;
      } else if (point_range - next_point_range > params.occlusion_thresh) {  // Case 2
        for (size_t n = 0; n < params.neighbor_points; n++) mask[idx - n] = false;
        continue;
      }

      // Check 4: Check if the point is on a plane nearly parallel to the LiDAR Beam
      double diff_next = std::abs(prev_point_range - point_range);
      double diff_prev = std::abs(next_point_range - point_range);
      if (diff_next > params.parallel_thresh * point_range && diff_prev > params.parallel_thresh * point_range) {
        mask[idx] = false;
      }
    }
  }
  return mask;
}

}  // namespace loam
