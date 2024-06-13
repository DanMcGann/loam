/** @brief Implementation of LOAM Features Module
 * @author Dan McGann
 * @date Mar 2024
 */
#include "loam/features.h"

namespace loam {
/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */
/*********************************************************************************************************************/
namespace features_internal {
/*********************************************************************************************************************/
bool markEdgesInvalid(const size_t& idx, const size_t& line_pt_idx, const LidarParams& lidar_params,
                      const FeatureExtractionParams& params, std::vector<bool>& mask) {
  if (line_pt_idx < params.neighbor_points || line_pt_idx >= lidar_params.points_per_line - params.neighbor_points) {
    mask[idx] = false;
    return true;
  }
  return false;
}

/*********************************************************************************************************************/
bool markOutOfRangeInvalid(const size_t& idx, const double& point_range, const LidarParams& lidar_params,
                           const FeatureExtractionParams& params, std::vector<bool>& mask) {
  if (point_range < lidar_params.min_range || point_range > lidar_params.max_range) {
    mask[idx] = false;
    for (size_t n = 1; n <= params.neighbor_points; n++) {
      mask[idx + n] = false;
      mask[idx - n] = false;
    }
    return true;
  }
  return false;
}

/*********************************************************************************************************************/
bool markOccludedInvalid(const size_t& idx, const double& point_range, const double& next_point_range,
                         const FeatureExtractionParams& params, std::vector<bool>& mask) {
  if (next_point_range - point_range > params.occlusion_thresh) {  // Case 1
    for (size_t n = 1; n <= params.neighbor_points; n++) mask[idx + n] = false;
    return true;
  } else if (point_range - next_point_range > params.occlusion_thresh) {  // Case 2
    for (size_t n = 0; n < params.neighbor_points; n++) mask[idx - n] = false;
    return true;
  }
  return false;
}

/*********************************************************************************************************************/
bool markParallelInvalid(const size_t& idx, const double& prev_point_range, const double& point_range,
                         const double& next_point_range, const FeatureExtractionParams& params,
                         std::vector<bool>& mask) {
  double diff_next = std::abs(prev_point_range - point_range);
  double diff_prev = std::abs(next_point_range - point_range);
  if (diff_next > params.parallel_thresh * point_range && diff_prev > params.parallel_thresh * point_range) {
    mask[idx] = false;
    return true;
  }

  return false;
}

}  // namespace features_internal

}  // namespace loam
