/** @brief Implementation of LOAM Geometry Module
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once

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
namespace geometry_internal {

/*********************************************************************************************************************/
template <typename T>
T pointToLineDistance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &line_a,
                      const Eigen::Matrix<T, 3, 1> &line_b) {
  T numerator = ((point - line_a).cross(point - line_b)).norm();
  T denominator = (line_a - line_b).norm();
  return numerator / denominator;
}

/*********************************************************************************************************************/
template <typename T>
T pointToPlaneDistance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &normal, const T distance) {
  return abs(normal.dot(point) - distance);
}

}  // namespace geometry_internal

}  // namespace loam