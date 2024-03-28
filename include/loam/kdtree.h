/** @brief This module provides helpers for working with KDTrees
 * We use the nanoflann library [2] to provide most KDTree functionality. However, it is a very generic library and some
 * of its functionality and interface are unused. This module primarily wraps nanoflann to make using it in the context
 * of loam easier.
 *
 * [2] Blanco, J., & Rai, P. (2014). nanoflann: a C++ header-only fork of FLANN, a library for Nearest Neighbor (NN)
 *     with KD-trees. https://github.com/jlblancoc/nanoflann
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <nanoflann.hpp>

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

/** @brief Adaptor class required by nanoflann
 * This class unfortunately needs to use our own adaptor (Accessor) to extract info from the points.
 * Thankfully it is not too complected to do so
 */
template <typename PointType, template <typename> class Accessor = FieldAccessor>
struct KDTreeDataAdaptor {
  const std::vector<PointType>& data;
  // Interface required by nanoflann
  size_t kdtree_get_point_count() const { return data.size(); }
  double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0) return Accessor<PointType>::x(data.at(idx));
    if (dim == 1) return Accessor<PointType>::y(data.at(idx));
    if (dim == 2) return Accessor<PointType>::z(data.at(idx));
    throw std::runtime_error("Dev error kdtree set to > 3 dim");
  }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
  KDTreeDataAdaptor(const std::vector<PointType>& data) : data(data) {}
};

/// @brief Type for the KDTree distance metric [required by nanoflann]
template <typename PointType, template <typename> class Accessor = FieldAccessor>
using KDTreeDistance = nanoflann::L2_Simple_Adaptor<double, KDTreeDataAdaptor<PointType, Accessor>>;

/// @brief Type for the KDTree using a compile time dimension of 3 for 3d pointclouds
template <typename PointType, template <typename> class Accessor = FieldAccessor>
using KDTree =
    nanoflann::KDTreeSingleIndexAdaptor<KDTreeDistance<PointType, Accessor>, KDTreeDataAdaptor<PointType, Accessor>, 3>;

/// @brief Type for the KDTree parameters for convience
using KDTreeParams = nanoflann::KDTreeSingleIndexAdaptorParams;

/**
 * #### ##    ## ######## ######## ########  ########    ###     ######  ########
 *  ##  ###   ##    ##    ##       ##     ## ##         ## ##   ##    ## ##
 *  ##  ####  ##    ##    ##       ##     ## ##        ##   ##  ##       ##
 *  ##  ## ## ##    ##    ######   ########  ######   ##     ## ##       ######
 *  ##  ##  ####    ##    ##       ##   ##   ##       ######### ##       ##
 *  ##  ##   ###    ##    ##       ##    ##  ##       ##     ## ##    ## ##
 * #### ##    ##    ##    ######## ##     ## ##       ##     ##  ######  ########
 */

/** @brief Runs a radius limited K Nearest Neighbor (knn) search on the tree.
 * @param tree: The KDTree overwhich to search
 * @param query: The query point for which neighbors are found
 * @param k: The number of points to search for
 * @param max_dist: The radius limit for the search. If max_dist <= 0 no radius limit is used
 */
// TODO (dan) remove templatization from this wrapper since we will only use with eigen now
template <typename PointType, template <typename> class Accessor = FieldAccessor>
std::vector<size_t> knnSearch(const KDTree<PointType, Accessor>& tree, const Eigen::Vector3d& query, const size_t k,
                              const double max_dist = -1) {
  // Setup the search
  std::vector<size_t> knn_indicies(k);
  std::vector<double> knn_distances_sq(k);
  nanoflann::KNNResultSet<double> result_set(k);
  result_set.init(&knn_indicies[0], &knn_distances_sq[0]);

  // Run the search on the kdtree
  tree.findNeighbors(result_set, &query[0]);

  // Setup the return structure
  std::vector<size_t> result;

  // Add any valid neighbors to the result
  for (size_t i = 0; i < result_set.size(); i++) {
    if (max_dist <= 0 || std::sqrt(knn_distances_sq[i]) < max_dist) result.push_back(knn_indicies[i]);
  }
  return result;
}

}  // namespace loam