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

#include "loam/common.h"

namespace loam {
namespace kdtree_internal {

/** @brief Adaptor class required by nanoflann
 * This class unfortunately needs to use our own adaptor (Accessor) to extract info from the points.
 * Thankfully it is not too complected to do so
 */
struct KDTreeDataAdaptor {
  const std::vector<Eigen::Vector3d>& data;
  // Interface required by nanoflann
  size_t kdtree_get_point_count() const { return data.size(); }
  double kdtree_get_pt(const size_t idx, const size_t dim) const { return data.at(idx)(dim); }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
  KDTreeDataAdaptor(const std::vector<Eigen::Vector3d>& data) : data(data) {}
};

/// @brief Type for the KDTree distance metric [required by nanoflann]
using KDTreeDistance = nanoflann::L2_Simple_Adaptor<double, KDTreeDataAdaptor>;
/// @brief Type for the KDTree using a compile time dimension of 3 for 3d pointclouds
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<KDTreeDistance, KDTreeDataAdaptor, 3>;
/// @brief Type for the KDTree parameters for convience
using KDTreeParams = nanoflann::KDTreeSingleIndexAdaptorParams;

/** @brief Runs a radius limited K Nearest Neighbor (knn) search on the tree.
 * @param tree: The KDTree over which to search
 * @param query: The query point for which neighbors are found
 * @param k: The number of points to search for
 * @param max_dist: The radius limit for the search. If max_dist <= 0 no radius limit is used
 */
std::vector<size_t> knnSearch(const KDTree& tree, const Eigen::Vector3d& query, const size_t k,
                              const double max_dist = -1);

}  // namespace kdtree_internal

}  // namespace loam

// Include the actual implementation
#include "loam/kdtree-inl.h"