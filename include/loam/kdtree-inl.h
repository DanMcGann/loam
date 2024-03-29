/** @brief Implementation of LOAM KDTree Module
 * @author Dan McGann
 * @date Mar 2024
 */
#include "loam/kdtree.h"

namespace loam {
namespace kdtree_internal {
/*********************************************************************************************************************/
std::vector<size_t> knnSearch(const KDTree& tree, const Eigen::Vector3d& query, const size_t k, const double max_dist) {
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

}  // namespace kdtree_internal
}  // namespace loam