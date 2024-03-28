#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loam/loam.h"

// Use short form as it appears a lot
namespace py = pybind11;

PYBIND11_MODULE(loam_python, m) {
  py::module gtsam = py::module::import("numpy");

  /**
   *  ######   #######  ##     ## ##     ##  #######  ##    ##
   * ##    ## ##     ## ###   ### ###   ### ##     ## ###   ##
   * ##       ##     ## #### #### #### #### ##     ## ####  ##
   * ##       ##     ## ## ### ## ## ### ## ##     ## ## ## ##
   * ##       ##     ## ##     ## ##     ## ##     ## ##  ####
   * ##    ## ##     ## ##     ## ##     ## ##     ## ##   ###
   *  ######   #######  ##     ## ##     ##  #######  ##    ##
   */

  py::class_<loam::LidarParams>(m, "LidarParams")
      .def(py::init<size_t, size_t, double, double>(),  // Constructor
           py::arg("scan_lines"), py::arg("points_per_line"), py::arg("min_range"), py::arg("max_range"))
      .def_readonly("scan_lines", &loam::LidarParams::scan_lines)
      .def_readonly("points_per_line", &loam::LidarParams::points_per_line)
      .def_readonly("min_range", &loam::LidarParams::min_range)
      .def_readonly("max_range", &loam::LidarParams::max_range);

  /**
   * ######## ########    ###    ######## ##     ## ########  ########  ######
   * ##       ##         ## ##      ##    ##     ## ##     ## ##       ##    ##
   * ##       ##        ##   ##     ##    ##     ## ##     ## ##       ##
   * ######   ######   ##     ##    ##    ##     ## ########  ######    ######
   * ##       ##       #########    ##    ##     ## ##   ##   ##             ##
   * ##       ##       ##     ##    ##    ##     ## ##    ##  ##       ##    ##
   * ##       ######## ##     ##    ##     #######  ##     ## ########  ######
   */

  py::class_<loam::FeatureExtractionParams>(m, "FeatureExtractionParams")
      .def(py::init<>())  // Constructor
      .def_readwrite("neighbor_points", &loam::FeatureExtractionParams::neighbor_points)
      .def_readwrite("number_sectors", &loam::FeatureExtractionParams::number_sectors)
      .def_readwrite("max_edge_feats_per_sector", &loam::FeatureExtractionParams::max_edge_feats_per_sector)
      .def_readwrite("max_planar_feats_per_sector", &loam::FeatureExtractionParams::max_planar_feats_per_sector)
      .def_readwrite("edge_feat_threshold", &loam::FeatureExtractionParams::edge_feat_threshold)
      .def_readwrite("planar_feat_threshold", &loam::FeatureExtractionParams::planar_feat_threshold)
      .def_readwrite("occlusion_thresh", &loam::FeatureExtractionParams::occlusion_thresh)
      .def_readwrite("parallel_thresh", &loam::FeatureExtractionParams::parallel_thresh);

  py::class_<loam::LoamFeatures<py::array_t<double>>>(m, "LoamFeatures")
      .def(py::init<>())  // Constructor
      .def_readwrite("edge_points", &loam::LoamFeatures<py::array_t<double>>::edge_points)
      .def_readwrite("planar_points", &loam::LoamFeatures<py::array_t<double>>::planar_points);

  m.def("extractFeatures", &loam::extractFeatures<py::array_t<double>, loam::AtAccessor>,  //
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams());

  m.def("computeCurvature", &loam::computeCurvature<py::array_t<double>, loam::AtAccessor>,  //
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams());

  m.def("computeValidPoints", &loam::computeValidPoints<py::array_t<double>, loam::AtAccessor>,  //
        py::arg("input_scan"), py::arg("lidar_params"), py::arg("params") = loam::FeatureExtractionParams());
}