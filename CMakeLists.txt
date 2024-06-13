project(Loam CXX C)
cmake_minimum_required(VERSION 2.8.3)
set (CMAKE_CXX_STANDARD 17)
include(FetchContent)

message(STATUS "================  BUILDING LOAM  ======================")

# Dependencies
# Eigen - used by ceres and during registration
find_package(Eigen3 REQUIRED)
# Ceres - used to solve ICF in registration 
set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build ceres examples [slow]" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Do not build ceres tests [slow]" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Do not build ceres uninstall target")
FetchContent_Declare(
  ceres
  GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
  GIT_TAG        2.2.0
)
FetchContent_MakeAvailable(ceres)

# nanoflann is a base dependency for KNN search in the database
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "Do not build nanoflann examples")
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "Do not build nanoflann tests")
FetchContent_Declare(
  nanoflann
  GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
  GIT_TAG        v1.5.5
)
FetchContent_MakeAvailable(nanoflann)

# Define the library
file(GLOB loam_srcs "${CMAKE_CURRENT_SOURCE_DIR}/loam/src/*.cpp")
file(GLOB loam_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/loam/include/loam/*.h")
add_library(loam SHARED ${loam_srcs} ${loam_hdrs})
target_link_libraries(loam PUBLIC Eigen3::Eigen Ceres::ceres nanoflann)
target_include_directories(loam PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/loam/include")

# If testing is configured, setup the tests
option(LOAM_BUILD_TESTS "Build unit tests" OFF)
if (${LOAM_BUILD_TESTS})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()


# If python bindings are configured, build the python bindings
option(LOAM_BUILD_PYTHON "Build python bindings" OFF)
if (${LOAM_BUILD_PYTHON})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/python")
endif()
