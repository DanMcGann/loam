message(STATUS "================  TESTING ENABLED. BUILDING TESTS  ======================")
# Setup gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        origin/main
)
option(INSTALL_GTEST OFF)
option(BUILD_GMOCK OFF)
FetchContent_MakeAvailable(googletest)
enable_testing() # enable gtest

# Make test target
file(GLOB_RECURSE tests_srcs *.cpp)
add_executable(loam-tests ${tests_srcs})
target_link_libraries(loam-tests PUBLIC loam Eigen3::Eigen gtest gtest_main)
add_test(NAME loam-tests COMMAND loam-tests)

# Make runnable from "make check" or "make test"
add_custom_target(loam-check COMMAND loam-tests)
add_custom_target(loam-test COMMAND loam-tests)