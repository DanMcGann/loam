message(STATUS "================  TESTING ENABLED. BUILDING TESTS  ======================")
# Find Gtest which we use for unit tests
find_package(GTest REQUIRED)
enable_testing()

# Setup all tests
file(GLOB test_executables "test*.cpp")
foreach(test ${test_executables})
    get_filename_component(stem ${test} NAME_WE)
    add_executable(${stem} ${test})
    target_link_libraries(${stem} loam gtest gtest_main pthread)
    add_test(${stem} ${stem})
endforeach()