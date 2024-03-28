message(STATUS "================  PYTHON BINDINGS ENABLED. BUILDING BINDINGS  ======================")

# Fetch pybind to build the python bindings
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.0
)
FetchContent_MakeAvailable(pybind11)

# Declare the bybind target
file(GLOB_RECURSE pybind_srcs "*.cpp")
pybind11_add_module(loam_python ${pybind_srcs})
target_link_libraries(loam_python PRIVATE loam)

set(LOAM_PYTHON_BUILD_DIRECTORY ${PROJECT_BINARY_DIR}/python)
set_target_properties(loam_python PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    OUTPUT_NAME "loam_python"
    LIBRARY_OUTPUT_DIRECTORY "${LOAM_PYTHON_BUILD_DIRECTORY}/loam"
    DEBUG_POSTFIX "" # Otherwise you will have a wrong name
    RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
    )

# Install Python
# Generate the setup.py file in the correct directory. This will be used to install the python module.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup.py.in
                ${LOAM_PYTHON_BUILD_DIRECTORY}/setup.py)

# Create an actual python module in the build structure. Technically this is the module that gets installed
# It simply imports everything from the .so module built by pybind11
file(GENERATE OUTPUT ${LOAM_PYTHON_BUILD_DIRECTORY}/loam/__init__.py CONTENT "from loam.loam_python import *")

# Setup the install command. Simply delegates to pip (should work with environmets)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_custom_target(python-install
        COMMAND ${Python_EXECUTABLE} -m pip install .
        DEPENDS loam_python
        WORKING_DIRECTORY ${LOAM_PYTHON_BUILD_DIRECTORY})

# Setup uninstall command. This can also be done with pip
add_custom_target(python-uninstall
        COMMAND pip uninstall loam
        WORKING_DIRECTORY ${LOAM_PYTHON_BUILD_DIRECTORY})