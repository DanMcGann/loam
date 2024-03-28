# Dependencies

Eigen

Ceres [Eigen, glog, libsuitesparse-dev, gflags]

Nanoflann




# Quirks
* We need ceres 2.2.0 to make use of manifolds so we access it via fetch content. This causes a cmake name collision of `uninstall` with nanoflann. Since neigher prefix their target names. Thankfully ceres provides an option `PROVIDE_UNINSTALL_TARGET` (also not prefixed :/) to resolve this collision. Additionally, since ceres is built locally, we will never need to install OR uninstall it, so loosing this target is a-okay.