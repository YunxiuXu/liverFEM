include(FindPackageHandleStandardArgs)

set(_eigen_hints "")
if (DEFINED EIGEN3_INCLUDE_DIR)
	list(APPEND _eigen_hints "${EIGEN3_INCLUDE_DIR}")
endif()
if (DEFINED PROJECT_PATH)
	list(APPEND _eigen_hints "${PROJECT_PATH}/extern/eigen")
endif()
list(APPEND _eigen_hints "${CMAKE_CURRENT_LIST_DIR}/../extern/eigen")

find_path(EIGEN3_INCLUDE_DIR
	NAMES Eigen/Core
	HINTS ${_eigen_hints}
	PATH_SUFFIXES eigen3
)

find_package_handle_standard_args(Eigen3 DEFAULT_MSG EIGEN3_INCLUDE_DIR)

if (Eigen3_FOUND)
	set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
endif()

