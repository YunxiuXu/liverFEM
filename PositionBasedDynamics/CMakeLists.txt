cmake_minimum_required(VERSION 3.10)

project(PositionBasedDynamics LANGUAGES C CXX)

# Some sub-CMakeLists use PROJECT_PATH to build absolute paths.
set(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_DOUBLE_PRECISION "Use double precision" ON)
option(USE_OpenMP "Use OpenMP (optional)" ON)
option(PBD_LIBS_ONLY "Build only libraries (no demos)" OFF)
option(USE_PYTHON_BINDINGS "Build python bindings" OFF)

set(PBD_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Where to place executables and resources")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PBD_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Some bundled deps assume non-empty postfix variables (CMake 4 enforces arg pairs).
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")
set(CMAKE_MINSIZEREL_POSTFIX "_s")

# Keep the original build behavior seen in this repo's scripts/build cache.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # gnu++11

if(APPLE)
	add_compile_definitions(GL_SILENCE_DEPRECATION)
endif()

if (USE_DOUBLE_PRECISION)
	add_compile_definitions(USE_DOUBLE)
endif()

# Prefer our bundled Eigen headers when the system doesn't provide Eigen3.
list(PREPEND CMAKE_INCLUDE_PATH "${PROJECT_PATH}/extern/eigen")

# CMake helper modules (missing in the upstream snapshot).
list(APPEND CMAKE_MODULE_PATH "${PROJECT_PATH}/cmake")

# Small include-order tweak that the previous build cache used.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -idirafter${PROJECT_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -idirafter${PROJECT_PATH}/extern/eigen -mcpu=native -idirafter${PROJECT_PATH}")

############################################################
# External Dependencies (vendored)
############################################################
add_subdirectory(extern/glfw)
add_subdirectory(extern/imgui)
add_subdirectory(extern/md5)

############################################################
# External Dependencies (git via ExternalProject)
############################################################
include(ExternalProject)

set(PBD_DISCREGRID_REPO "https://github.com/InteractiveComputerGraphics/Discregrid.git"
	CACHE STRING "Discregrid git repository URL/path")
set(PBD_GENERICPARAMETERS_REPO "https://github.com/InteractiveComputerGraphics/GenericParameters.git"
	CACHE STRING "GenericParameters git repository URL/path")

set(Discregrid_INCLUDE_DIR "${CMAKE_BINARY_DIR}/Discregrid/include")
set(Discregrid_LIBRARIES "${CMAKE_BINARY_DIR}/Discregrid/lib/libDiscregrid.a")
set(GenericParameters_INCLUDE_DIR "${CMAKE_BINARY_DIR}/GenericParameters/include")

ExternalProject_Add(Ext_GenericParameters
	PREFIX "${CMAKE_BINARY_DIR}/GenericParameters"
	GIT_REPOSITORY "${PBD_GENERICPARAMETERS_REPO}"
	UPDATE_DISCONNECTED 1
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
		-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/GenericParameters
		-DGENERICPARAMETERS_NO_TESTS:BOOL=1
		-DCMAKE_POLICY_VERSION_MINIMUM=3.10
)

ExternalProject_Add(Ext_Discregrid
	PREFIX "${CMAKE_BINARY_DIR}/extern/Discregrid"
	GIT_REPOSITORY "${PBD_DISCREGRID_REPO}"
	UPDATE_DISCONNECTED 1
	PATCH_COMMAND
		${CMAKE_COMMAND} -DDISCREGRID_SOURCE_DIR=<SOURCE_DIR> -P "${PROJECT_PATH}/CMake/PatchDiscregridOpenMP.cmake"
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
		-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/Discregrid
		-DBUILD_CMD_EXECUTABLE:BOOL=0
		-DEIGEN3_INCLUDE_DIR:PATH=${PROJECT_PATH}/extern/eigen
		"-DCMAKE_CXX_FLAGS= -idirafter${PROJECT_PATH}/extern/eigen"
		-DCMAKE_POLICY_VERSION_MINIMUM=3.10
)

############################################################
# Resource copy targets (used by demo CMakeLists)
############################################################
add_custom_target(CopyPBDShaders
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold "Copying PBD shaders"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${PROJECT_PATH}/data/shaders"
		"${PBD_RUNTIME_OUTPUT_DIRECTORY}/resources/shaders"
)

add_custom_target(CopyPBDModels
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold "Copying PBD models"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${PROJECT_PATH}/data/models"
		"${PBD_RUNTIME_OUTPUT_DIRECTORY}/resources/models"
)

add_custom_target(CopyPBDScenes
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold "Copying PBD scenes"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${PROJECT_PATH}/data/scenes"
		"${PBD_RUNTIME_OUTPUT_DIRECTORY}/resources/scenes"
)

add_custom_target(CopyImguiFonts
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue --bold "Copying fonts"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${PROJECT_PATH}/extern/imgui/misc/fonts"
		"${PBD_RUNTIME_OUTPUT_DIRECTORY}/resources/fonts"
)

############################################################
# Core libraries + demos
############################################################
add_subdirectory(PositionBasedDynamics)
add_subdirectory(Utils)
add_subdirectory(Simulation)
add_subdirectory(Demos)

if (USE_PYTHON_BINDINGS)
	add_subdirectory(pyPBD)
endif()
