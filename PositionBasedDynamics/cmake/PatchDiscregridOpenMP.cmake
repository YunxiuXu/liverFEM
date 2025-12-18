if (NOT DEFINED DISCREGRID_SOURCE_DIR)
	message(FATAL_ERROR "DISCREGRID_SOURCE_DIR not set")
endif()

set(_root "${DISCREGRID_SOURCE_DIR}")
if (NOT EXISTS "${_root}/CMakeLists.txt")
	message(STATUS "[PatchDiscregridOpenMP] Nothing to patch (no CMakeLists.txt at ${_root})")
	return()
endif()

file(GLOB_RECURSE _cmake_lists RELATIVE "${_root}" "${_root}/*/CMakeLists.txt")
list(PREPEND _cmake_lists "CMakeLists.txt")
list(REMOVE_DUPLICATES _cmake_lists)

set(_patched 0)
foreach (_rel IN LISTS _cmake_lists)
	set(_path "${_root}/${_rel}")
	if (NOT EXISTS "${_path}")
		continue()
	endif()
	file(READ "${_path}" _txt)

	# Some Discregrid versions require OpenMP for command line tools; make it optional.
	string(REPLACE "find_package(OpenMP REQUIRED)" "find_package(OpenMP QUIET)" _new "${_txt}")
	if (NOT "${_new}" STREQUAL "${_txt}")
		file(WRITE "${_path}" "${_new}")
		math(EXPR _patched "${_patched} + 1")
	endif()
endforeach()

message(STATUS "[PatchDiscregridOpenMP] Patched ${_patched} file(s)")

