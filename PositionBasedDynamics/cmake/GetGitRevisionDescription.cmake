include_guard(GLOBAL)

find_package(Git QUIET)

function(get_git_head_revision _refspec_var _sha1_var)
	if (GIT_FOUND)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" rev-parse --verify HEAD
			WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
			OUTPUT_VARIABLE _sha
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET
			RESULT_VARIABLE _res
		)
		if (_res EQUAL 0 AND NOT "${_sha}" STREQUAL "")
			set(${_sha1_var} "${_sha}" PARENT_SCOPE)
		else()
			set(${_sha1_var} "unknown" PARENT_SCOPE)
		endif()

		execute_process(
			COMMAND "${GIT_EXECUTABLE}" rev-parse --symbolic-full-name HEAD
			WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
			OUTPUT_VARIABLE _ref
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET
		)
		if ("${_ref}" STREQUAL "")
			set(_ref "HEAD")
		endif()
		set(${_refspec_var} "${_ref}" PARENT_SCOPE)
	else()
		set(${_refspec_var} "HEAD" PARENT_SCOPE)
		set(${_sha1_var} "unknown" PARENT_SCOPE)
	endif()
endfunction()

function(git_local_changes _changes_var)
	if (GIT_FOUND)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" status --porcelain
			WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
			OUTPUT_VARIABLE _out
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET
		)
		if ("${_out}" STREQUAL "")
			set(${_changes_var} "CLEAN" PARENT_SCOPE)
		else()
			set(${_changes_var} "DIRTY" PARENT_SCOPE)
		endif()
	else()
		set(${_changes_var} "UNKNOWN" PARENT_SCOPE)
	endif()
endfunction()

