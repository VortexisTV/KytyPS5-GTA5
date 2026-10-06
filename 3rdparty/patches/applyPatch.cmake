# FetchContent can rerun the patch step when the project is reconfigured. Recognize an
# already-applied patch, but fail if the source matches neither side of it.
execute_process(
	COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace --reverse --check "${PATCH_FILE}"
	RESULT_VARIABLE reverse_result
	OUTPUT_QUIET ERROR_QUIET
)
if(reverse_result EQUAL 0)
	return()
endif()

execute_process(
	COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace --check "${PATCH_FILE}"
	RESULT_VARIABLE check_result
	ERROR_VARIABLE check_error
)
if(NOT check_result EQUAL 0)
	message(FATAL_ERROR "Cannot apply ${PATCH_FILE}: ${check_error}")
endif()
execute_process(
	COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${PATCH_FILE}"
	COMMAND_ERROR_IS_FATAL ANY
)
