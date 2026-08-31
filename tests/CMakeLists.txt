include("${PROJECT_SOURCE_DIR}/cmake/GENDILTestHelpers.cmake")

add_subdirectory(cmake)
add_subdirectory(unit)
add_subdirectory(integration)

get_property(gendil_test_targets GLOBAL PROPERTY GENDIL_TEST_TARGETS)
add_custom_target(
  gendil-tests
  COMMENT "Building all GenDiL test executables"
)
if(gendil_test_targets)
  add_dependencies(gendil-tests ${gendil_test_targets})
endif()
