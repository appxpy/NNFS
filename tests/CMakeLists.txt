add_executable(nnfs_tests test_loss.cpp test_dense.cpp test_activation.cpp test_metrics.cpp) # test_callback.cpp  test_layer.cpp test_neural_network.cpp
target_link_libraries(nnfs_tests PRIVATE NNFS::NNFSCore GTest::gtest_main)
target_compile_options(nnfs_tests PRIVATE)

include(GoogleTest)
gtest_discover_tests(nnfs_tests)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
set(COVERAGE_EXCLUDES ".*external.*" ".*tests.*")
else()
set(COVERAGE_EXCLUDES "${CMAKE_SOURCE_DIR}/external/*" "${CMAKE_SOURCE_DIR}/tests/*")
endif()

if (CODE_COVERAGE)
	INCLUDE(CodeCoverage)
    # Check if llvm, use regex EXCLUDE, if gcc use glob EXCLUDE

	target_code_coverage(
		nnfs_tests  # Name for custom target.
		EXCLUDE ${COVERAGE_EXCLUDES} # Files to EXCLUDE.
		AUTO
	)
endif (CODE_COVERAGE)