add_executable(nnfs_tests test_activation.cpp test_callback.cpp test_loss.cpp)
target_link_libraries(nnfs_tests PRIVATE NNFS::NNFSCore GTest::gtest_main)
target_compile_options(nnfs_tests PRIVATE)

include(GoogleTest)
gtest_discover_tests(nnfs_tests)


if (CODE_COVERAGE)
	INCLUDE(CodeCoverage)

	target_code_coverage(
		nnfs_tests  # Name for custom target.
		EXCLUDE ${CMAKE_SOURCE_DIR}/external/* ${CMAKE_SOURCE_DIR}/tests/*
		AUTO
	)
endif (CODE_COVERAGE)