add_executable(nnfs_tests test_activation.cpp test_callback.cpp)
target_link_libraries(nnfs_tests PRIVATE NNFS::NNFSCore GTest::gtest_main --coverage)
target_compile_options(nnfs_tests PRIVATE --coverage)

include(GoogleTest)
gtest_discover_tests(nnfs_tests)

if (GENERATE_COV)
	INCLUDE(CodeCoverage)
	SETUP_TARGET_FOR_COVERAGE(
					coverage  # Name for custom target.
					ctest     # Name of the test driver executable that runs the tests.
										# NOTE! This should always have a ZERO as exit code
										# otherwise the coverage generation will not complete.
					coverage            # Name of output directory.
					)
endif (GENERATE_COV)