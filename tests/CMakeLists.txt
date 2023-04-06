INCLUDE(CodeCoverage)

add_executable(test_neural_network test_neural_network.cpp)
target_link_libraries(test_neural_network PRIVATE NNFS::NNFSCore GTest::gtest_main --coverage)
target_compile_options(test_neural_network PRIVATE --coverage)

add_test(test_all test_neural_network)

SETUP_TARGET_FOR_COVERAGE(
				coverage  # Name for custom target.
				ctest     # Name of the test driver executable that runs the tests.
									# NOTE! This should always have a ZERO as exit code
									# otherwise the coverage generation will not complete.
				coverage            # Name of output directory.
				)