find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE NNFS::NNFSCore GTest::gtest_main CURL::libcurl ZLIB::ZLIB)
target_compile_options(train PRIVATE)