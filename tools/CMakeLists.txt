find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")

add_executable(train train.cpp)
target_link_libraries(train PRIVATE NNFS::NNFSCore GTest::gtest_main CURL::libcurl ZLIB::ZLIB)
target_compile_options(train PRIVATE)