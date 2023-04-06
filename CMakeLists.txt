cmake_minimum_required(VERSION 3.15)
project(NNFS)

enable_testing()


# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS " --coverage")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)
SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage ${CMAKE_C_FLAGS}")

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(tools)
add_subdirectory(tests)
add_subdirectory(docs)

