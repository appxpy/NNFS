cmake_minimum_required(VERSION 3.15)
project(NNFS)

enable_testing()


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)
SET(CMAKE_CXX_FLAGS "-g -O3 ${CMAKE_CXX_FLAGS}") # -fprofile-arcs -ftest-coverage -fno-elide-constructors -fno-default-inline -fopenmp
SET(CMAKE_C_FLAGS "-g -O3 ${CMAKE_C_FLAGS}") # -ftest-coverage -fopenmp 


option(GENERATE_DOCS "Create target for doxygen auto-docs report" OFF)
option(CODE_COVERAGE "Create targets for test coverage report" OFF)


if (GENERATE_DOCS)
    set(GENERATE_DOCS ON)
endif (GENERATE_DOCS)

if (CODE_COVERAGE)
    set(CODE_COVERAGE ON)
endif (CODE_COVERAGE)

add_subdirectory(external)
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(GTest REQUIRED)
add_subdirectory(include)
add_subdirectory(tools)
add_subdirectory(tests)
# add_subdirectory(docs)

