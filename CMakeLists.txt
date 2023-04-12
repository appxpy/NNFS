cmake_minimum_required(VERSION 3.15)
project(NNFS)

enable_testing()


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)
SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage -fno-elide-constructors -fno-default-inline -fopenmp ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage -fopenmp ${CMAKE_C_FLAGS}")


option(GENERATE_DOCS "Create target for doxygen auto-docs report" ON)
option(CODE_COVERAGE "Create targets for test coverage report" ON)


if (GENERATE_DOCS)
    set(GENERATE_DOCS ON)
endif (GENERATE_DOCS)

if (CODE_COVERAGE)
    set(CODE_COVERAGE ON)
endif (CODE_COVERAGE)

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(tools)
# add_subdirectory(tests)
# add_subdirectory(docs)

