cmake_minimum_required(VERSION 3.15)
project(NNFSProject)

enable_testing()


SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)
SET(CMAKE_CXX_FLAGS "-g -O3 ${CMAKE_CXX_FLAGS}") # -fprofile-arcs -ftest-coverage -fno-elide-constructors -fno-default-inline -fopenmp
SET(CMAKE_C_FLAGS "-g -O3 ${CMAKE_C_FLAGS}") # -ftest-coverage -fopenmp 


SET(GENERATE_DOCS ON) # "Create target for doxygen auto-docs report"
SET(CODE_COVERAGE ON) # "Create targets for test coverage report"

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(tools)
add_subdirectory(tests)
add_subdirectory(docs)