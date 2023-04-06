# This project requires at least CMake 3.5
cmake_minimum_required (VERSION 3.5 FATAL_ERROR)

# Set project name
project(NNFSCore)

# Create a header-only library (INTERFACE)
# Create alias lib::myLibrary so other projects can link to it 
add_library(${PROJECT_NAME} INTERFACE)

add_library(NNFS::NNFSCore ALIAS ${PROJECT_NAME})


# Specify where the header-only library is
target_include_directories( ${PROJECT_NAME}
    INTERFACE ${PROJECT_SOURCE_DIR}
)

target_link_libraries(NNFSCore INTERFACE Eigen3::Eigen)