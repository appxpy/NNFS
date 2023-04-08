# C++ Skeleton Header-only library 
# Generating Doxygen Documentation

project (NNFSDoc)

cmake_minimum_required (VERSION 3.5)

find_package(Doxygen)

if (DOXYGEN_FOUND AND BUILD_DOCS)
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)

    message("Doxygen build started")

    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND AND BUILD_DOCS)