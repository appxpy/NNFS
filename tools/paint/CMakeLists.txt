cmake_minimum_required(VERSION 3.5)

project(paint VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Test)

set(PROJECT_SOURCES
        main.cpp
        canvas.cpp
        canvas.h
        paint.cpp
        paint.h
        paint.ui
    )

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(paint
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(paint SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(paint
            ${PROJECT_SOURCES}
        )
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(paint PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Eigen3::Eigen NNFSProject::NNFS)

set_target_properties(paint PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS paint
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(paint)
endif()

enable_testing(true)

add_executable(tests test_canvas.h test_canvas.cpp canvas.h canvas.cpp)
add_test(NAME CanvasTest COMMAND tests)

target_link_libraries(tests PRIVATE Qt${QT_VERSION_MAJOR}::Test Eigen3::Eigen NNFSProject::NNFS Qt${QT_VERSION_MAJOR}::Widgets)