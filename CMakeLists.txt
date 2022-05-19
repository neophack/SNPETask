# create by Ricardo Lu in 05-17-2022

cmake_minimum_required(VERSION 3.10)

project(yolov5s)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR "/opt/thundersoft/opencv-4.2.0/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

include(FindPkgConfig)
pkg_check_modules(GFLAGS REQUIRED gflags)

include_directories(
    /usr/include/zdl
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/snpetask
    ${PROJECT_SOURCE_DIR}/utility
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${GFLAGS_LIBRARY_DIRS}
)

add_library(TSYolov5s
    SHARED
    ${PROJECT_SOURCE_DIR}/src/TSYolov5s.cpp
    ${PROJECT_SOURCE_DIR}/src/TSYolov5sImpl.cpp
    ${PROJECT_SOURCE_DIR}/snpetask/SNPETask.cpp
    ${PROJECT_SOURCE_DIR}/utility/TSImgData.cpp
    ${PROJECT_SOURCE_DIR}/utility/imgbuf.cpp
)

target_link_libraries(TSYolov5s
    SNPE
    pthread
    dl
    ${OpenCV_LIBS}
)

add_executable(test-yolov5s
    ${PROJECT_SOURCE_DIR}/test/main.cpp
)

target_link_libraries(test-yolov5s
    PUBLIC
    pthread
    dl
    ${OpenCV_LIBS}
    ${GFLAGS_LIBRARIES}
    TSYolov5s
)

#----------------------------------------------
add_subdirectory(alg)