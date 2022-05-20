include(FindPkgConfig)
pkg_check_modules(GST  REQUIRED gstreamer-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON REQUIRED json-glib-1.0)

add_library(AlgYolov5s
    SHARED
    AlgYolov5s.cpp
)

target_link_libraries(AlgYolov5s
    ${OpenCV_LIBS}
    ${GLIB_LIBRARIES}
    ${JSON_LIBRARIES}
    TSYolov5s
)

target_include_directories(AlgYolov5s
    PUBLIC
    ${GST_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

install(
    TARGETS AlgYolov5s
    LIBRARY DESTINATION /opt/thundersoft/algs/lib
)

install(
    FILES yolov5s.json
    DESTINATION /opt/thundersoft/configs
)