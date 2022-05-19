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

install(
    TARGETS AlgYolov5s
    LIBRARY DESTINATION /opt/thundersoft/algs/lib
)

install(
    FILES yolov5s.json
    DESTINATION /opt/thundersoft/configs
)