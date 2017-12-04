cmake_minimum_required(VERSION 3.2)
project(dimensionHeadTracking)

set(CMAKE_CXX_STANDARD 11)

find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIB_WEBSOCKETS REQUIRED libwebsockets)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp server.cpp)
add_executable(dimensionHeadTracking ${SOURCE_FILES})

target_link_libraries(dimensionHeadTracking ${freenect2_LIBRARIES} ${OpenCV_LIBS}  ${LIB_WEBSOCKETS_LIBRARIES})

file(COPY websocket.html DESTINATION .)
