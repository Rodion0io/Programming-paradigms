cmake_minimum_required(VERSION 3.24)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled1 parallel/main.cpp
        Functional/functions.cpp)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(untitled1 ${OpenCV_LIBS})





