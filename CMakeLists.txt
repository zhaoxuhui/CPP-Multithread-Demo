cmake_minimum_required(VERSION 3.1)
project(multiThread)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(multiThread main.cpp)
# 由于pthread库不是Linux系统默认的库，连接时需要指定一下，不然编译不通过
target_link_libraries(multiThread pthread ${OpenCV_LIBS})
