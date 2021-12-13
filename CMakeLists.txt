cmake_minimum_required(VERSION 3.20)
project(hw5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(include)

add_executable(hw5 src/main.cpp src/graphics.cpp include/graphics.h)
