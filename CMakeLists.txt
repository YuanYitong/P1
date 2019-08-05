cmake_minimum_required(VERSION 3.13)
project(untitled)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(untitled_1 main.cpp)
TARGET_LINK_LIBRARIES(untitled_1 ${CERES_LIBRARIES})