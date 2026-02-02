cmake_minimum_required(VERSION 3.30)

file(GLOB_RECURSE SOURCES
    "*.cpp"
    "*.h"
    "*.cu"
    "*.cuh"
)

add_executable(${PROJECT_NAME} ${SOURCES})