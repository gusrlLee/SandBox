cmake_minimum_required(VERSION 3.30)

file(GLOB_RECURSE SOURCES
    "*.cpp"
    "*.h"
    "*.cu"
    "*.cuh"
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC tiny-cuda-nn)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/lib/tiny-cuda-nn/dependencies)