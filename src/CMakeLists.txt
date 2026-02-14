cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

file(GLOB_RECURSE SOURCES
    common.cu
    common.cuh
    main.cu
    math_helpder.cuh
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/external)