cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

file(GLOB_RECURSE SOURCES
    main.cu
    math.cuh
    window.cuh
    window.cu
)

add_executable(${PROJECT_NAME} 
    main.cu
    math.cuh
    window.cuh
    window.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)

target_link_libraries(${PROJECT_NAME} PUBLIC glad)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
