cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} 
    main.cu
    math.cuh
    window.cuh
    window.cu
    renderer.cuh
    renderer.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)

target_link_libraries(${PROJECT_NAME} PUBLIC glad)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)
target_include_directories(${PROJECT_NAME} PUBLIC ${OptiX_INSTALL_DIR}/include)
