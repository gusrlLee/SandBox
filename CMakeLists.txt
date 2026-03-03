cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
project(SandBox LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

function(compile_cuda_to_ptx target_name input_file output_ptx)
    add_custom_command(
        OUTPUT ${output_ptx}
        COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx 
        -I${OptiX_INSTALL_DIR}/include
        -I${CMAKE_SOURCE_DIR}/ext
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -arch=compute_75
        --use_fast_math
        -lineinfo
        --keep-device-functions
        -o ${output_ptx} 
        ${input_file}
        DEPENDS ${input_file}
        COMMENT "Compiling CUDA to PTX: ${input_file}"
    )
    add_custom_target(${target_name}_ptx DEPENDS ${output_ptx})
    add_dependencies(${SAMPLE_NAME} ${target_name}_ptx)
endfunction()


if(NOT DEFINED OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "Path to OptiX SDK")
endif()

message(STATUS "Using OptiX SDK at: ${OptiX_INSTALL_DIR}")

message(STATUS "---------------------------------------------------")
message(STATUS ">> Current CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS ">> CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---------------------------------------------------")

add_subdirectory(ext)
add_subdirectory(src)