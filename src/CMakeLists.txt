cmake_minimum_required(VERSION 3.30)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} 
	main.cpp

	core/Window.h
	core/Window.cpp

	core/App.h
	core/App.cpp

	render/RenderContext.h
	render/RenderContext.cpp

	render/GLInterop.h
	render/GLInterop.cpp

	render/TestKernel.h
	render/TestKernel.cu
	render/Buffer.h

	shaders/Common.h
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OptiX_INSTALL_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PUBLIC glad)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

compile_cuda_to_ptx(
    PathTracingKernel                                             # target_name
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/PathTracingKernel.cu      # input_file
    ${PTX_OUTPUT_DIR}/PathTracingKernel.ptx                       # output_ptx
)