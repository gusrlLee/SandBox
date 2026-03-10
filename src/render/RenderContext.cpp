#include "render/RenderContext.h"

#include <iostream>

// for optix api
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

RenderContext::RenderContext() {}

RenderContext::~RenderContext()
{
	if (m_Context)
		optixDeviceContextDestroy(m_Context);
}

void RenderContext::init()
{
	cudaFree(0);

	if (optixInit() != OPTIX_SUCCESS)
	{
		std::cerr << "Failed to initialize OptiX!\n";
		exit(EXIT_FAILURE);
	}

	OptixDeviceContextOptions options = {};
#if _DEBUG
	options.logCallbackLevel = 4;
#else
	options.logCallbackLevel = 0;
#endif

	if (optixDeviceContextCreate(0, &options, &m_Context) != OPTIX_SUCCESS) 
	{
		std::cerr << "Failed to create OptiX Context!\n";
		exit(EXIT_FAILURE);
	}
}
