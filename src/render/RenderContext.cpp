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

	createModule();
    createProgramGroups();
}

void RenderContext::createModule()
{
    m_PipelineCompileOptions.usesMotionBlur = false;
    m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_PipelineCompileOptions.numPayloadValues = 3;
    m_PipelineCompileOptions.numAttributeValues = 2;
    m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    std::string ptxPath = "../ptx/PathTracingKernel.ptx";

    std::ifstream ptxFile(ptxPath);
    if (!ptxFile.is_open()) 
    {
        std::cerr << "Failed to find PTX file at: " << ptxPath << "\n";
        exit(EXIT_FAILURE);
    }

    std::stringstream ptxCode;
    ptxCode << ptxFile.rdbuf();
    std::string ptxString = ptxCode.str();

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixResult res = optixModuleCreate(
        m_Context,
        &moduleCompileOptions,
        &m_PipelineCompileOptions,
        ptxString.c_str(),
        ptxString.size(),
        log,
        &sizeof_log,
        &m_Module
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << "Failed to create OptiX Module! Log:\n" << log << "\n";
        exit(EXIT_FAILURE);
    }
}

void RenderContext::createProgramGroups()
{
    OptixProgramGroupOptions pgOptions = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    // 1. Raygen 프로그램 그룹 생성
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = m_Module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg"; // cu 파일의 함수명과 일치해야 함

    optixProgramGroupCreate(m_Context, &raygenDesc, 1, &pgOptions, log, &sizeof_log, &m_RayGenPG);

    // 2. Miss 프로그램 그룹 생성
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = m_Module;
    missDesc.miss.entryFunctionName = "__miss__ms";

    optixProgramGroupCreate(m_Context, &missDesc, 1, &pgOptions, log, &sizeof_log, &m_MissPG);

    // 3. Closest Hit 프로그램 그룹 생성 (HitGroup)
    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = m_Module; // CH = Closest Hit
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    optixProgramGroupCreate(m_Context, &hitgroupDesc, 1, &pgOptions, log, &sizeof_log, &m_HitGroupPG);
}