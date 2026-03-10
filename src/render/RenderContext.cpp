#include "render/RenderContext.h"

#include <iostream>
#include <vector>

// for optix api
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
    std::cerr << "[" << level << "][" << tag << "]: " << message << "\n";
}


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
    options.logCallbackFunction = &context_log_cb;
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
    createPipeline();
    buildSBT();
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

void RenderContext::createPipeline()
{
    m_PipelineLinkOptions.maxTraceDepth = 1; 

    char log[2048];
    size_t sizeof_log = sizeof(log);

    std::vector<OptixProgramGroup> programGroups = { m_RayGenPG, m_MissPG, m_HitGroupPG };

    OptixResult res = optixPipelineCreate(
        m_Context,
        &m_PipelineCompileOptions,
        &m_PipelineLinkOptions,
        programGroups.data(),
        programGroups.size(),
        log,
        &sizeof_log,
        &m_Pipeline
    );

    if (res != OPTIX_SUCCESS) 
    {
        std::cerr << "Failed to create OptiX Pipeline! Log:\n" << log << "\n";
        exit(EXIT_FAILURE);
    }
}

void RenderContext::buildSBT()
{
    RaygenRecord rgRecord;
    optixSbtRecordPackHeader(m_RayGenPG, &rgRecord); // 헤더에 RayGen 프로그램 그룹 정보 삽입
    m_RaygenRecordsBuffer.allocAndUpload(&rgRecord, sizeof(RaygenRecord));

    m_SBT.raygenRecord = m_RaygenRecordsBuffer.getPtr();

    MissRecord msRecord;
    optixSbtRecordPackHeader(m_MissPG, &msRecord);
    m_MissRecordsBuffer.allocAndUpload(&msRecord, sizeof(MissRecord));

    m_SBT.missRecordBase = m_MissRecordsBuffer.getPtr();
    m_SBT.missRecordStrideInBytes = sizeof(MissRecord);
    m_SBT.missRecordCount = 1;

    HitgroupRecord hgRecord;
    optixSbtRecordPackHeader(m_HitGroupPG, &hgRecord);
    m_HitgroupRecordsBuffer.allocAndUpload(&hgRecord, sizeof(HitgroupRecord));

    m_SBT.hitgroupRecordBase = m_HitgroupRecordsBuffer.getPtr();
    m_SBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_SBT.hitgroupRecordCount = 1;
}

void RenderContext::render(uchar4* pboDevicePtr, uint32_t width, uint32_t height)
{
    // 1. LaunchParams 업데이트
    LaunchParams params;
    params.width = width;
    params.height = height;
    params.colorBuffer = pboDevicePtr;

    // 2. Host의 params 데이터를 Device 메모리로 복사
    m_LaunchParamsBuffer.upload(&params, sizeof(LaunchParams));

    // 3. OptiX Launch 실행
    // 디바이스에 올라간 파라미터 포인터를 넘겨줍니다.
    OptixResult res = optixLaunch(
        m_Pipeline,
        0,                                   // CUDA stream (0은 기본 스트림)
        m_LaunchParamsBuffer.getPtr(),    // 디바이스 상의 LaunchParams 메모리 주소
        m_LaunchParamsBuffer.size(),         // LaunchParams 크기
        &m_SBT,                              // 앞서 구축한 Shader Binding Table
        width,                               // Launch 크기 (X)
        height,                              // Launch 크기 (Y)
        1                                    // Launch 크기 (Z - 2D 이미지이므로 1)
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << "optixLaunch failed with error code: " << res << "\n";
    }

    // 디바이스 연산이 끝날 때까지 동기화 (초기 테스트용)
    cudaDeviceSynchronize();
}