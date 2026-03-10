#ifndef SANDBOX_RENDER_CONTEXT_HEADER
#define SANDBOX_RENDER_CONTEXT_HEADER

#include <iostream>

#include <optix.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "shaders/Common.h"
#include "render/Buffer.h"

class RenderContext 
{
public:
	RenderContext();
	~RenderContext();

	void init();
	OptixDeviceContext getContext() { return m_Context; }

	void render(uchar4* pboDevicePtr, uint32_t width, uint32_t height);

private:

	void createModule();
	void createProgramGroups();
	void createPipeline();
	void buildSBT();

	OptixDeviceContext m_Context;
	OptixModule m_Module;

	OptixPipelineCompileOptions m_PipelineCompileOptions = {};
	OptixPipelineLinkOptions m_PipelineLinkOptions = {};
	OptixShaderBindingTable m_SBT = {};

	OptixProgramGroup m_RayGenPG = nullptr;
	OptixProgramGroup m_MissPG = nullptr;
	OptixProgramGroup m_HitGroupPG = nullptr;

	OptixPipeline m_Pipeline = nullptr;

	Buffer m_RaygenRecordsBuffer;
	Buffer m_MissRecordsBuffer;
	Buffer m_HitgroupRecordsBuffer;

	Buffer m_LaunchParamsBuffer;
};

#endif