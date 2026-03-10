#ifndef SANDBOX_RENDER_CONTEXT_HEADER
#define SANDBOX_RENDER_CONTEXT_HEADER

#include <iostream>

#include <optix.h>

#include <iostream>
#include <fstream>
#include <sstream>

class RenderContext 
{
public:
	RenderContext();
	~RenderContext();

	void init();
	OptixDeviceContext getContext() { return m_Context; }

private:

	void createModule();
	void createProgramGroups();

	OptixDeviceContext m_Context;
	OptixModule m_Module;
	OptixPipelineCompileOptions m_PipelineCompileOptions = {};

	OptixProgramGroup m_RayGenPG = nullptr;
	OptixProgramGroup m_MissPG = nullptr;
	OptixProgramGroup m_HitGroupPG = nullptr;

};

#endif