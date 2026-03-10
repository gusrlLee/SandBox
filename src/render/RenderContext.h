#ifndef SANDBOX_RENDER_CONTEXT_HEADER
#define SANDBOX_RENDER_CONTEXT_HEADER

#include <iostream>

#include <optix.h>

class RenderContext 
{
public:
	RenderContext();
	~RenderContext();

	void init();
	OptixDeviceContext getContext() { return m_Context; }

private:
	OptixDeviceContext m_Context;
};

#endif