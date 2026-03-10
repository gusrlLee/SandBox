#ifndef SANDBOX_APPLICATION_HEADER
#define SANDBOX_APPLICATION_HEADER

#include <iostream>

#include "core/Window.h"
#include "render/RenderContext.h"

class Application 
{
public:
	Application();
	~Application();

	void run();

private:
	std::unique_ptr<Window> m_Window;
	std::unique_ptr<RenderContext> m_RenderContext;
};

#endif