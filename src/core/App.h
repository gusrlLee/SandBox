#ifndef SANDBOX_APPLICATION_HEADER
#define SANDBOX_APPLICATION_HEADER

#include <iostream>
#include <memory>

#include "core/Window.h"
#include "render/RenderContext.h"
#include "render/GLInterop.h"


class Application 
{
public:
	Application();
	~Application();

	void run();

private:
	std::unique_ptr<Window> m_Window;
	std::unique_ptr<RenderContext> m_RenderContext;
	std::unique_ptr<GLInterop> m_GLInterop;
};

#endif