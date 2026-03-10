#include "core/App.h"

Application::Application()
{
	m_Window = std::make_unique<Window>(1280, 720, "SandBox - Rendering Framework");

    m_RenderContext = std::make_unique<RenderContext>();
    m_RenderContext->init();
}

Application::~Application()
{

}

void Application::run()
{
    while (!m_Window->shouldClose())
    {
        m_Window->swapBuffer();
        m_Window->pollEvents();
    }
}