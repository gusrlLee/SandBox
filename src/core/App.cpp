#include "core/App.h"

#include "render/TestKernel.h"

Application::Application()
{
	m_Window = std::make_unique<Window>(1280, 720, "SandBox - Rendering Framework");

    m_RenderContext = std::make_unique<RenderContext>();
    m_RenderContext->init();

    m_GLInterop = std::make_unique<GLInterop>();
    m_GLInterop->init(m_Window->getWidth(), m_Window->getHeight());
}

Application::~Application()
{
    if (m_GLInterop)
    {
        m_GLInterop->destroy();
    }
}

void Application::run()
{
    while (!m_Window->shouldClose())
    {
        void* dPbo = m_GLInterop->map();
        
        m_RenderContext->render(
            static_cast<uchar4*>(dPbo),
            m_Window->getWidth(),
            m_Window->getHeight()
        );

        m_GLInterop->unmap();
        m_GLInterop->draw();

        m_Window->swapBuffer();
        m_Window->pollEvents();
    }
}