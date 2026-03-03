#include "Window.cuh"

Window::Window(const char* title, int width, int height)
{
    m_Width = width; m_Height = height;
    glfwInit();
    m_Window = glfwCreateWindow(m_Width, m_Height, title, nullptr, nullptr);

    if (!m_Window)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create glfw window!\n");
    }
    
    glfwMakeContextCurrent(m_Window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        glfwTerminate();
        throw std::runtime_error("Failed to initailization glad library!\n");
    }

    init();
}

Window::~Window()
{
    cudaGraphicsUnregisterResource(m_Frame);
    glDeleteTextures(1, &m_Texture);

    if (m_Window)
        glfwDestroyWindow(m_Window);
    glfwTerminate();
}


void Window::init()
{
    glGenTextures(1, &m_Texture);
    glBindTexture(GL_TEXTURE_2D, m_Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    cudaGraphicsGLRegisterImage(&m_Frame, m_Texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    glGenFramebuffers(1, &m_Fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_Fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_Texture, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}