#include "Window.cuh"

Window::Window(const char *title, int width, int height)
{
    m_Title = title;
    m_Width = width;
    m_Height = height;

    init();
}

Window::~Window()
{
    cudaGraphicsUnregisterResource(m_cuResource);
    glDeleteTextures(1, &m_Texture);

    if (m_Window)
        glfwDestroyWindow(m_Window);
    glfwTerminate();
}

void Window::init()
{
    glfwInit();
    m_Window = glfwCreateWindow(m_Width, m_Height, m_Title, nullptr, nullptr);

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

    glGenTextures(1, &m_Texture);
    glBindTexture(GL_TEXTURE_2D, m_Texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    cudaGraphicsGLRegisterImage(&m_cuResource, m_Texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore);

    glGenFramebuffers(1, &m_Fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_Fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_Texture, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Window::beginFrame(cudaSurfaceObject_t &frame)
{
    cudaGraphicsMapResources(1, &m_cuResource, 0);
    cudaArray_t cuArray;
    cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cuResource, 0, 0);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaCreateSurfaceObject(&frame, &resDesc);
}

void Window::endFrame(cudaSurfaceObject_t &frame)
{
    cudaDestroySurfaceObject(frame);
    cudaGraphicsUnmapResources(1, &m_cuResource, 0);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_Fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); 

    glBlitFramebuffer(0, 0, m_Width, m_Width,
                      0, 0, m_Height, m_Height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Window::swapBuffer()
{
    glfwSwapBuffers(m_Window);
    glfwPollEvents(); 
}