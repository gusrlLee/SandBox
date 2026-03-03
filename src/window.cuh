#ifndef SANDBOX_WINDOW_CUDA_HEADER
#define SANDBOX_WINDOW_CUDA_HEADER

#include <iostream>

#include "glad/glad.h"
#include "glfw/glfw3.h"

// CUDA 및 CUDA-OpenGL 연동 헤더
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class Window 
{
public:
    Window(const char* title, int width, int height);
    ~Window();

    void init();
    bool shouldClose() { return glfwWindowShouldClose(m_Window); }

    void beginFrame(cudaSurfaceObject_t &frame);
    void endFrame(cudaSurfaceObject_t &frame);
    void draw();


    void swapBuffer()
    {
        glfwSwapBuffers(m_Window);
        glfwPollEvents(); 
    }

private:
    const char* m_Title;
    int m_Width, m_Height;
    GLFWwindow* m_Window;

    GLuint m_Fbo;
    GLuint m_Texture;
    cudaGraphicsResource* m_cuResource;
};

#endif // SANDBOX_WINDOW_CUDA_HEADER