#ifndef SANDBOX_GL_INTEROP_HEADER
#define SANDBOX_GL_INTEROP_HEADER

#include "glad/glad.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GLInterop 
{
public:
    void init(int width, int height);
    void destroy();

    void* map();
    void unmap();

    void draw();

private:
    unsigned int m_Pbo = 0;
    unsigned int m_Tex = 0;

    cudaGraphicsResource_t m_CudaResource = nullptr;

    int m_Width = 0;
    int m_Height = 0;

    unsigned int m_ShaderProgram = 0;
    unsigned int m_Vao = 0;
    unsigned int m_Vbo = 0;

    unsigned int compileShader(unsigned int type, const char* src);
};

#endif