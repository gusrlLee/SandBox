#include "render/GLInterop.h"
#include <iostream>

static const char* s_vert_source = R"(
    #version 330 core
    layout(location = 0) in vec3 vertexPosition_modelspace;
    out vec2 UV;
    void main() {
        gl_Position = vec4(vertexPosition_modelspace, 1.0);
        UV = (vec2(vertexPosition_modelspace.x, vertexPosition_modelspace.y) + vec2(1.0, 1.0)) / 2.0;
    }
    )";

static const char* s_frag_source = R"(
    #version 330 core
    in vec2 UV;
    out vec3 color;
    uniform sampler2D render_tex;
    void main() {
        color = texture(render_tex, UV).xyz;
    }
    )";


void GLInterop::init(int width, int height) 
{
    m_Width = width;
    m_Height = height;

    glGenBuffers(1, &m_Pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_Width * m_Height * 4 * sizeof(unsigned char), nullptr, GL_DYNAMIC_DRAW);    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Pbo, cudaGraphicsMapFlagsWriteDiscard);

    glGenTextures(1, &m_Tex);
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    unsigned int vs = compileShader(GL_VERTEX_SHADER, s_vert_source);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, s_frag_source);
    m_ShaderProgram = glCreateProgram();
    glAttachShader(m_ShaderProgram, vs);
    glAttachShader(m_ShaderProgram, fs);
    glLinkProgram(m_ShaderProgram);
    glDeleteShader(vs);
    glDeleteShader(fs);

    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f
    };

    glGenVertexArrays(1, &m_Vao);
    glGenBuffers(1, &m_Vbo);
    glBindVertexArray(m_Vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_Vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);
}

void* GLInterop::map() 
{
    void* d_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &m_CudaResource, 0);
    cudaGraphicsResourceGetMappedPointer(&d_ptr, &num_bytes, m_CudaResource);
    return d_ptr;
}

void GLInterop::unmap() 
{
    cudaGraphicsUnmapResources(1, &m_CudaResource, 0);
}

void GLInterop::draw() 
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Pbo);
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_ShaderProgram);
    glBindVertexArray(m_Vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_Tex);

    glUniform1i(glGetUniformLocation(m_ShaderProgram, "render_tex"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 6); 

    glBindVertexArray(0);
    glUseProgram(0);
}

void GLInterop::destroy() 
{
    cudaGraphicsUnregisterResource(m_CudaResource);
    glDeleteBuffers(1, &m_Pbo);
    glDeleteTextures(1, &m_Tex);
}

unsigned int GLInterop::compileShader(unsigned int type, const char* src)
{
    unsigned int id = glCreateShader(type);
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    return id;
}