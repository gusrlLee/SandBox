#ifndef SANDBOX_MESH_H
#define SANDBOX_MESH_H

#include <vector>
#include <string>
#include <cuda_runtime.h> 

#include "render/Buffer.h"

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texcoord;
};


class Mesh
{
public:
    Mesh();
    ~Mesh();

    bool loadFromFile(const std::string& fp);
    const std::vector<Vertex>& getVertices() const { return m_Vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_Indices; }

private:

    std::vector<Vertex> m_Vertices;
    std::vector<uint32_t> m_Indices;

    Buffer m_VertexBuffer;
    Buffer m_IndexBuffer;

};

#endif // SANDBOX_MESH_H