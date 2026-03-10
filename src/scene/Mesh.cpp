#include "scene/Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

#include <iostream>

Mesh::Mesh() {}

Mesh::~Mesh() {}

bool Mesh::loadFromFile(const std::string& fp)
{
	tinyobj::ObjReaderConfig cfg;
	cfg.mtl_search_path = "./";
	cfg.triangulate = true;

	tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(fp, cfg))
    {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader Error: " << reader.Error();
        }
        return false;
    }

    if (!reader.Warning().empty()) 
    {
        std::cout << "TinyObjReader Warning: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    uint32_t current_index = 0;

    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            for (size_t v = 0; v < fv; v++)
            {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                Vertex vertex = {};

                vertex.position.x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                vertex.position.y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                vertex.position.z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                if (idx.normal_index >= 0) {
                    vertex.normal.x = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    vertex.normal.y = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    vertex.normal.z = attrib.normals[3 * size_t(idx.normal_index) + 2];
                }

                if (idx.texcoord_index >= 0) {
                    vertex.texcoord.x = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    vertex.texcoord.y = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

                // 배열에 추가
                m_Vertices.push_back(vertex);
                m_Indices.push_back(current_index++);
            }
            index_offset += fv;
        }
    }

    std::cout << "Successfully loaded mesh: " << fp
        << " (Vertices: " << m_Vertices.size() << " / Triangles: " << m_Indices.size() / 3 << ")\n";

    return true;
}
