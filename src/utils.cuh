#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cmath>

#include "structs.cuh"

#include "stb/stb_image.h"
#include "tinyobjloader/tiny_obj_loader.h"

__host__ Scene loadScene(const std::string &fp, const std::string &mtlFp)
{
    Scene scn;
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = mtlFp;
    cfg.triangulate = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(fp, cfg))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "TinyObjReader Error: " << reader.Error() << std::endl;
        }
        return scn;
    }

    if (!reader.Warning().empty())
    {
        std::cerr << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    for (const auto &tMat : materials)
    {
        Material m;
        m.albedo = make_float3(tMat.diffuse[0], tMat.diffuse[1], tMat.diffuse[2]);
        m.emission = make_float3(tMat.emission[0], tMat.emission[1], tMat.emission[2]);

        // for checking light
        float emissionIntensity = m.emission.x + m.emission.y + m.emission.z;

        if (emissionIntensity > 1e-4f)
        {
            m.type = eDIFFUSE_LIGHT;
        }
        else if (tMat.illum == 5)
        {
            m.type = eSPECULAR;
        }
        else if (tMat.illum == 7)
        {
            m.type = eDIELECTRIC;
        }
        else
        {
            m.type = eLAMBERTIAN;
        }

        scn.materials.push_back(m);
    }

    if (scn.materials.empty())
    {
        std::cerr << "Materials of scene is empty." << std::endl;
        Material defaultMat;
        defaultMat.albedo = make_float3(0.7f, 0.7f, 0.7f); // 밝은 회색
        defaultMat.emission = make_float3(0.0f, 0.0f, 0.0f);
        defaultMat.type = eLAMBERTIAN;
        scn.materials.push_back(defaultMat);
    }

    for (const auto &shape : shapes)
    {
        size_t idxOffset = 0;
        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); face++)
        {
            int fv = shape.mesh.num_face_vertices[face];

            float3 vertices[3];
            for (int i = 0; i < 3; i++)
            {
                tinyobj::index_t idx = shape.mesh.indices[idxOffset + i];

                vertices[i] = make_float3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);
            }

            int matId = shape.mesh.material_ids[face];
            if (matId < 0 || matId >= scn.materials.size())
            {
                matId = 0;
            }

            scn.triangles.push_back(Triangle(vertices[0], vertices[1], vertices[2], matId));
            idxOffset += fv;
        }
    }
    return scn;
}

void download(std::string filename, int width, int height, float3 *data)
{
    std::cout << "[Info] Processing and saving image..." << std::endl;
    std::vector<unsigned char> image(width * height * 3);

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            size_t pixel_index = (height - 1 - j) * width + i;
            float3 pixel = data[pixel_index];

            float r = sqrt(pixel.x);
            float g = sqrt(pixel.y);
            float b = sqrt(pixel.z);

            // Gamma Correction
            int ir = int(255.99f * (r > 1.0f ? 1.0f : (r < 0.0f ? 0.0f : r)));
            int ig = int(255.99f * (g > 1.0f ? 1.0f : (g < 0.0f ? 0.0f : g)));
            int ib = int(255.99f * (b > 1.0f ? 1.0f : (b < 0.0f ? 0.0f : b)));

            size_t img_index = (j * width + i) * 3;
            image[img_index + 0] = static_cast<unsigned char>(ir);
            image[img_index + 1] = static_cast<unsigned char>(ig);
            image[img_index + 2] = static_cast<unsigned char>(ib);
        }
    }

    if (stbi_write_png(filename.c_str(), width, height, 3, image.data(), width * 3))
    {
        std::cout << "[Info] Image saved successfully: " << filename << std::endl;
    }
    else
    {
        std::cerr << "[Error] Failed to save image!" << std::endl;
    }
}

#endif // UTILS_H