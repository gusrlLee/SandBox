#include <iostream>

#include "core/App.h"

#include "scene/Mesh.h"

int main()
{
    Mesh mesh;
    if (!mesh.loadFromFile("../../scn/CornellBox/CornellBox-Original.obj"))
    {
        std::runtime_error("Failed to load obj file!\n");
        return -1;
    }

    std::unique_ptr<Application> app = std::make_unique<Application>();
    app->run();

    return 0;
}
