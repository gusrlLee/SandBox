#include <iostream>

#include "window.cuh"


int main()
{
    std::shared_ptr<Window> window = std::make_shared<Window>("Check", 720, 480);

    while (!window->shouldClose())
    {
        window->update();
    }

    return 0; 
}