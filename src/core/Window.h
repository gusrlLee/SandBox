#ifndef SANDBOX_WINDOW_HEADER
#define SANDBOX_WINDOW_HEADER

#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

class Window 
{
public:
	Window(int width, int height, const char* title);
	~Window();

	bool shouldClose() const;
	void pollEvents() const;
	void swapBuffer() const;

	GLFWwindow* getHandle() const { return m_Window; }
private:
	GLFWwindow* m_Window;
	int m_Width, m_Height;
	const char* m_Title;
};

#endif // SANDBOX_WINDOW_HEADER