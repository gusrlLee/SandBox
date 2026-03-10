#include "core/Window.h"

Window::Window(int width, int height, const char* title)
{
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW\n";
		exit(EXIT_FAILURE);
	}

	m_Width = width;
	m_Height = height;
	m_Title = title;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_Window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (!m_Window) 
	{
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(m_Window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) 
	{
		std::cerr << "Failed to initialize GLAD\n";
		exit(EXIT_FAILURE);
	}
}

Window::~Window()
{
	glfwDestroyWindow(m_Window);
	glfwTerminate();
}

bool Window::shouldClose() const
{
	return glfwWindowShouldClose(m_Window);
}

void Window::pollEvents() const
{
	glfwPollEvents();
}

void Window::swapBuffer() const
{
	glfwSwapBuffers(m_Window);
}
