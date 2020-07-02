#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include <stdio.h>
#include <cstdlib>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "Scene.h"
#include "TestScene.h"
#include "kernel.h"
#include "util.h"

GLFWwindow *window;
int current_width, current_height;

GLuint pbo_id, texture_id;
cudaGraphicsResource *resource;

Scene *scene;

static void glfw_error_callback(int error, const char* description) {
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
	while (1);
}

void render_gui() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	scene->ui(window);

	ImGui::Render();

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void render_texture() {

	uchar3 *buf;
	size_t buf_size;

	CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&buf, &buf_size, resource));

	scene->set_output_buffer(buf);
	scene->render();

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, scene->get_width(), scene->get_height(), GL_RGB, GL_UNSIGNED_BYTE, nullptr);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void render_quad() {
	glViewport(0, 0, current_width, current_height);

	glBindTexture(GL_TEXTURE_2D, texture_id);

	glBegin(GL_QUADS);
	glTexCoord2i(0, 0); glVertex2i(-1, -1);
	glTexCoord2i(0, 1); glVertex2i(-1, 1);
	glTexCoord2i(1, 1); glVertex2i(1, 1);
	glTexCoord2i(1, 0); glVertex2i(1, -1);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
}

int init_GL() {
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, 1);

	float scale_w, scale_h;

	GLFWmonitor *monitor = glfwGetPrimaryMonitor();
	glfwGetMonitorContentScale(monitor, &scale_w, &scale_h);

	window = glfwCreateWindow(scene->get_width(), scene->get_height(), scene->getWindowTitle(), NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	bool err = glewInit() != GLEW_OK;
	if (err) {
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.FontGlobalScale = scale_w;

	ImGui::StyleColorsDark();
	ImGui::GetStyle().ScaleAllSizes(scale_w);

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");

	return 0;
}

void init_CUDA() {
	int dev_id;
	
	cudaDeviceProp prop;
	memset(&prop, 0, sizeof(prop));
	prop.major = 1;
	prop.minor = 0;
	CUDA_CHECK(cudaChooseDevice(&dev_id, &prop));

}

void init_Buffer_And_Texture() {
	glGenBuffers(1, &pbo_id);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, scene->get_width()*scene->get_height() * 3, nullptr, GL_DYNAMIC_DRAW_ARB);

	cudaGraphicsGLRegisterBuffer(&resource, pbo_id, cudaGLMapFlagsNone);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scene->get_width(), scene->get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
}

int main(int argc, char **argv) {
	scene = new CornellBoxScene(1260, 720);

	if (init_GL()) return 1;
	init_CUDA();

	init_Buffer_And_Texture();

	scene->init_renderer();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		glfwGetWindowSize(window, &current_width, &current_height);

		render_texture();
		render_quad();
		render_gui();

		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	delete scene;

	return 0;
}
