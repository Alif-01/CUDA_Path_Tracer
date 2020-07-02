#pragma once
#include <GL\glew.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <GLFW\glfw3.h>

#include "Renderer.h"
#include "Texture.h"

#include <map>

class Scene {

protected:
	Renderer renderer;
	CameraHost camera_host;
	bool reset_flag;

	std::vector<ObjectDef> object_defs;
	std::vector<Texture*> textures;
public:
	Scene(int width, int height, CameraHost camera_host);
	virtual ~Scene();

	virtual void ui(GLFWwindow *window);
	virtual void param_control();
	virtual const char* getWindowTitle() = 0;

	void render();
	void set_output_buffer(uchar3 *buffer);
	void init_renderer();
	void set_reset();
	void update_camera();
	void set_env_texture(cudaTextureObject_t tex);
	void add_object(Object *object, Material *material);
	int get_width();
	int get_height();

	std::map<std::string, Material*> load_mtllib(const char * file_name, const char * res_dir = "");
	cudaTextureObject_t load_texture(const char *file_name, bool load_hdr = false);
	vector<float3> load_points(const char *file_name, float scale = 1);

	void add_textured_rect(float3 p1, float3 p2, float3 p3, float3 p4, Material *mat, float scale, cudaTextureObject_t normal_map = 0);
	void add_textured_box(float3 p1, float3 p2, Material *mat, float scale, cudaTextureObject_t normal_map = 0);

	void save_result(const char *file_name);
};

