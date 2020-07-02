#include "Scene.h"
#include "kernel.h"
#include "imgui\imgui.h"

#include <fstream>
#include <sstream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

Scene::Scene(int width, int height, CameraHost camera_host)
	:renderer(width, height, Camera(camera_host)), camera_host(camera_host) {
	reset_flag = false;
}

Scene::~Scene() {
	renderer.free();
	for (auto tex : textures) delete tex;
}

void Scene::ui(GLFWwindow * window) {
	ImGui::Begin(getWindowTitle());

	ImGui::Text("%.3f ms/frame %.1f FPS", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("%d spp", renderer.sample_count);

	const float move_speed = 5;
	const float theta_move_speed = 1;
	const float phi_move_speed = 0.6, phi_limit = 1.5;

	if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	float delta_time = ImGui::GetIO().DeltaTime;

	param_control();

	renderer.debug_flag = ImGui::IsKeyDown('C');

	if (ImGui::IsKeyDown('O')) save_result("output.bmp");

	if (ImGui::IsKeyDown('W')) {
		float3 front = camera_host.front();
		camera_host.eye += front * delta_time * move_speed;
		camera_host.center += front * delta_time * move_speed;
		set_reset();
	}

	if (ImGui::IsKeyDown('S')) {
		float3 front = camera_host.front();
		camera_host.eye -= front * delta_time  * move_speed;
		camera_host.center -= front * delta_time  * move_speed;
		set_reset();
	}

	if (ImGui::IsKeyDown('A')) {
		float3 right = camera_host.right();
		camera_host.eye -= right * delta_time * move_speed;
		camera_host.center -= right * delta_time * move_speed;
		set_reset();
	}

	if (ImGui::IsKeyDown('D')) {
		float3 right = camera_host.right();
		camera_host.eye += right * delta_time * move_speed;
		camera_host.center += right * delta_time * move_speed;
		set_reset();
	}

	if (ImGui::IsKeyDown(ImGui::GetKeyIndex(ImGuiKey_Space))) {
		camera_host.eye.y += delta_time * move_speed;
		camera_host.center.y += delta_time * move_speed;
		set_reset();
	}

	if (ImGui::GetIO().KeyShift) {
		camera_host.eye.y -= delta_time * move_speed;
		camera_host.center.y -= delta_time * move_speed;
		set_reset();
	}

	if (ImGui::IsKeyDown('J')) {
		float2 angle = camera_host.get_angle();
		angle.x -= ImGui::GetIO().DeltaTime * theta_move_speed;
		camera_host.set_angle(angle);
		set_reset();
	}

	if (ImGui::IsKeyDown('L')) {
		float2 angle = camera_host.get_angle();
		angle.x += ImGui::GetIO().DeltaTime * theta_move_speed;
		camera_host.set_angle(angle);
		set_reset();
	}

	if (ImGui::IsKeyDown('I')) {
		float2 angle = camera_host.get_angle();
		angle.y += ImGui::GetIO().DeltaTime * phi_move_speed;
		if (angle.y > phi_limit) angle.y = phi_limit;
		camera_host.set_angle(angle);
		set_reset();
	}

	if (ImGui::IsKeyDown('K')) {
		float2 angle = camera_host.get_angle();
		angle.y -= ImGui::GetIO().DeltaTime * phi_move_speed;
		if (angle.y < -phi_limit) angle.y = -phi_limit;
		camera_host.set_angle(angle);
		set_reset();
	}

	if (ImGui::IsKeyPressed('F')) {
		set_reset();
	}

	if (ImGui::IsKeyPressed('N')) {
		camera_host.save_camera("res/camera.txt");
	}

	if (ImGui::IsKeyPressed('M')) {
		camera_host.load_camera("res/camera.txt");
		set_reset();
	}

	update_camera();

	//if (ImGui::IsAnyMouseDown()) {
	//	ImVec2 mouse_pos = ImGui::GetMousePos();
	//	printf("mouse clicked: %.2f %.2f\n", mouse_pos.x / 2.5, renderer.screen_height - mouse_pos.y / 2.5);
	//}

	ImGui::End();
}

void Scene::param_control() {
	ImGui::DragFloat("exposure", &camera_host.exposure, 0.01f, 0.0f, 10.0f);
	ImGui::DragFloat("gamma", &camera_host.gamma, 0.001f, 0.0f, 10.0f);
	ImGui::DragFloat("focus_length", &camera_host.focus_length, 0.01f, 0.0f, 100.0f);
	ImGui::DragFloat("aperture", &camera_host.aperture, 0.001f, 0.0f, 10.0f);
	ImGui::DragFloat("fovy", &camera_host.fovy, 1, 0.0f, 180.0f);
	ImGui::DragFloat("env_rotate", &camera_host.env_rotate, 0.001f, 0.0f, 1.0f);
}

void Scene::render() {
	if(reset_flag){
		kernel_clear(renderer);
		reset_flag = false;
	}
	kernel_render(renderer);
}

void Scene::set_output_buffer(uchar3 * buffer) {
	renderer.out_buffer = buffer;
}

void Scene::init_renderer() {
	renderer.create_buffer(object_defs);

	kernel_init_curand_state(renderer);
}

void Scene::set_reset() {
	reset_flag = true;
}

void Scene::update_camera() {
	renderer.camera.update_from_host(camera_host);
}

void Scene::set_env_texture(cudaTextureObject_t tex) {
	renderer.env_texture = tex;
}

void Scene::add_object(Object * object, Material * material) {
	//object->get_AABB().debug_print();
	ObjectDef def;
	def.obj = object;
	def.mat = material;
	object_defs.push_back(def);
}

cudaTextureObject_t Scene::load_texture(const char * file_name, bool load_hdr) {
	Texture *tex = new Texture(file_name, cudaFilterModePoint, load_hdr);

	textures.push_back(tex);

	return tex->texture_obj;
}

vector<float3> Scene::load_points(const char * file_name, float scale) {
	using std::string;

	std::ifstream f(file_name);
	vector<float3> res;
	string line;

	while (std::getline(f, line)) {
		std::istringstream is(line);
		float x, y, z;
		is >> x >> y >> z;
		res.push_back(make_float3(x, y, z)*scale);
	}

	return res;
}

void Scene::add_textured_rect(float3 p1, float3 p2, float3 p3, float3 p4, Material * mat, float scale, cudaTextureObject_t normal_map) {
	float3 norm = normalize(cross(p3 - p2, p1 - p2));
	float2 uv1 = vec2(0, 0);
	float2 uv2 = vec2(0, length(p2 - p1)*scale);
	float2 uv3 = vec2(length(p4 - p1)*scale, length(p2 - p1)*scale);
	float2 uv4 = vec2(length(p4 - p1)*scale, 0);
	add_object(new Triangle(p1, p2, p3, norm, norm, norm, uv1, uv2, uv3, normal_map), mat);
	add_object(new Triangle(p1, p3, p4, norm, norm, norm, uv1, uv3, uv4, normal_map), mat);
}

void Scene::add_textured_box(float3 p1, float3 p2, Material * mat, float scale, cudaTextureObject_t normal_map) {
	float3 t1 = vec3(p1.x, p1.y, p1.z), t3 = vec3(p2.x, p1.y, p1.z);
	float3 t2 = vec3(p1.x, p1.y, p2.z), t4 = vec3(p2.x, p1.y, p2.z);
	float3 t5 = vec3(p1.x, p2.y, p1.z), t7 = vec3(p2.x, p2.y, p1.z);
	float3 t6 = vec3(p1.x, p2.y, p2.z), t8 = vec3(p2.x, p2.y, p2.z);
	add_textured_rect(t5, t6, t8, t7, mat, scale, normal_map);
	add_textured_rect(t6, t2, t4, t8, mat, scale, normal_map);
	add_textured_rect(t5, t1, t2, t6, mat, scale, normal_map);
	add_textured_rect(t8, t4, t3, t7, mat, scale, normal_map);
	add_textured_rect(t7, t3, t1, t5, mat, scale, normal_map);
	add_textured_rect(t2, t1, t3, t4, mat, scale, normal_map);
}

void Scene::save_result(const char * file_name) {
	float3* result_buf = new float3[get_width()*get_height()];
	CUDA_CHECK(cudaMemcpy(result_buf, renderer.acc_buffer, get_width()*get_height() * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	uchar * host_result = new uchar[get_width()*get_height() * 3];
	for (int i = 0;i < get_height();i++)
		for (int j = 0;j < get_width();j++){
			float3 tc = result_buf[(get_height() - 1 - i)*get_width() + j];
			tc = pow(tc / renderer.sample_count, 1.0 / camera_host.gamma);
			host_result[(i*get_width() + j) * 3 + 0] = (uchar)(tc.x * 255);
			host_result[(i*get_width() + j) * 3 + 1] = (uchar)(tc.y * 255);
			host_result[(i*get_width() + j) * 3 + 2] = (uchar)(tc.z * 255);
		}
	delete[] result_buf;

	stbi_write_bmp(file_name, get_width(), get_height(), 3, host_result);
	delete[] host_result;
}

int Scene::get_width() {
	return renderer.screen_width;
}

int Scene::get_height() {
	return renderer.screen_height;
}

std::map<std::string, Material*> Scene::load_mtllib(const char * file_name, const char * res_dir) {
	using std::string;

	std::map<std::string, Material*> res;
	std::ifstream f(file_name);

	string line;
	string cur_mat = "";

	float3 Kd, Ks, Ke;
	float Ns;

	while (std::getline(f, line)) {
		std::istringstream is(line);
		string op;
		if (is.peek() == '#') continue;
		is >> op;
		if (op == "newmtl") {
			is >> cur_mat;
			res[cur_mat] = new UE4CookTorrance(0, 0.01, vec3(1), 0);
			//res[cur_mat] = new Glass(vec3(1), 1.55);
			Ks = Ke = Kd = vec3();
			Ns = 1;
		}
		if (op == "Kd") {
			is >> Kd.x >> Kd.y >> Kd.z;
			if (res[cur_mat]) delete res[cur_mat];
			res[cur_mat] = convert_mtllib(Kd, Ks, Ke, Ns);
		}
		if (op == "Ks") {
			is >> Ks.x >> Ks.y >> Ks.z;
			if (res[cur_mat]) delete res[cur_mat];
			res[cur_mat] = convert_mtllib(Kd, Ks, Ke, Ns);
		}
		if (op == "Ke") {
			is >> Ke.x >> Ke.y >> Ke.z;
			if (res[cur_mat]) delete res[cur_mat];
			res[cur_mat] = convert_mtllib(Kd, Ks, Ke, Ns);
		}
		if (op == "Ns") {
			is >> Ns;
			if (res[cur_mat]) delete res[cur_mat];
			res[cur_mat] = convert_mtllib(Kd, Ks, Ke, Ns);
		}
		if (op == "BRDF") {
			UE4CookTorrance * t = (UE4CookTorrance*)res[cur_mat];
			is >> t->metallic >> t->roughness;
			t->albedo = vec3(1);
		}
		if (op == "map_Kd") {
			string tex_file;
			is >> tex_file;
			cudaTextureObject_t tex = load_texture((res_dir + tex_file).c_str(), true);
			((UE4CookTorrance*)res[cur_mat])->texture_albedo = tex;
		}
		if (op == "texture") {
			string tex_file;
			is >> tex_file;
			((UE4CookTorrance*)res[cur_mat])->texture_albedo = load_texture((res_dir + tex_file).c_str(), true);
			//((Glass*)res[cur_mat])->texture_albedo = load_texture(tex_file.c_str());
		}
	}

	return res;
}
