#pragma once
#include "util.h"
#include <fstream>

struct CameraHost {
	float3 eye, center, up;
	float fovy, aspect;
	float focus_length, aperture;
	float exposure, gamma;
	float env_rotate;

	__host__ float3 front() {
		float theta = get_angle().x;
		return vec3(cos(theta), 0, sin(theta));
	}

	__host__ float3 right() {
		float theta = get_angle().x;
		return vec3(-sin(theta), 0, cos(theta));
	}

	__host__ float2 get_angle() {
		float3 d = normalize(center - eye);
		float theta = atan2f(d.z, d.x);
		float phi = asin(d.y);
		return make_float2(theta, phi);
	}

	__host__ void set_angle(float2 ang) {
		float theta = ang.x, phi = ang.y;
		float3 d = vec3(cos(theta)*cos(phi), sin(phi), sin(theta)*cos(phi));
		center = eye + d;
	}

	__host__ CameraHost(float3 eye, float3 center, float3 up, float fovy, float aspect, float focus_length = 1, float aperture = 0) {
		center = eye + normalize(center - eye);
		this->eye = eye;
		this->center = center;
		this->up = up;
		this->fovy = fovy;
		this->aspect = aspect;
		this->exposure = 1.0f;
		this->gamma = 2.2f;
		this->env_rotate = 0;
		this->focus_length = focus_length;
		this->aperture = aperture;
	}

	__host__ void save_camera(const char *filename) {
		std::ofstream f(filename);
		f << eye.x << ' ' << eye.y << ' ' << eye.z << std::endl;
		f << center.x << ' ' << center.y << ' ' << center.z << std::endl;
		f << up.x << ' ' << up.y << ' ' << up.z << std::endl;
		f << fovy << std::endl;
		f << aspect << std::endl;
		f << exposure << std::endl;
		f << gamma << std::endl;
		f << focus_length << std::endl;
		f << aperture << std::endl;
		f << env_rotate << std::endl;
	}

	__host__ void load_camera(const char *filename) {
		std::ifstream f(filename);
		f >> eye.x >> eye.y >> eye.z;
		f >> center.x >> center.y >> center.z;
		f >> up.x >> up.y >> up.z;
		f >> fovy >> aspect >> exposure >> gamma >> focus_length >> aperture >> env_rotate;
	}
};

struct Camera {
	float3 t0, tx, ty;
	float3 eye;
	float3 unit_x, unit_y;
	float exposure, gamma;
	float aperture;
	float env_rotate;

	__device__ __host__ void update_from_host(CameraHost host) {
		eye = host.eye;

		float3 unit_z = normalize(host.center - eye);
		unit_x = normalize(cross(unit_z, host.up));
		unit_y = normalize(cross(unit_x, unit_z));
	
		float h = tanf(host.fovy*PI / 360);
		float w = h*host.aspect;
	
		t0 = unit_z - w / 2 * unit_x - h / 2 * unit_y;
		tx = w*unit_x;
		ty = h*unit_y;

		t0 *= host.focus_length;
		tx *= host.focus_length;
		ty *= host.focus_length;

		aperture = host.aperture;
		exposure = host.exposure;
		gamma = host.gamma;
		env_rotate = host.env_rotate;
	}

	__device__ __host__ Camera(CameraHost host) {
		update_from_host(host);
	}

	__device__ __host__ Camera(float3 p_eye, float3 p_center, float3 p_up, float fovy, float aspect) {
		CameraHost host(p_eye, p_center, p_up, fovy, aspect);
		update_from_host(host);
	}

	__device__ Ray get_ray(float x, float y, curandState *state) {
		float theta = curand_uniform(state)*PI * 2, r = sqrtf(curand_uniform(state))*aperture;
		float3 p = t0 + tx*x + ty*y, origin = (cosf(theta)*unit_x + sinf(theta)*unit_y)*r;
		return Ray(eye + origin, normalize(p - origin));
	}
};

