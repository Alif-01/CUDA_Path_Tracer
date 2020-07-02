#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cutil_math.h>

const __device__ float PI = 3.14159265359;

typedef unsigned char uchar;

#define CUDA_CHECK(error) \
if (error != cudaSuccess) { \
	fprintf(stderr, "CUDA ERROR %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
	while (1); \
	exit(-1); \
}

#ifndef __CUDACC__
template<typename T> T tex2DLayered(cudaTextureObject_t tex, float u, float v, int layer) {}
#endif

struct Ray {
	float3 p, d;

	__device__ __host__ Ray(float3 p_p, float3 p_d) :p(p_p), d(p_d) {}

	__device__ __host__ float3 at(float k) {
		return p + d*k;
	}

	__device__ __host__ void normalize() {
		d = ::normalize(d);
	}
};

struct Hit {
	int material_id, object_id;
	float3 normal;
	float2 uv;
	float t;

	__device__ __host__ Hit(int mat_id, int obj_id, float3 p_n, float p_t) :material_id(mat_id), object_id(obj_id), normal(p_n), t(p_t) {}
};

template<typename T> __host__ T* malloc_host(int size) {
	return (T*)malloc(sizeof(T)*size);
}

template<typename T> __host__ T* malloc_device(int size) {
	T* dev_ptr;
	CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(T)*size));
	return dev_ptr;
}

template<typename T> __host__ T* copy_to_device(T* host_ptr, int size) {
	T* dev_ptr;
	CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(T)*size));
	CUDA_CHECK(cudaMemcpy(dev_ptr, host_ptr, sizeof(T)*size, cudaMemcpyHostToDevice));
	return dev_ptr;
}

template<typename T> __host__ void assign_host(T* dev_ptr, T v) {
	*dev_ptr = v;
}

template<typename T> __host__ void assign_device(T* dev_ptr, T v) {
	CUDA_CHECK(cudaMemcpy(dev_ptr, &v, sizeof(T), cudaMemcpyHostToDevice));
}

inline __host__ void free_device(void *dev_str) {
	CUDA_CHECK(cudaFree(dev_str));
}

__device__ __host__ __inline__ float2 vec2() {
	return make_float2(0);
}

__device__ __host__ __inline__ float2 vec2(float x) {
	return make_float2(x, x);
}

__device__ __host__ __inline__ float2 vec2(float x, float y) {
	return make_float2(x, y);
}

__device__ __host__ __inline__ float3 vec3() {
	return make_float3(0);
}

__device__ __host__ __inline__ float3 vec3(float x) {
	return make_float3(x, x, x);
}

__device__ __host__ __inline__ float3 vec3(float x, float y) {
	return make_float3(x, y, 0);
}

__device__ __host__ __inline__ float3 vec3(float x, float y, float z) {
	return make_float3(x, y, z);
}

__device__ __host__ __inline__ float3 pow(float3 x, float e){
	return make_float3(pow(x.x, e), pow(x.y, e), pow(x.z, e));
}

__device__ __host__ __inline__ float3 exp(float3 x){
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

__device__ __host__ __inline__ float max3(float x, float y, float z) {
	return (x > y && x > z) ? x : (y > z ? y : z);
}

__device__ __host__ __inline__ float max3(float3 x) {
	return (x.x > x.y && x.x > x.z) ? x.x : (x.y > x.z ? x.y : x.z);
}

__device__ __host__ __inline__ void swap(float &x, float &y) {
	float t = x;
	x = y; y = t;
}

__device__ __host__ __inline__ void print_float3(float3 x) {
	printf("(%.5f,%.5f,%.5f) ", x.x, x.y, x.z);
}

__device__ __inline__ float3 uniform_sphere(curandState *state) {
	float z = curand_uniform(state) * 2 - 1;
	float r = sqrtf(1 - z*z);
	float theta = curand_uniform(state)*PI * 2;
	return make_float3(r*cos(theta), r*sin(theta), z);
}

__device__ __inline__ float3 uniform_sphere_2(curandState *state) {
	float3 d = make_float3(curand_uniform(state) * 2 - 1, curand_uniform(state) * 2 - 1, curand_uniform(state) * 2 - 1);
	while (dot(d, d) > 1)
		d = make_float3(curand_uniform(state) * 2 - 1, curand_uniform(state) * 2 - 1, curand_uniform(state) * 2 - 1);
	return normalize(d);
}

__device__ __inline__ float3 sample_hemisphere_uniform(curandState *state, float3 normal) {
	float3 p = uniform_sphere(state);
	return dot(p, normal) > 0 ? p : -p;
}

__device__ __inline__ float3 sample_hemisphere_cosine(curandState *state, float3 normal){
	float3 U = fabs(normal.z) < 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0);
	float3 V = normalize(cross(U, normal));
	U = cross(normal, V);
	float y = sqrtf(curand_uniform(state));
	float r = sqrtf(fmaxf(0.0f, 1.0f - y*y));
	float theta = curand_uniform(state) * PI * 2;
	return normal*y + U*r*cos(theta) + V*r*sin(theta);
}

__host__ __device__ __inline__ bool debug_pos(int x, int y) {
#ifdef __CUDACC__
	return blockIdx.x*blockDim.x + threadIdx.x == x && blockIdx.y*blockDim.y + threadIdx.y == y;
#else
	return 0;
#endif
}

__device__ __inline__ bool debug_left(){
	return (blockIdx.x*blockDim.x + threadIdx.x) * 2 < blockDim.x*gridDim.x;
}

__device__ __host__ __inline__ float mean(float3 x) {
	return (x.x + x.y + x.z) / 3;
}

__device__ __inline__ float sample_tex_float(cudaTextureObject_t tex, float2 p) {
	return tex2DLayered<float>(tex, p.x, p.y, 0);
}

__device__ __inline__ float3 sample_tex_float3(cudaTextureObject_t tex, float2 p) {
	return make_float3(tex2DLayered<float>(tex, p.x, p.y, 0),
		tex2DLayered<float>(tex, p.x, p.y, 1),
		tex2DLayered<float>(tex, p.x, p.y, 2));
}

