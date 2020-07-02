#pragma once

#include "util.h"

const float INF = 1e50, EPS = 1e-5;

class AABB {
public:
	float3 p_min, p_max;

	AABB(float3 p_min = vec3(INF), float3 p_max = vec3(-INF)) : p_min(p_min), p_max(p_max) {}

	static AABB max_AABB() { return AABB(vec3(-INF), vec3(INF)); }

	AABB operator + (const AABB& aabb) {
		return AABB(fminf(p_min, aabb.p_min), fmaxf(p_max, aabb.p_max));
	}

	AABB& operator += (const AABB& aabb) {
		*this = (*this) + aabb;
		return *this;
	}

	AABB operator * (const AABB& aabb) {
		return AABB(fmaxf(p_min, aabb.p_min), fminf(p_max, aabb.p_max));
	}

	AABB& operator *= (const AABB& aabb) {
		*this = (*this) * aabb;
		return *this;
	}

	bool empty() {
		float3 d = p_max - p_min;
		return d.x < -EPS || d.y < -EPS || d.z < -EPS;
	}

	float area() {
		float3 d = p_max - p_min;
		if (d.x < 0 || d.y < 0 || d.z < 0) return 0;
		return 2 * (d.x*d.y + d.x*d.z + d.y*d.z);
	}

	float volumn() {
		float3 d = p_max - p_min;
		if (d.x < 0 || d.y < 0 || d.z < 0) return 0;
		return d.x*d.y*d.z;
	}

	bool clip_segment(float3 &p1, float3 &p2) {
		float3 s_min = fminf(p1, p2), s_max = fmaxf(p1, p2);
		if ((AABB(s_min, s_max) * (*this)).empty()) return false;
		if (p_min.x > s_min.x + EPS && p_min.x < s_max.x - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_min.x - p1.x) / (p2.x - p1.x);
			if (p1.x < p2.x) p1 = pt; else p2 = pt;
		}
		if (p_max.x > s_min.x + EPS && p_max.x < s_max.x - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_max.x - p1.x) / (p2.x - p1.x);
			if (p1.x < p2.x) p2 = pt; else p1 = pt;
		}
		if (p_min.y > s_min.y + EPS && p_min.y < s_max.y - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_min.y - p1.y) / (p2.y - p1.y);
			if (p1.y < p2.y) p1 = pt; else p2 = pt;
		}
		if (p_max.y > s_min.y + EPS && p_max.y < s_max.y - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_max.y - p1.y) / (p2.y - p1.y);
			if (p1.y < p2.y) p2 = pt; else p1 = pt;
		}
		if (p_min.z > s_min.z + EPS && p_min.z < s_max.z - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_min.z - p1.z) / (p2.z - p1.z);
			if (p1.z < p2.z) p1 = pt; else p2 = pt;
		}
		if (p_max.z > s_min.z + EPS && p_max.z < s_max.z - EPS) {
			float3 pt = p1 + (p2 - p1)*(p_max.z - p1.z) / (p2.z - p1.z);
			if (p1.z < p2.z) p2 = pt; else p1 = pt;
		}
		return true;
	}

	void debug_print() {
		printf("AABB ");
		print_float3(p_min);
		print_float3(p_max);
		puts("");
	}
};

__device__ __host__ __inline__ float intersect_ray_aabb(Ray &ray, float3 &p_min, float3 &p_max) {
	float3 inv_d = make_float3(1.0f) / ray.d;
	float3 hit1 = (p_min - ray.p)*inv_d, hit2 = (p_max - ray.p)*inv_d;
	float3 mint = fminf(hit1, hit2), maxt = fmaxf(hit1, hit2);
	float t1 = (mint.x > mint.y && mint.x > mint.z) ? mint.x : (mint.y > mint.z ? mint.y : mint.z);
	float t2 = (maxt.x < maxt.y && maxt.x < maxt.z) ? maxt.x : (maxt.y < maxt.z ? maxt.y : maxt.z);
	return t1 < t2 + 1e-3 ? (t1 < 0 ? fminf(t2, 0) : t1) : -1;
}
