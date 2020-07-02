#pragma once
#include "util.h"
#include "Material.h"
#include "AABB.h"
#include "curve.h"
#define MIN_T 5e-3

const int OBJ_BITS = 4;

enum ObjectType {
	OBJ_UNDEFINED,
	OBJ_BOX,
	OBJ_SPHERE,
	OBJ_REVOLVED,
	OBJ_TRIANGLE,
};

class Object {
public:
	int material_id, object_id;

	__host__ virtual int get_type() { return OBJ_UNDEFINED; }

	__host__ virtual AABB get_AABB(AABB limit = AABB::max_AABB()) { return AABB(); }

	__device__ void intersect(Ray ray, Hit &hit) {}

	__device__ float3 sample_surface(curandState *state) { return vec3(); }

	__device__ float surface_area() { return 0; }
};

class Box : public Object {
public:
	float3 p1, p2;

	__host__ virtual int get_type() { return OBJ_BOX; }

	__host__ virtual AABB get_AABB(AABB limit = AABB::max_AABB()) {
		return AABB(p1, p2) * limit;
	}

	__device__ __host__ Box(float3 p1, float3 p2) :p1(p1), p2(p2) {
		if (p1.x > p2.x) swap(p1.x, p2.x);
		if (p1.y > p2.y) swap(p1.y, p2.y);
		if (p1.z > p2.z) swap(p1.z, p2.z);
	}

	__device__ void intersect(Ray ray, Hit &hit) {
		float3 hit1 = (p1 - ray.p) / ray.d, hit2 = (p2 - ray.p) / ray.d;
		float3 mint = fminf(hit1, hit2), maxt = fmaxf(hit1, hit2);
		if (max3(mint) < MIN_T) {
			if (maxt.x < maxt.y && maxt.x < maxt.z) {
				if (maxt.x < MIN_T || maxt.x > hit.t) return;
				hit.material_id = material_id;
				hit.object_id = object_id;
				hit.t = maxt.x;
				hit.normal = ray.d.x > 0 ? make_float3(1, 0, 0) : make_float3(-1, 0, 0);
			} else if (maxt.y < maxt.z) {
				if (maxt.y < MIN_T || maxt.y > hit.t) return;
				hit.material_id = material_id;
				hit.object_id = object_id;
				hit.t = maxt.y;
				hit.normal = ray.d.y > 0 ? make_float3(0, 1, 0) : make_float3(0, -1, 0);
			} else {
				if (maxt.z < MIN_T || maxt.z > hit.t) return;
				hit.material_id = material_id;
				hit.object_id = object_id;
				hit.t = maxt.z;
				hit.normal = ray.d.z > 0 ? make_float3(0, 0, 1) : make_float3(0, 0, -1);
			}
			return;
		}
		if (mint.x > mint.y && mint.x > mint.z) {
			if (mint.x > maxt.y || mint.x > maxt.z || mint.x > hit.t) return;
			hit.material_id = material_id;
			hit.object_id = object_id;
			hit.t = mint.x;
			hit.normal = ray.d.x < 0 ? make_float3(1, 0, 0) : make_float3(-1, 0, 0);
		} else if (mint.y > mint.z) {
			if (mint.y<MIN_T || mint.y > maxt.x || mint.y > maxt.z || mint.y > hit.t) return;
			hit.material_id = material_id;
			hit.object_id = object_id;
			hit.t = mint.y;
			hit.normal = ray.d.y < 0 ? make_float3(0, 1, 0) : make_float3(0, -1, 0);
		} else {
			if (mint.z<MIN_T || mint.z > maxt.x || mint.z > maxt.y || mint.z > hit.t) return;
			hit.material_id = material_id;
			hit.object_id = object_id;
			hit.t = mint.z;
			hit.normal = ray.d.z < 0 ? make_float3(0, 0, 1) : make_float3(0, 0, -1);
		}
	}

	__device__ void sample_surface(curandState *state, float3 &position, float3 &normal) {
		float a1 = fabs((p1.x - p2.x)*(p1.y - p2.y));
		float a2 = fabs((p1.x - p2.x)*(p1.z - p2.z));
		float a3 = fabs((p1.y - p2.y)*(p1.z - p2.z));
		float k = curand_uniform(state)*(a1 + a2 + a3) * 2 - a1 - a2 - a3;
		float r1 = curand_uniform(state), r2 = curand_uniform(state);
		if (fabs(k) < a1) {
			if (k > 0) {
				normal = vec3(0, 0, -1);
				position = vec3(lerp(p1.x, p2.x, r1), lerp(p1.y, p2.y, r2), p1.z);
			} else {
				normal = vec3(0, 0, 1);
				position = vec3(lerp(p1.x, p2.x, r1), lerp(p1.y, p2.y, r2), p2.z);
			}
		} else if (fabs(k) < a1 + a2) {
			if (k > 0) {
				normal = vec3(0, -1, 0);
				position = vec3(lerp(p1.x, p2.x, r1), p1.y, lerp(p1.z, p2.z, r2));
			} else {
				normal = vec3(0, 1, 0);
				position = vec3(lerp(p1.x, p2.x, r1), p2.y, lerp(p1.z, p2.z, r2));
			}
		} else {
			if (k > 0) {
				normal = vec3(-1, 0, 0);
				position = vec3(p1.x, lerp(p1.y, p2.y, r1), lerp(p1.z, p2.z, r2));
			} else {
				normal = vec3(1, 0, 0);
				position = vec3(p2.x, lerp(p1.y, p2.y, r1), lerp(p1.z, p2.z, r2));
			}
		}
	}

	__device__ float surface_area() {
		return (fabs((p1.x - p2.x)*(p1.y - p2.y)) +
			fabs((p1.x - p2.x)*(p1.z - p2.z)) +
			fabs((p1.y - p2.y)*(p1.z - p2.z))) * 2;
	}
};

class Sphere : public Object {
public:
	float3 o;	//center
	float r;	//radius

	__host__ virtual int get_type() { return OBJ_SPHERE; }

	__host__ virtual AABB get_AABB(AABB limit = AABB::max_AABB()) {
		float3 p_min = o - vec3(r), p_max = o + vec3(r);
		return AABB(p_min, p_max) * limit;
	}

	__device__ __host__ Sphere(float3 o, float r) :o(o), r(r) {}

	__device__ void intersect(Ray ray, Hit &hit) {
		float3 p_k = ray.d, p_o = ray.p - o;
		float a = dot(p_k, p_k), b = 2 * dot(p_k, p_o), c = dot(p_o, p_o) - r*r;
		float root = b*b - 4 * a*c;
		if (root < 0) return;
		float i2a = 1.0f / (2 * a);
		root = sqrtf(root);
		float lambda = (-b - root) * i2a;
		if (lambda > hit.t) return;
		if (lambda > MIN_T) {
			hit.t = lambda;
			hit.normal = (ray.at(lambda) - o) / r;
			hit.material_id = material_id;
			hit.object_id = object_id;
			return;
		}
		else {
			lambda = (-b + root) * i2a;
			if (lambda > MIN_T && lambda < hit.t) {
				hit.t = lambda;
				hit.normal = (ray.at(lambda) - o) / r;
				hit.material_id = material_id;
				hit.object_id = object_id;
				return;
			}
			else return;
		}
	}

	__device__ void sample_surface(curandState *state, float3 &position, float3 &normal) {
		normal = uniform_sphere(state);
		position = o + normal*r;
	}

	__device__ float surface_area() {
		return 4 * PI*r*r;
	}
};

class Triangle : public Object {
public:
	float3 p1, p2, p3, n1, n2, n3, t1, t2, t3;
	float2 uv1, uv2, uv3;
	float disp_height;

	cudaTextureObject_t normal_map;
	cudaTextureObject_t disp_map;

	__host__ virtual int get_type() { return OBJ_TRIANGLE; }

	__host__ virtual AABB get_AABB(AABB limit = AABB::max_AABB()) {
		AABB res;
		float3 t1 = p1, t2 = p2;
		if (limit.clip_segment(t1, t2))
			res += AABB(t1, t1) + AABB(t2, t2);
		t1 = p1, t2 = p3;
		if (limit.clip_segment(t1, t2))
			res += AABB(t1, t1) + AABB(t2, t2);
		t1 = p2, t2 = p3;
		if (limit.clip_segment(t1, t2))
			res += AABB(t1, t1) + AABB(t2, t2);
		return res;
	}

	__device__ __host__ Triangle(float3 p1, float3 p2, float3 p3)
		:Triangle(p1, p2, p3, normalize(cross(p2 - p1, p3 - p1)), 
			normalize(cross(p2 - p1, p3 - p1)), normalize(cross(p2 - p1, p3 - p1))) {
	}

	__device__ __host__ Triangle(float3 p1, float3 p2, float3 p3, float3 n1, float3 n2, float3 n3,
		float2 uv1 = make_float2(0), float2 uv2 = make_float2(0), float2 uv3 = make_float2(0),
		cudaTextureObject_t normal_map = 0, cudaTextureObject_t disp_map = 0, float disp_height = 1)
		: p1(p1), p2(p2), p3(p3), uv1(uv1), uv2(uv2), uv3(uv3) {
		this->n1 = normalize(n1);
		this->n2 = normalize(n2);
		this->n3 = normalize(n3);

		float2 d1 = uv2 - uv1, d2 = uv3 - uv1;
		float3 e1 = p2 - p1, e2 = p3 - p1;
		float det = d1.x*d2.y - d1.y*d2.x;
		float k1 = d2.y / det, k2 = -d1.y / det;
		float tx = e1.x*k1 + e2.x*k2;
		float ty = e1.y*k1 + e2.y*k2;
		float tz = e1.z*k1 + e2.z*k2;

		t1 = t2 = t3 = vec3(tx, ty, tz);

		this->normal_map = normal_map;
		this->disp_map = disp_map;
		this->disp_height = disp_height;
	}

	__device__ void intersect(Ray ray, Hit &hit) {
		float3 d = ray.d;
		float3 e1 = p2 - p1, e2 = p3 - p1;
		float3 h = cross(d, e2);
		float a = dot(e1, h);
		if (fabs(a) < 1e-5) return;
		float ia = 1.0f / a;

		float3 d1 = ray.p - p1;
		float3 c = cross(d1, e1);
		float u = dot(d1, h)*ia;
		float v = dot(d, c)*ia;
		float t = dot(e2, c) * ia;
		if (u < 0 || v < 0 || u + v>1 || t<0 || t>hit.t) return;
		hit.material_id = material_id;
		hit.object_id = object_id;
		hit.t = t;
		hit.uv = make_float2(u, v);
	}

	__device__ void collect(Ray ray, Hit &hit) {
		float3 d = ray.d;
		float u = hit.uv.x, v = hit.uv.y;

		float3 tn1, tn2, tn3;
		tn1 = (dot(d, n1) > 0) ? cross(cross(d, n1), d) : n1;
		tn2 = (dot(d, n2) > 0) ? cross(cross(d, n2), d) : n2;
		tn3 = (dot(d, n3) > 0) ? cross(cross(d, n3), d) : n3;
		hit.uv = u*uv2 + v*uv3 + (1 - u - v)*uv1;
		float3 normal = normalize(u*tn2 + v*tn3 + (1 - u - v)*tn1);
		float3 tangent = u*t2 + v*t3 + (1 - u - v)*t1;
		float3 bitangent = normalize(cross(normal, tangent));
		tangent = cross(bitangent, normal);
		if (normal_map) {
			float3 t = sample_tex_float3(normal_map, hit.uv) * 2 - 1;
			hit.normal = normalize(tangent*t.x + bitangent*t.y + normal*t.z);
		} else {
			hit.normal = normal;
		}
		//hit.normal = normalize(u*n2 + v*n3 + (1 - u - v)*n1);
	}

	__device__ void sample_surface(curandState *state, float3 &position, float3 &normal) {
		float u = sqrtf(curand_uniform(state)), v = curand_uniform(state);
		position = (1 - u)*p1 + (u*(1 - v))*p2 + (u*v)*p3;
		normal = normalize((1 - u)*n1 + (u*(1 - v))*n2 + (u*v)*n3);
	}

	__device__ float surface_area() {
		return length(cross(p2 - p1, p3 - p1));
	}
};

class Revolved : public Object {
public:
	float3 p;
	float r[4], y[4];
	float t1, t2;

	__host__ virtual int get_type() { return OBJ_REVOLVED; }

	__host__ virtual AABB get_AABB(AABB limit = AABB::max_AABB()) {
		float2 r_lim = Curve::get_bound(r, 3, 0, 1);
		float2 y_lim = Curve::get_bound(y, 3, 0, 1);
		float r_bound = fmaxf(fabsf(r_lim.x), fabsf(r_lim.y));
		float3 p_min = p + vec3(-r_bound, y_lim.x, -r_bound);
		float3 p_max = p + vec3(r_bound, y_lim.y, r_bound);
		return AABB(p_min, p_max) * limit;
	}

	__device__ __host__ Revolved(float t1, float t2, float3 p=vec3()){
		this->p = p;
		this->t1 = t1;
		this->t2 = t2;
		for (int i = 0;i <= 3;i++) r[i] = y[i] = 0;
	}

	__host__ Revolved(vector<float> &t, vector<float2> &p, int i, int k, float3 p0=vec3()){
		this->p = p0;
		this->t1 = t[i];
		this->t2 = t[i+1];
		calc_poly(t, p, i, k);
	}

	__host__ void calc_poly(vector<float> &t, vector<float2> &p, int i, int k) {
		vector<double2> poly = Curve::createBSplineSegment(t, p, i, k);
		while (poly.size() < 4) poly.push_back(make_double2(0, 0));
		double b = t[i];
		double r0[4], y0[4];
		r0[0] = poly[3].x*b*b*b + poly[2].x*b*b + poly[1].x*b + poly[0].x;
		r0[1] = poly[3].x*b*b * 3 + poly[2].x*b * 2 + poly[1].x;
		r0[2] = poly[3].x*b * 3 + poly[2].x;
		r0[3] = poly[3].x;
		y0[0] = poly[3].y*b*b*b + poly[2].y*b*b + poly[1].y*b + poly[0].y;
		y0[1] = poly[3].y*b*b * 3 + poly[2].y*b * 2 + poly[1].y;
		y0[2] = poly[3].y*b * 3 + poly[2].y;
		y0[3] = poly[3].y;
		double tk = 1, ti = t[i + 1] - t[i];
		for (int i = 0;i <= 3;i++) r[i] = y[i] = 0;
		for (int i = 0;i <= k;i++)
			r[i] = r0[i] * tk, y[i] = y0[i] * tk, tk *= ti;
	}

	__host__ __device__ float3 calc_normal(float u, float v) {
		float t[4], rt[4];
		for (int i = 0;i < 3;i++) {
			t[i] = r[i + 1] * (i + 1);
			rt[i] = y[i + 1] * (i + 1);
		}
		float dr = Curve::calc_poly(t, 2, u);
		float dy = Curve::calc_poly(rt, 2, u);
		float dd = sqrtf(dr*dr + dy*dy);
		float tdr = dr;
		dr = dy / dd;
		dy = - tdr / dd;
		float3 p1 = vec3(cosf(v*PI * 2), 0, sin(v*PI * 2));
		return make_float3(p1.x*dr, dy, p1.z*dr);
	}

	__device__ void intersect(Ray ray, Hit &hit) {
		double tt[7];
		float t[7], rt[6];
		double3 s = make_double3((ray.p - p).x, (ray.p - p).y, (ray.p - p).z), d = make_double3(ray.d.x, ray.d.y, ray.d.z);
		double bx = s.x*d.y - d.x*s.y, bz = s.z*d.y - d.z*s.y;
		double k2 = d.x*d.x + d.z*d.z, k1 = (bx*d.x + bz*d.z) * 2, k0 = bx*bx + bz*bz, kr = -d.y*d.y;
		for (int i = 0;i <= 6;i++) tt[i] = 0;
		tt[0] = k0;
		for (int i = 0;i <= 3;i++) {
			tt[i] += k1*y[i];
			for (int j = 0;j <= 3;j++)
				tt[i + j] += k2*y[i] * y[j] + kr*r[i] * r[j];
		}
		int k = 6;
		while (fabs(tt[k]) < 1e-8 && k) k--;
		double tk = tt[k];
		//for (int i = 0;i <= k;i++) tk = fmaxf(tk, fabsf(tt[i]));
		for (int i = 0;i <= k;i++) t[i] = tt[i] / tk;
		//if (debug_pos(630, 360)) {
		//	Curve::print_poly(r, 3);
		//	Curve::print_poly(y, 3);
		//	Curve::print_poly(t, k);
		//}
		Curve::get_root(t, rt, k);
		//if (debug_pos(630, 360)) {
		//	for (int i = 0;i < k;i++) printf("%f ", rt[i]);
		//	printf("\n\n");
		//}
		float res_t = -1;
		for (int i = k-1;i>=0;i--)
			if (rt[i] > 0 && rt[i] < 1) {
				float yv = Curve::calc_poly(y, 3, rt[i]) + p.y;
				float rv = Curve::calc_poly(r, 3, rt[i]);
				float a = d.x*d.x + d.z*d.z;
				float b = ((s.x - p.x)*d.x + (s.z - p.z)*d.z) * 2;
				float c = (s.x - p.x)*(s.x - p.x) + (s.z - p.z)*(s.z - p.z) - rv*rv;
				float temp = b*b - a*c * 4;
				if (temp > -1e-5) {
					temp = sqrtf(fmaxf(temp,0));
					float l1 = (-b - temp) / (a * 2);
					float l2 = l1 + temp / a;
					if (fabsf(s.y + d.y*l1 - yv) < 1e-3*fmaxf(1.0f, fabsf(yv)) && l1 > MIN_T && l1 < hit.t) {
						res_t = rt[i];
						hit.t = l1;
						hit.normal = vec3(0, 0, -1);
						hit.material_id = material_id;
						hit.object_id = object_id;
					} else if (fabsf(s.y + d.y*l2 - yv) < 1e-3*fmaxf(1.0f, fabsf(yv)) && l2 > MIN_T && l2 < hit.t) {
						res_t = rt[i];
						hit.t = l2;
						hit.normal = vec3(0, 0, -1);
						hit.material_id = material_id;
						hit.object_id = object_id;
					}
				}
			}
		//if (res_t < -0.5 && debug_pos(630, 360)) {
		//	for (int i = 0;i < k;i++)
		//		printf("%f ", rt[i]);
		//	printf("bad \n");
		//}
		if (res_t > -0.5) {
			for (int i = 0;i < 3;i++) {
				t[i] = r[i + 1] * (i + 1);
				rt[i] = y[i + 1] * (i + 1);
			}
			float dr = Curve::calc_poly(t, 2, res_t);
			float dy = Curve::calc_poly(rt, 2, res_t);
			float dd = sqrtf(dr*dr + dy*dy);
			float tdr = dr;
			dr = dy / dd;
			dy = - tdr / dd;
			float3 p1 = ray.p - p + ray.d*hit.t - p;
			p1.y = 0;
			p1 = normalize(p1);
			hit.normal = make_float3(p1.x*dr, dy, p1.z*dr);
			hit.uv = make_float2(0, res_t*(t2 - t1) + t1);
			//if (debug_pos(630, 360))
			//	printf("ok %f %f %f %f\n", res_t, (s + d*hit.t).x, (s + d*hit.t).y, (s + d*hit.t).z);
		} else {
			//if (debug_pos(630, 360)) {
			//	for (int i = 0;i < k;i++)
			//		printf("%f ", rt[i]);
			//	printf("\n");
			//}
		}
	}

	__host__ vector<Triangle*> convert_triangles(int n1, int n2,
		cudaTextureObject_t normal_map = 0, cudaTextureObject_t disp_map = 0, float disp_height = 1) {
		vector<Triangle*> res;
		for (int i = 0;i < n1;i++) {
			float t1 = (float)i / n1, t2 = (float)(i + 1) / n1;
			float r1 = Curve::calc_poly(r, 3, t1), r2 = Curve::calc_poly(r, 3, t2);
			float y1 = Curve::calc_poly(y, 3, t1), y2 = Curve::calc_poly(y, 3, t2);
			for (int j = 0;j < n2;j++) {
				float u1 = (float)j / n2, u2 = (float)(j + 1) / n2;
				float3 p1 = vec3(cosf(u1*PI * 2)*r1 + p.x, y1 + p.y, sinf(u1*PI * 2)*r1 + p.z);
				float3 p2 = vec3(cosf(u2*PI * 2)*r1 + p.x, y1 + p.y, sinf(u2*PI * 2)*r1 + p.z);
				float3 p3 = vec3(cosf(u1*PI * 2)*r2 + p.x, y2 + p.y, sinf(u1*PI * 2)*r2 + p.z);
				float3 p4 = vec3(cosf(u2*PI * 2)*r2 + p.x, y2 + p.y, sinf(u2*PI * 2)*r2 + p.z);
				float3 n1 = calc_normal(t1, u1);
				float3 n2 = calc_normal(t1, u2);
				float3 n3 = calc_normal(t2, u1);
				float3 n4 = calc_normal(t2, u2);
				float2 uv1 = vec2(t1*(this->t2 - this->t1) + this->t1, u1);
				float2 uv2 = vec2(t1*(this->t2 - this->t1) + this->t1, u2);
				float2 uv3 = vec2(t2*(this->t2 - this->t1) + this->t1, u1);
				float2 uv4 = vec2(t2*(this->t2 - this->t1) + this->t1, u2);
				res.push_back(new Triangle(p1, p2, p3, n1, n2, n3, uv1, uv2, uv3, normal_map, disp_map, disp_height));
				res.push_back(new Triangle(p3, p2, p4, n3, n2, n4, uv3, uv2, uv4, normal_map, disp_map, disp_height));
			}
		}
		return res;
	}
};

inline Box * get_bounding_box(Object *obj) {
	AABB aabb = obj->get_AABB();
	return new Box(aabb.p_min, aabb.p_max);
}

struct ObjectDef {
	Object *obj;
	Material *mat;
};

