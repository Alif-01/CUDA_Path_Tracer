#pragma once
#include "util.h"

const int MAT_BITS = 4;

enum MaterialType {
	MAT_UNDEFINED,
	MAT_LAMBERT,
	MAT_UE4COOKTORRANCE,
	MAT_EMITTER,
	MAT_GLASS,
	MAT_SSS,
};

class Material {
public:
	__host__ virtual int get_type() { return MAT_UNDEFINED; }

	__device__ float3 sample(curandState *state, const Hit &hit, float3 out) {}

	__device__ float pdf(const Hit &hit, float3 in, float3 out) {}

	__device__ __host__ float3 brdf(const Hit &hit, float3 in, float3 out) {}
};

class Lambert : public Material{
public:
	float3 diffuse_color;

	__host__ virtual int get_type() { return MAT_LAMBERT; }

	__device__ __host__ Lambert(float3 diffuse_color) :diffuse_color(diffuse_color) {}

	__device__ float3 sample(curandState *state, const Hit &hit, float3 out) {
		return sample_hemisphere_cosine(state, hit.normal);
	}

	__device__ float pdf(const Hit &hit, float3 in, float3 out) {
		return dot(in, hit.normal) / PI;
	}

	__device__ float3 brdf(const Hit &hit, float3 in, float3 out) {
		return diffuse_color / PI;
	}
}; 

class UE4CookTorrance : public Material{
public:
	float metallic, roughness, specular;
	float3 albedo;
	cudaTextureObject_t texture_albedo;
	cudaTextureObject_t texture_metallic;
	cudaTextureObject_t texture_roughness;
	cudaTextureObject_t texture_ao;

	__host__ virtual int get_type() { return MAT_UE4COOKTORRANCE; }

	__device__ __host__ UE4CookTorrance(float metallic, float roughness, float3 albedo,
		cudaTextureObject_t texture_metallic = 0, cudaTextureObject_t texture_roughness = 0, 
		cudaTextureObject_t texture_albedo = 0, cudaTextureObject_t texture_ao = 0) {
		this->metallic = metallic;
		this->roughness = roughness;
		this->albedo = albedo;
		this->texture_metallic = texture_metallic;
		this->texture_roughness = texture_roughness;
		this->texture_albedo = texture_albedo;
		this->texture_ao = texture_ao;
		this->specular = 0.04;
	}

	__device__ __host__ __inline__ float3 specular_F(float3 F0, float dotNH) {
		float t = 1 - dotNH;
		float t2 = t*t;
		return F0 + (1 - F0)*t2*t2*t;
	}

	__device__ __host__ __inline__ float specular_D(float dotNH) {
		float a = roughness*roughness + 0.001;
		float a2 = a*a;
		float t = (a2 - 1.0f)*dotNH*dotNH + 1.0f;
		return a2 / (PI*t*t);
	}

	__device__ __host__ __inline__ float specular_G(float dotNV) {
		float k = (roughness*roughness + 0.001) / 2;
		return 0.5f / (dotNV*(1 - k) + k);
	}

	__device__ float3 sample(curandState *state, const Hit &hit, float3 out) {
		if (texture_metallic) metallic *= sample_tex_float(texture_metallic, hit.uv);
		if (texture_roughness) roughness *= sample_tex_float(texture_roughness, hit.uv);
		if (texture_albedo) albedo *= sample_tex_float3(texture_albedo, hit.uv);
		if (curand_uniform(state) < specular + metallic*(1-specular)) {
			float epsilon = curand_uniform(state);
			float theta = curand_uniform(state) * 2 * PI;

			float r2 = roughness*roughness + 0.001;
			float y = sqrt((1.0 - epsilon) / (1.0 + (r2*r2 - 1.0) *epsilon));
			float r = sqrtf(fmaxf(1.0 - y*y, 0.0f));

			float3 U = fabs(hit.normal.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0);
			float3 V = normalize(cross(U, hit.normal));
			U = cross(hit.normal, V);

			float3 h = U * (cosf(theta)*r) + V * (sinf(theta)*r) + hit.normal * y;

			return 2.0*dot(out, h)*h - out;
		} else {
			return sample_hemisphere_cosine(state, hit.normal);
		}
	}

	__device__ float pdf(const Hit &hit, float3 in, float3 out) {
		float3 h = normalize(in + out);
		float dotNH = dot(hit.normal, h);
		if (dotNH < 0) return 0;

		return dot(hit.normal, in) *(1 - metallic)*0.96 / PI
			+ specular_D(dotNH) * dotNH * (specular + metallic*(1-specular)) / (4.0 * dot(in, h));
	}

	__device__ float3 brdf(const Hit &hit, float3 in, float3 out) {
		float3 H = normalize(in + out);
		float dotNH = dot(hit.normal, H);
		float dotNV = dot(hit.normal, out);
		float dotNL = dot(hit.normal, in);
		float dotVH = dot(out, H);

		if (dotNL < 0 || dotNV < 0)
			return vec3();

		float3 F0 = vec3(specular)*(1 - metallic) + albedo*metallic;

		float3 diffuse = albedo / PI;
		float3 specular = specular_F(F0, dotNH)*specular_D(dotNH)*specular_G(dotNV)*specular_G(dotNL);

		float3 ao = vec3(1.0f);
		if (texture_ao) ao = sample_tex_float3(texture_ao, hit.uv);

		return (diffuse*(1 - metallic) * 0.96 + specular)*ao;
	}
};

class Emitter : public Material {
public:
	float3 emitter_color;

	__host__ virtual int get_type() { return MAT_EMITTER; }

	__device__ __host__ Emitter(float3 emitter_color) :emitter_color(emitter_color) {}
};

class Glass : public Material{
public:
	float3 albedo;
	float eta; //relative refractive index
	cudaTextureObject_t texture_albedo;

	__host__ virtual int get_type() { return MAT_GLASS; }

	__device__ __host__ Glass(float3 albedo, float eta, cudaTextureObject_t tex = 0) :albedo(albedo), eta(eta) {
		texture_albedo = tex;
	}

	__device__ __host__ __inline__ float fresnel(float R0, float dotNH) {
		float t = 1 - dotNH;
		float t2 = t*t;
		return R0 + (1 - R0)*t2*t2*t;
	}

	__device__ float3 sample(curandState *state, Hit &hit, float3 out) {
		if (texture_albedo) albedo *= vec3(
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 0),
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 1),
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 2)
		);

		float dotNV = dot(hit.normal, out);
		float eta1 = dotNV > 0 ? 1.0 / eta : eta;

		float R0 = (1 - eta1) / (1 + eta1);
		R0 = R0 * R0;

		float3 normal = dotNV > 0 ? hit.normal : -hit.normal;
		float k = 1 - eta1*eta1*(1 - dotNV*dotNV);
		dotNV = fabs(dotNV);

		if (curand_uniform(state) < fresnel(R0, dotNV) || k < 0) {
			return 2 * normal*dot(normal, out) - out;
		} else {
			float3 res = normal*(eta1*dotNV - sqrtf(k)) - out*eta1;
			return res;
		}
	}
}; 

class SSS : public Material{
public:
	float3 albedo, scattering_dis;
	cudaTextureObject_t texture_albedo;

	__host__ virtual int get_type() { return MAT_SSS; }

	__device__ __host__ SSS(float3 albedo, float3 scattering_dis, cudaTextureObject_t texture_albedo = 0) {
		this->albedo = albedo;
		this->scattering_dis = scattering_dis;
		this->texture_albedo = texture_albedo;
	}

	__device__ __host__ __inline__ float3 specular_F(float3 F0, float dotNH) {
		float t = 1 - dotNH;
		float t2 = t*t;
		return F0 + (1 - F0)*t2*t2*t;
	}

	__device__ float3 sample(curandState *state, const Hit &hit, float3 out) {
		if (texture_albedo) albedo *= vec3(
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 0),
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 1),
			tex2DLayered<float>(texture_albedo, hit.uv.x, hit.uv.y, 2)
		);
	}

	__device__ float pdf(const Hit &hit, float3 in, float3 out) {
		float3 h = normalize(in + out);
		float dotNH = dot(hit.normal, h);
		if (dotNH < 0) return 0;

		//return dot(hit.normal, in) *(1 - metallic)*0.96 / PI
		//	+ specular_D(dotNH) * dotNH * (0.04 + metallic*0.96) / (4.0 * dot(in, h));
	}

	__device__ __host__ float3 brdf(const Hit &hit, float3 in, float3 out) {
		float3 H = normalize(in + out);
		float dotNH = dot(hit.normal, H);
		float dotNV = dot(hit.normal, out);
		float dotNL = dot(hit.normal, in);
		float dotVH = dot(out, H);

		if (dotNL < 0 || dotNV < 0)
			return vec3();

		//float3 F0 = vec3(0.04)*(1 - metallic) + albedo*metallic;

		//float3 diffuse = albedo / PI;
		//float3 specular = specular_F(F0, dotNH)*specular_D(dotNH)*specular_G(dotNV)*specular_G(dotNL);

		//return diffuse*(1 - metallic) * 0.96 + specular;
	}
};

__host__ __inline__ Material * convert_mtllib(float3 Kd, float3 Ks, float3 Ke, float Ns, cudaTextureObject_t albedo_map = 0) {
	if (length(Ke) > 0.01) return new Emitter(Ke);
	float roughness = 1 - log(min(Ns, 1000.0f)) / log(1000);
	float metallic = fmaxf(fmaxf(Ks.x, Ks.y), Ks.z);
	return new UE4CookTorrance(metallic, roughness, Kd + Ks, 0, 0, albedo_map, 0);
}
