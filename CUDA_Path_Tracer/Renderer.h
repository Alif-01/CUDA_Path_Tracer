#pragma once
#include "util.h"
#include "Camera.h"
#include "Object.h"
#include "Material.h"
#include "BVH.h"

#include <driver_functions.h>

#include <vector>

class Renderer {
public:
	static const int SAMPLE_PER_FRAME = 1;
	static const int MAX_DEPTH = 10;
	bool USE_ENV;

	int screen_width, screen_height;
	int sample_count;

	curandState *curand_state;
	float3 *acc_buffer;
	uchar3 *out_buffer;

	Box *box_list;
	Sphere *sphere_list;
	Revolved *revolved_list;
	Triangle *triangle_list;

	BVHNode *bvh_node_list;
	int num_bvh_node;
	int *obj_ref_list;

	UE4CookTorrance *mat_ue4_list;
	Lambert *mat_lambert_list;
	Emitter *mat_emitter_list;
	Glass *mat_glass_list;

	int *emitter_id_list;

	int box_size, sphere_size, revolved_size, triangle_size;
	int ue4_size, lambert_size, emitter_size, glass_size;

	cudaTextureObject_t env_texture;

	Camera camera;

	int debug_flag;

	__host__ Renderer(int screen_width, int screen_height, Camera camera) :
		camera(camera) {
		this->screen_width = screen_width;
		this->screen_height = screen_height;

		sample_count = 0;
		env_texture = 0;

		USE_ENV = true;
	}

	__host__ void create_bvh(Object **obj_list, int num_obj) {
		BVHHost bvh_host(obj_list, num_obj);
		BVHNode *temp_node_list = new BVHNode[bvh_host.num_nodes];
		bvh_host.traverse(bvh_host.root, temp_node_list);
		bvh_node_list = malloc_device<BVHNode>(bvh_host.num_nodes);
		obj_ref_list = malloc_device<int>(bvh_host.obj_id_list.size());
		CUDA_CHECK(cudaMemcpy(bvh_node_list, temp_node_list, sizeof(BVHNode)*bvh_host.num_nodes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(obj_ref_list, &bvh_host.obj_id_list[0], sizeof(int)*bvh_host.obj_id_list.size(), cudaMemcpyHostToDevice));
		delete[] temp_node_list;
		puts("building BVH done");
	}

	__host__ void create_buffer(const std::vector<ObjectDef> & object_defs) {
		curand_state = malloc_device<curandState>(screen_width * screen_height);
		acc_buffer = malloc_device<float3>(screen_width * screen_height);

		box_size = sphere_size = revolved_size = triangle_size = 0;
		lambert_size = ue4_size = emitter_size = glass_size = 0;

		for (ObjectDef def : object_defs) {
			if (!def.obj->get_type()) {
				puts("object undefined");
				continue;
			}

			if (!def.mat->get_type()) {
				puts("material undefined");
				continue;
			}

			switch (def.obj->get_type()) {
			case OBJ_BOX:
				def.obj->object_id = (box_size++) << OBJ_BITS | OBJ_BOX;
				break;

			case OBJ_SPHERE:
				def.obj->object_id = (sphere_size++) << OBJ_BITS | OBJ_SPHERE;
				break;

			case OBJ_REVOLVED:
				def.obj->object_id = (revolved_size++) << OBJ_BITS | OBJ_REVOLVED;
				break;

			case OBJ_TRIANGLE:
				def.obj->object_id = (triangle_size++) << OBJ_BITS | OBJ_TRIANGLE;
				break;
			}

			switch (def.mat->get_type()) {
			case MAT_LAMBERT:
				def.obj->material_id = (lambert_size++) << MAT_BITS | MAT_LAMBERT;
				break;

			case MAT_UE4COOKTORRANCE:
				def.obj->material_id = (ue4_size++) << MAT_BITS | MAT_UE4COOKTORRANCE;
				break;

			case MAT_EMITTER:
				def.obj->material_id = (emitter_size++) << MAT_BITS | MAT_EMITTER;
				break;

			case MAT_GLASS:
				def.obj->material_id = (glass_size++) << MAT_BITS | MAT_GLASS;
				break;
			}
		}

		box_list = malloc_host<Box>(box_size);
		sphere_list = malloc_host<Sphere>(sphere_size);
		revolved_list = malloc_host<Revolved>(revolved_size);
		triangle_list = malloc_host<Triangle>(triangle_size);

		mat_lambert_list = malloc_host<Lambert>(lambert_size);
		mat_ue4_list = malloc_host<UE4CookTorrance>(ue4_size);
		mat_emitter_list = malloc_host<Emitter>(emitter_size);
		mat_glass_list = malloc_host<Glass>(glass_size);

		emitter_id_list = malloc_host<int>(emitter_size);

		Object **obj_list = new Object*[box_size + sphere_size + revolved_size + triangle_size];
		int cur_obj = 0;

		box_size = sphere_size = revolved_size = triangle_size = 0;
		lambert_size = ue4_size = emitter_size = glass_size = 0;

		int cnt = 0;

		for (ObjectDef def : object_defs) {
			if (!def.obj->get_type() || !def.mat->get_type())
				continue;

			obj_list[cur_obj++] = def.obj;

			switch (def.obj->get_type()) {
			case OBJ_BOX:
				if (def.mat->get_type() == MAT_EMITTER)
					assign_host<int>(emitter_id_list + emitter_size, box_size << OBJ_BITS | OBJ_BOX);
				assign_host<Box>(box_list + (box_size++), *((Box*)def.obj));
				break;

			case OBJ_SPHERE:
				if (def.mat->get_type() == MAT_EMITTER)
					assign_host<int>(emitter_id_list + emitter_size, sphere_size << OBJ_BITS | OBJ_SPHERE);
				assign_host<Sphere>(sphere_list + (sphere_size++), *((Sphere*)def.obj));
				break;

			case OBJ_REVOLVED:
				if (def.mat->get_type() == MAT_EMITTER)
					assign_host<int>(emitter_id_list + emitter_size, revolved_size << OBJ_BITS | OBJ_REVOLVED);
				assign_host<Revolved>(revolved_list + (revolved_size++), *((Revolved*)def.obj));
				break;

			case OBJ_TRIANGLE:
				if (def.mat->get_type() == MAT_EMITTER)
					assign_host<int>(emitter_id_list + emitter_size, triangle_size << OBJ_BITS | OBJ_TRIANGLE);
				assign_host<Triangle>(triangle_list + (triangle_size++), *((Triangle*)def.obj));
				break;
			}

			switch (def.mat->get_type()) {
			case MAT_LAMBERT:
				assign_host<Lambert>(mat_lambert_list + (lambert_size++), *((Lambert*)def.mat));
				break;

			case MAT_UE4COOKTORRANCE:
				assign_host<UE4CookTorrance>(mat_ue4_list + (ue4_size++), *((UE4CookTorrance*)def.mat));
				break;

			case MAT_EMITTER:
				assign_host<Emitter>(mat_emitter_list + (emitter_size++), *((Emitter*)def.mat));
				break;

			case MAT_GLASS:
				assign_host<Glass>(mat_glass_list + (glass_size++), *((Glass*)def.mat));
				break;
			}
		}

		box_list = copy_to_device<Box>(box_list, box_size);
		sphere_list = copy_to_device<Sphere>(sphere_list, sphere_size);
		revolved_list = copy_to_device<Revolved>(revolved_list, revolved_size);
		triangle_list = copy_to_device<Triangle>(triangle_list, triangle_size);

		mat_lambert_list = copy_to_device<Lambert>(mat_lambert_list, lambert_size);
		mat_ue4_list = copy_to_device<UE4CookTorrance>(mat_ue4_list, ue4_size);
		mat_emitter_list = copy_to_device<Emitter>(mat_emitter_list, emitter_size);
		mat_glass_list = copy_to_device<Glass>(mat_glass_list, glass_size);

		emitter_id_list = copy_to_device<int>(emitter_id_list, emitter_size);

		puts("creating object done");

		create_bvh(obj_list, cur_obj);
		delete[] obj_list;
	}

	__host__ void free() {
		free_device(curand_state);
		free_device(acc_buffer);

		free_device(box_list);
		free_device(sphere_list);
		free_device(revolved_list);
		free_device(triangle_list);

		free_device(mat_lambert_list);
		free_device(mat_ue4_list);
		free_device(mat_emitter_list);
		free_device(mat_glass_list);

		free_device(emitter_id_list);

		free_device(bvh_node_list);
		free_device(obj_ref_list);
	}

	__device__ bool ray_hit_test(Ray ray, float max_t = 1e50) {
		Hit hit(0, 0, make_float3(0), max_t);
		int stack[MAX_BVH_DEPTH];
		int stack_size = 1;
		stack[0] = 0;
		while (stack_size) {
			int node_id = stack[--stack_size];
			BVHNode node = bvh_node_list[node_id];
			if (node.index.x < 0) {
				BVHNode cl = bvh_node_list[-node.index.x], cr = bvh_node_list[-node.index.y];
				float tl = intersect_ray_aabb(ray, cl.p_min, cl.p_max);
				float tr = intersect_ray_aabb(ray, cr.p_min, cr.p_max);
				if (tl > -1e-3 && tl < hit.t && tr > -1e-3 && tr < hit.t) {
					if (tl < tr) {
						stack[stack_size++] = -node.index.y;
						stack[stack_size++] = -node.index.x;
					} else {
						stack[stack_size++] = -node.index.x;
						stack[stack_size++] = -node.index.y;
					}
				} else if (tl > -1e-3 && tl < hit.t) {
					stack[stack_size++] = -node.index.x;
				} else if (tr > -1e-3 && tr < hit.t) {
					stack[stack_size++] = -node.index.y;
				}
			} else {
				int ref_l = node.index.x, ref_r = node.index.y;
				for (int i = ref_l;i < ref_r;i++) {
					int obj_ref = obj_ref_list[i];
					int obj_type = obj_ref & ((1 << OBJ_BITS) - 1), obj_id = obj_ref >> OBJ_BITS;
					if (obj_type == OBJ_BOX) {
						box_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_SPHERE) {
						sphere_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_REVOLVED) {
						revolved_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_TRIANGLE) {
						triangle_list[obj_id].intersect(ray, hit);
					}
					if (hit.object_id) return true;
				}
			}
		}
		return false;
	}

	__device__ Hit ray_intersect(Ray ray, int last_obj_id, float max_t = 1e50) {
		Hit hit(0, 0, make_float3(0), max_t);
		int stack[MAX_BVH_DEPTH];
		int stack_size = 1;
		stack[0] = 0;
		while (stack_size) {
			int node_id = stack[--stack_size];
			BVHNode node = bvh_node_list[node_id];
			if (node.index.x < 0) {
				BVHNode cl = bvh_node_list[-node.index.x], cr = bvh_node_list[-node.index.y];
				float tl = intersect_ray_aabb(ray, cl.p_min, cl.p_max);
				float tr = intersect_ray_aabb(ray, cr.p_min, cr.p_max);
				if (tl > -1e-4 && tl < hit.t && tr > -1e-4 && tr < hit.t) {
					if (tl < tr) {
						stack[stack_size++] = -node.index.y;
						stack[stack_size++] = -node.index.x;
					} else {
						stack[stack_size++] = -node.index.x;
						stack[stack_size++] = -node.index.y;
					}
				} else if (tl > -1e-4 && tl < hit.t) {
					stack[stack_size++] = -node.index.x;
				} else if (tr > -1e-4 && tr < hit.t) {
					stack[stack_size++] = -node.index.y;
				}
			} else {
				int ref_l = node.index.x, ref_r = node.index.y;
				for (int i = ref_l;i < ref_r;i++) {
					int obj_ref = obj_ref_list[i];
					int obj_type = obj_ref & ((1 << OBJ_BITS) - 1), obj_id = obj_ref >> OBJ_BITS;
					if (obj_type == OBJ_BOX) {
						box_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_SPHERE) {
						sphere_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_REVOLVED) {
						if (obj_ref != last_obj_id)
							revolved_list[obj_id].intersect(ray, hit);
					} else if (obj_type == OBJ_TRIANGLE) {
						if (obj_ref != last_obj_id)
							triangle_list[obj_id].intersect(ray, hit);
					}
				}
			}
		}
		return hit;
	}

	__device__ void path_trace_debug(int depth, Hit hit, Ray ray, int screen_x, int screen_y) {
		if (!debug_pos(screen_x, screen_y)) return;
		float3 pos = ray.at(hit.t);
		if (hit.material_id)
			printf("hit %d: pos %.3f,%.3f,%.3f, normal %.3f,%.3f,%.3f, ray %.3f,%.3f,%.3f %f\n", depth, pos.x, pos.y, pos.z, hit.normal.x, hit.normal.y, hit.normal.z, ray.d.x, ray.d.y, ray.d.z, dot(ray.d, hit.normal));
		else
			printf("nohit %d\n", depth);
	}

	__device__ float3 enviroment(float3 d) {
		if (env_texture) {
			double x = atan2(d.z, d.x) / (double)(2 * PI) + camera.env_rotate;
			double y = acos(d.y) / (double)PI;
			float3 env_color = make_float3(
				tex2DLayered<float>(env_texture, x, y, 0),
				tex2DLayered<float>(env_texture, x, y, 1),
				tex2DLayered<float>(env_texture, x, y, 2)
			);
			return env_color;
		}
		return (vec3(0.75, 0.85, 1) - vec3(0.25, 0.15) * d.y);
	}

	__device__ __inline__ float MIS_weight(float x, float y) {
		float x2 = x*x;
		return x2 / (x2 + y*y);
	}

	__device__ void sample_emitter(curandState *state, float3 &color, float3 &position, float3 &normal, float &pdf) {
		int index = curand_uniform(state)*emitter_size;
		color = mat_emitter_list[index].emitter_color*emitter_size;
		int obj_id = emitter_id_list[index] >> OBJ_BITS, obj_type = emitter_id_list[index] & ((1 << OBJ_BITS) - 1);

		if (obj_type == OBJ_BOX) {
			Box obj = box_list[obj_id];
			obj.sample_surface(state, position, normal);
			pdf = 1.0 / obj.surface_area();
		} else if (obj_type == OBJ_SPHERE) {
			Sphere obj = sphere_list[obj_id];
			obj.sample_surface(state, position, normal);
			pdf = 1.0 / obj.surface_area();
		//} else if (obj_type == OBJ_REVOLVED) {
		//	Revolved obj = revolved_list[obj_id];
		//	obj.sample_surface(state, position, normal);
		//	pdf = 1.0 / obj.surface_area();
		} else if (obj_type == OBJ_TRIANGLE) {
			Triangle obj = triangle_list[obj_id];
			obj.sample_surface(state, position, normal);
			pdf = 1.0 / obj.surface_area();
		}
	}

	__device__ float3 direct_light(curandState *state, Ray &ray, const Hit &hit, Material *material, MaterialType mat_type) {
		if (!emitter_size) return vec3();

		float3 hit_position = ray.at(hit.t);
		float3 emitter_color, emitter_pos, emitter_normal;
		float emitter_pdf, brdf_pdf;

		sample_emitter(state, emitter_color, emitter_pos, emitter_normal, emitter_pdf);

		emitter_pos -= hit_position;
		float3 emitter_dir = normalize(emitter_pos);
		float emitter_dis2 = dot(emitter_pos, emitter_pos);
		float emitter_dis = sqrtf(emitter_dis2);

		emitter_pdf *= emitter_dis2 / dot(emitter_normal, -emitter_dir);

		if (!ray_hit_test(Ray(hit_position + emitter_dir*1e-3, emitter_dir), emitter_dis - 2e-3)) {
			float3 attenuation;
			if (mat_type == MAT_LAMBERT) {
				brdf_pdf = ((Lambert*)material)->pdf(hit, emitter_dir, -ray.d);
				attenuation = ((Lambert*)material)->brdf(hit, emitter_dir, -ray.d);
			} else if (mat_type == MAT_UE4COOKTORRANCE) {
				brdf_pdf = ((UE4CookTorrance*)material)->pdf(hit, emitter_dir, -ray.d);
				attenuation = ((UE4CookTorrance*)material)->brdf(hit, emitter_dir, -ray.d);
			}

			attenuation *= dot(hit.normal, emitter_dir);

			return MIS_weight(emitter_pdf, brdf_pdf)*emitter_color*attenuation / emitter_pdf;
		}
		return vec3();
	}

	__device__ __inline__ float3 monte_carlo(Ray ray, curandState *curand_state) {
		float3 radiance = vec3(), throughput = vec3(1.0f);
		float last_pdf = -1;
		int last_obj_id = -1;
		for (int depth = 0;depth < MAX_DEPTH;depth++) {
			Hit hit = ray_intersect(ray, last_obj_id);
			if (!hit.material_id) {
				if (USE_ENV) {
					// TODO: enviroment MIS
					radiance += throughput * enviroment(ray.d);
				}
				break;
			}
			float3 hit_position = ray.at(hit.t);
			last_obj_id = hit.object_id;
			int mat_type = hit.material_id&((1 << MAT_BITS) - 1), mat_id = hit.material_id >> MAT_BITS;
			int obj_type = hit.object_id&((1 << OBJ_BITS) - 1), obj_id = hit.object_id >> OBJ_BITS;
			if (obj_type == OBJ_TRIANGLE)
				triangle_list[obj_id].collect(ray, hit);
			if (debug_flag)
				path_trace_debug(depth, hit, ray, 630, 360);
			if (mat_type == MAT_LAMBERT) {
				if (dot(ray.d, hit.normal) > 0) hit.normal = -hit.normal;

				Lambert material = mat_lambert_list[mat_id];
				float3 refl_dir = material.sample(curand_state, hit, -ray.d);
				if (dot(refl_dir, hit.normal) < 0) break;
				float3 brdf = material.brdf(hit, refl_dir, -ray.d);

				radiance += throughput*direct_light(curand_state, ray, hit, &material, MAT_LAMBERT);

				last_pdf = material.pdf(hit, refl_dir, -ray.d);
				throughput *= brdf * dot(refl_dir, hit.normal) / last_pdf;
				ray.p = hit_position;
				ray.d = refl_dir;
			} else if (mat_type == MAT_UE4COOKTORRANCE) {
				if (dot(ray.d, hit.normal) > 0) hit.normal = -hit.normal;

				UE4CookTorrance material = mat_ue4_list[mat_id];
				float3 refl_dir = material.sample(curand_state, hit, -ray.d);
				if (dot(refl_dir, hit.normal) < 0) break;
				float3 brdf = material.brdf(hit, refl_dir, -ray.d);

				radiance += throughput*direct_light(curand_state, ray, hit, &material, MAT_UE4COOKTORRANCE);

				last_pdf = material.pdf(hit, refl_dir, -ray.d);
				throughput *= brdf * dot(refl_dir, hit.normal) / last_pdf;
				ray.p = hit_position;
				ray.d = refl_dir;
			} else if (mat_type == MAT_EMITTER) {
				Emitter material = mat_emitter_list[mat_id];
				if (last_pdf < 0)
					radiance += throughput * material.emitter_color;
				else {
					float emitter_pdf;
					if (obj_type == OBJ_SPHERE)
						emitter_pdf = 1.0 / sphere_list[obj_id].surface_area();
					else if (obj_type == OBJ_BOX)
						emitter_pdf = 1.0 / box_list[obj_id].surface_area();
					else if (obj_type == OBJ_REVOLVED)
						emitter_pdf = 1.0 / revolved_list[obj_id].surface_area();
					else if (obj_type == OBJ_TRIANGLE)
						emitter_pdf = 1.0 / triangle_list[obj_id].surface_area();
					emitter_pdf *= hit.t*hit.t / dot(-ray.d, hit.normal);
					radiance += MIS_weight(last_pdf, emitter_pdf) * throughput * material.emitter_color;
				}
				break;
			} else if (mat_type == MAT_GLASS) {
				Glass material = mat_glass_list[mat_id];
				float3 dir = material.sample(curand_state, hit, -ray.d);
				last_pdf = -1;
				if(dot(hit.normal, dir)<0) throughput *= material.albedo;
				ray.p = hit_position;
				ray.d = dir;
			}
			if (isnan(throughput.x) || isnan(throughput.y) || isnan(throughput.z))
				break;
		}
		return radiance;
	}

	__device__ void update(int x, int y) {
		if (x >= screen_width || y >= screen_height) return;
		int offset = y*screen_width + x;
		for (int sample_t = 0;sample_t < SAMPLE_PER_FRAME;sample_t++) {
			float px = (float(x) + curand_uniform(curand_state + offset)) / float(screen_width);
			float py = (float(y) + curand_uniform(curand_state + offset)) / float(screen_height);
			Ray ray = camera.get_ray(px, py, curand_state + offset);
			float3 color = monte_carlo(ray, curand_state + offset);
			float3 mapped_color = vec3(1.0f) - exp(-color*camera.exposure);
			if (mapped_color.x>-0.1 && mapped_color.x<1.1 && mapped_color.y>-0.1 && mapped_color.y<1.1 && mapped_color.z>-0.1 && mapped_color.z<1.1)
				acc_buffer[offset] += mapped_color;
		}
		//if (debug_pos(630, 360)) {
		//	out_buffer[offset] = make_uchar3(255, 0, 0);
		//	return;
		//}
		float3 acc_color = acc_buffer[offset] / sample_count;
		//float3 mapped_color = fminf(acc_color, make_float3(1.0f));
		//float3 mapped_color = acc_color / (acc_color + make_float3(1.0f));
		float power = 1.0f / camera.gamma;
		out_buffer[offset].x = pow(acc_color.x, power) * 255;
		out_buffer[offset].y = pow(acc_color.y, power) * 255;
		out_buffer[offset].z = pow(acc_color.z, power) * 255;
	}
};
