#pragma once
#include "util.h"
#include "Object.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using std::vector;
using std::pair;
using std::string;

class ObjLoader {
private:
	vector<float3> vertex_list;
	vector<float2> uv_list;
	vector<float3> normal_list;
public:
	vector<pair<Triangle*, string> > triangles;

	ObjLoader(const char* filename, float scale=1.0, float3 bias=vec3(), float3 bias_u=vec3(), float rotate_z=0){
		std::ifstream f(filename);
		string line;
		string cur_mat = "";
		string cur_obj = "";
		int vc = 0;
		while (std::getline(f, line)) {
			std::istringstream is(line);
			string op;
			if (is.peek() == '#') continue;
			is >> op;
			if (op == "o") {
				is >> cur_obj;
			}
			if (op == "matlib") {
			}
			if (op == "usemtl") {
				is >> cur_mat;
			}
			if (op == "v") {
				float x, y, z;
				is >> x >> y >> z;
				x *= scale;
				y *= scale;
				z *= scale;
				//vertex_list.push_back(vec3(x, y, z) + bias);
				vertex_list.push_back(vec3(x*cos(rotate_z) + y*sin(rotate_z), y*cos(rotate_z) - x*sin(rotate_z), z) + bias);
			}
			if (op == "vt") {
				float x, y;
				is >> x >> y;
				uv_list.push_back(vec2(x, y));
			}
			if (op == "vn") {
				float x, y, z;
				is >> x >> y >> z;
				//normal_list.push_back(vec3(x, y, z));
				normal_list.push_back(vec3(x*cos(rotate_z) + y*sin(rotate_z), y*cos(rotate_z) - x*sin(rotate_z), z));
			}
			if (op == "f") {
				int x, y, z;
				is >> x;
				if (is.peek() == '/') {
					is.get();
					if (is.peek() == '/') {
						is.get();
						int nx, ny, nz;
						is >> nx >> y;
						is.get(); is.get();
						is >> ny >> z;
						is.get(); is.get();
						is >> nz;
						if (x < 0) x += vertex_list.size(); else x--;
						if (y < 0) y += vertex_list.size(); else y--;
						if (z < 0) z += vertex_list.size(); else z--;
						if (nx < 0) nx += normal_list.size(); else nx--;
						if (ny < 0) ny += normal_list.size(); else ny--;
						if (nz < 0) nz += normal_list.size(); else nz--;
						//Triangle *tri = new Triangle(vertex_list[x-1], vertex_list[y-1], vertex_list[z-1]);
						//triangles.push_back(tri);
						Triangle *tri = new Triangle(vertex_list[x], vertex_list[y], vertex_list[z],
							normal_list[nx], normal_list[ny], normal_list[nz]);
						triangles.push_back(std::make_pair(tri, cur_mat));
					} else {
						int ux, uy, uz;
						int nx, ny, nz;
						is >> ux; is.get(); is >> nx >> y;
						is.get(); is >> uy; is.get(); is >> ny >> z;
						is.get(); is >> uz; is.get(); is >> nz;
						if (x < 0) x += vertex_list.size(); else x--;
						if (y < 0) y += vertex_list.size(); else y--;
						if (z < 0) z += vertex_list.size(); else z--;
						if (ux < 0) ux += uv_list.size(); else ux--;
						if (uy < 0) uy += uv_list.size(); else uy--;
						if (uz < 0) uz += uv_list.size(); else uz--;
						if (nx < 0) nx += normal_list.size(); else nx--;
						if (ny < 0) ny += normal_list.size(); else ny--;
						if (nz < 0) nz += normal_list.size(); else nz--;
						float2 uvx = uv_list[ux]; uvx.y = 1 - uvx.y;
						float2 uvy = uv_list[uy]; uvy.y = 1 - uvy.y;
						float2 uvz = uv_list[uz]; uvz.y = 1 - uvz.y;
						uvx.x += dot(bias_u, vertex_list[x]);
						uvy.x += dot(bias_u, vertex_list[y]);
						uvz.x += dot(bias_u, vertex_list[z]);
						float3 temp_norm = normalize(cross(vertex_list[y] - vertex_list[x], vertex_list[z] - vertex_list[x]));
						//SmoothTriangle *tri = new SmoothTriangle(vertex_list[x], vertex_list[y], vertex_list[z],
						//	temp_norm, temp_norm, temp_norm, uvx, uvy, uvz);
						//if (cur_mat == "Wall") uvx *= 3, uvy *= 3, uvz *= 3;
						Triangle *tri = new Triangle(vertex_list[x], vertex_list[y], vertex_list[z],
							normal_list[nx], normal_list[ny], normal_list[nz], uvx, uvy, uvz);
						triangles.push_back(std::make_pair(tri, cur_mat));
					}
				} else {
					is >> y >> z;
					if (x < 0) x += vertex_list.size(); else x--;
					if (y < 0) y += vertex_list.size(); else y--;
					if (z < 0) z += vertex_list.size(); else z--;
					float3 temp_norm = normalize(cross(vertex_list[y] - vertex_list[x], vertex_list[z] - vertex_list[x]));
					Triangle *tri = new Triangle(vertex_list[x], vertex_list[y], vertex_list[z], 
						temp_norm, temp_norm, temp_norm);
					triangles.push_back(std::make_pair(tri, cur_mat));
				}
			}
		}
	}
};
