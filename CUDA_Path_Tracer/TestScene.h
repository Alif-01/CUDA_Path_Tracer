#pragma once
#include "Scene.h"
#include "ObjLoader.h"

class TestScene :
	public Scene {
public:
	TestScene(int w, int h)
		:Scene(w, h, CameraHost(vec3(0, 0.5, -5), vec3(0, 0.5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		set_env_texture(load_texture("res/cayley_interior_4k.hdr", true));

		camera_host.exposure = 2.3;

		//auto normal_map = load_texture("res/Wood26_nrm.jpg");
		//auto albedo_map = load_texture("res/Wood26_col.jpg", true);
		//auto rough_map = load_texture("res/Wood26_rgh.jpg");

		//renderer.USE_ENV = false;

	}

	virtual const char* getWindowTitle() {
		return "Test Scene";
	}
};

class CornellBoxScene :
	public Scene {
public:
	CornellBoxScene(int w, int h) 
		: Scene(w, h, CameraHost(vec3(0, 5, 15), vec3(0, 5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		auto gold = pow(vec3(1, 0.77, 0.21), 2.2);
		auto mat = new UE4CookTorrance(1, 0, gold);

		ObjLoader bunny("res/bunny_smooth.obj", 20);
		for (auto obj : bunny.triangles)
			add_object(obj.first, mat);

		add_object(new Box(vec3(-5, 0, -5), vec3(5, 1, 100)), new Lambert(vec3(1)));
		add_object(new Box(vec3(-5, 9, -5), vec3(5, 10, 100)), new Lambert(vec3(1)));
		add_object(new Box(vec3(-5, 0, -5), vec3(5, 10, -4)), new Lambert(vec3(1)));
		add_object(new Box(vec3(-5, 0, -5), vec3(-4, 10, 100)), new Lambert(vec3(1, 0, 0)));
		add_object(new Box(vec3(4, 0, -5), vec3(5, 10, 100)), new Lambert(vec3(0, 1, 0)));
		add_object(new Box(vec3(-0.5, 8.9, -4.9), vec3(0.5, 10.1, 100)), new Emitter(vec3(4)));
		//add_object(new Sphere(vec3(0, 3, 0), 2), new Glass(vec3(1, 0.71, 0.29), 1.02));
	}

	virtual const char* getWindowTitle() {
		return "Cornell Box";
	}
};

class CheshuoScene : public Scene {
public:
	CheshuoScene(int w, int h) 
		: Scene(w, h, CameraHost(vec3(0, 5, 15), vec3(0, 5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		set_env_texture(load_texture("res/veranda_4k.hdr", true));

		auto albedo_map = load_texture("res/Tiles40_col.jpg", true);
		//auto metal_map = load_texture("res/PavingStones28.jpg");
		auto normal_map = load_texture("res/Tiles40_nrm.jpg");
		auto rough_map = load_texture("res/Tiles40_rgh.jpg");
		auto ao_map = 0;//load_texture("res/PavingStones28_AO.jpg");

		//Material *mat = new UE4CookTorrance(1, 1, vec3(1), metal_map, rough_map, albedo_map, ao_map);

		//add_textured_box(vec3(0), vec3(10), mat, 0.2, normal_map);

		auto cheshuo_mtl = load_mtllib("res/salle_de_bain/salle_de_bain.mtl", "res/salle_de_bain/");

		ObjLoader model("res/salle_de_bain/salle_de_bain.obj", 1, vec3(0, 0, 0));
		for (auto obj : model.triangles) {
			add_object(obj.first, cheshuo_mtl[obj.second]);
		}

		//for (int i = 0;i < 8;i++)
		//	for (int j = 0;j < 8;j++)
		//		add_object(new Box(vec3(i, 0, j), vec3(i + 1, 1, j + 1)), new UE4CookTorrance(0.1, 0, ((i + j) % 2) ? vec3(0.95) : vec3(0.05)));

		//add_object(new Sphere(vec3(20, 31, 10), 10), new Emitter(vec3(20)));

		//add_object(new Box(vec3(-100), vec3(100, 0, 100)), new UE4CookTorrance(0.3, 0.2, vec3(0.95, 0.95, 0.7)));

		//renderer.USE_ENV = false;

		vector<float3> points3 = load_points("res/wineglass.txt", 1 / 1.5);

		vector<float2> points;

		for (float3 p : points3) points.push_back(make_float2(p.x, p.y));

		int k = 3;

		vector<float> controls(points.size() + k + 1);
		for (int i = 0;i < controls.size();i++)
			controls[i] = i*1.0 / (controls.size() - 1);

		auto mat = new Glass(vec3(1), 1.55);

		for (int i = k;i + k + 1 < controls.size();i++) {
			auto rev_i = new Revolved(controls, points, i, k, vec3(-8.62, 8.0415 + 2.459 / 1.5, 12.729));
			auto tri_ls = rev_i->convert_triangles(30, 100);
			for (auto t : tri_ls)
				add_object(t, mat);
		}
	}

	virtual const char* getWindowTitle() { return "Cheshuo"; }
};

class ShenbiScene : public Scene {
public:
	ShenbiScene(int w, int h)
		: Scene(w, h, CameraHost(vec3(0, 5, 15), vec3(0, 5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		set_env_texture(load_texture("res/cayley_interior_4k.hdr", true));

		//add_object(new Sphere(vec3(10, 20, 10), 10), new Emitter(vec3(10, 10, 20)));


		for(int i=-20;i<=10;i++)
			for (int j = -20;j <= 10;j++) {
				float3 color = vec3(rand() % 10001 / 10000.0, rand() % 10001 / 10000.0, rand() % 10001 / 10000.0);
				auto mat = new UE4CookTorrance(rand() % 10001 / 10000.0, 0, pow(color, 2.2));
				//auto mat = new UE4CookTorrance(0.5, 0, pow(vec3(0.99), 2.2), 0, rough_map, albedo_map, 0);
				ObjLoader model("res/stick.obj", 1, vec3(i * 2.1, rand() % 10001 / 4000.0 + (i + j)*0.5, j * 2.1), vec3(0, 0.1, 0));
				for (auto tri : model.triangles) {
					add_object(tri.first, mat);
				}
			}
	}

	virtual const char* getWindowTitle() { return "Shenbi Scene"; }
};

class TestScene2 : public Scene {
public:
	TestScene2(int w, int h)
		: Scene(w, h, CameraHost(vec3(0, 5, 15), vec3(0, 5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		set_env_texture(load_texture("res/veranda_4k.hdr", true));

		auto normal_map = load_texture("res/Wood26_nrm.jpg");
		auto albedo_map = load_texture("res/Wood26_col.jpg", true);
		auto rough_map = load_texture("res/Wood26_rgh.jpg");
		auto AO_map = 0;

		add_textured_box(vec3(-100, -10, -100), vec3(100, 0, 100), new UE4CookTorrance(0, 1, vec3(1), 0, rough_map, albedo_map, AO_map), 1, normal_map);

		normal_map = load_texture("res/Fabric30_nrm.jpg");
		albedo_map = load_texture("res/Fabric30_col.jpg", true);
		rough_map = load_texture("res/Fabric30_rgh.jpg");

		add_textured_box(vec3(0.01), vec3(1.01), new UE4CookTorrance(0, 1, vec3(1), 0, rough_map, albedo_map, AO_map), 0.2, normal_map);
	}

	virtual const char* getWindowTitle() { return "Test2"; }
};

class TestScene3 : public Scene {
public:
	TestScene3(int w, int h)
		: Scene(w, h, CameraHost(vec3(0, 5, 15), vec3(0, 5, 0), vec3(0, 1, 0), 60, 1.0*w / h)) {

		camera_host.exposure = 2.3;

		set_env_texture(load_texture("res/snowy_park_01_4k.hdr", true));

		auto cheshuo_mtl = load_mtllib("res/koishi2.mtl", "res/koishi2/");

		ObjLoader model("res/koishi2.obj");
		for (auto i : model.triangles)
			add_object(i.first, cheshuo_mtl[i.second]);

		auto mat_heart = new UE4CookTorrance(0.3, 0.3, vec3(1, 0.1, 0.1));

		for (int i = 0;i < 10;i++) {
			float r = pow(i, 0.85) * 7 + 15;
			for (int j = 0;j < 10;j++) {
				float theta = PI * 2 / 10 * (j + 0.5 - i*0.13);
				ObjLoader heart("res/heart.obj", -2, vec3(r*cos(theta), r*sin(theta) + 10, -5), vec3(), PI / 2 - theta + i*0.1);
				for (auto tri : heart.triangles)
					add_object(tri.first, mat_heart);
			}
			if (i > 0) {
				for (int j = 0;j < 10;j++) {
					float theta = PI * 2 / 10 * (j + i*0.13 + 0.5);
					ObjLoader heart("res/heart.obj", -2, vec3(r*cos(theta), r*sin(theta) + 10, -5), vec3(), PI / 2 - theta - i*0.1);
					for (auto tri : heart.triangles)
						add_object(tri.first, mat_heart);
				}
			}
		}

		for (int i = 0;i < 10;i++) {
			float r = pow(i, 0.85) * 7 + 15;
			for (int j = 0;j < 10;j++) {
				float theta = PI * 2 / 10 * (j - i*0.13);
				ObjLoader heart("res/heart.obj", -2, vec3(r*cos(theta), r*sin(theta) + 10, 5), vec3(), PI / 2 - theta + i*0.1);
				for (auto tri : heart.triangles)
					add_object(tri.first, mat_heart);
			}
			if (i > 0) {
				for (int j = 0;j < 10;j++) {
					float theta = PI * 2 / 10 * (j + i*0.13);
					ObjLoader heart("res/heart.obj", -2, vec3(r*cos(theta), r*sin(theta) + 10, 5), vec3(), PI / 2 - theta - i*0.1);
					for (auto tri : heart.triangles)
						add_object(tri.first, mat_heart);
				}
			}
		}
	}

	virtual const char* getWindowTitle() { return "Test2"; }
};
