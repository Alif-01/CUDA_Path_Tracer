#pragma once
#include "util.h"
#include "Object.h"
#include "AABB.h"
#include <vector>
#include <algorithm>

using std::vector;
using std::sort;

static const int MAX_BVH_DEPTH = 40;

struct BVHNode {
	float3 p_min, p_max;
	int2 index;
};

struct ObjectRef {
	Object *obj;
	AABB aabb;
};

static bool cmp_ref_x(ObjectRef x, ObjectRef y) {
	return x.aabb.p_min.x + x.aabb.p_max.x < y.aabb.p_min.x + y.aabb.p_max.x;
}

static bool cmp_ref_y(ObjectRef x, ObjectRef y) {
	return x.aabb.p_min.y + x.aabb.p_max.y < y.aabb.p_min.y + y.aabb.p_max.y;
}

static bool cmp_ref_z(ObjectRef x, ObjectRef y) {
	return x.aabb.p_min.z + x.aabb.p_max.z < y.aabb.p_min.z + y.aabb.p_max.z;
}

class BVHHost {
public:
	struct BVHHostNode {
		AABB aabb;
		BVHHostNode *child_l, *child_r;
		int obj_id_l, obj_id_r;

		BVHHostNode(AABB aabb, BVHHostNode *cl, BVHHostNode *cr, int idl, int idr)
			:aabb(aabb), child_l(cl), child_r(cr), obj_id_l(idl), obj_id_r(idr) {}
	};

	struct ObjectSplit {
		int sort_dim, split_index;
		float SAH;
	};

	const float COST_INNER = 1, COST_LEAF = 5, NUM_BINS = 50;

	int num_objects, num_nodes;
	vector<int> obj_id_list;
	vector<ObjectRef> ref_list;
	vector<AABB> prefix, suffix, bins;
	BVHHostNode *root;
	int traverse_pos;
	
	ObjectSplit find_object_split(int ref_l, int ref_r) {
		ObjectRef *ptr_ref = &ref_list[ref_l];
		int n = ref_r - ref_l;
		ObjectSplit best_split;
		best_split.SAH = INF;
		for (int sort_dim = 0;sort_dim < 3;sort_dim++) {
			if (sort_dim == 0)		sort(ptr_ref, ptr_ref + n, cmp_ref_x);
			else if (sort_dim == 1) sort(ptr_ref, ptr_ref + n, cmp_ref_y);
			else					sort(ptr_ref, ptr_ref + n, cmp_ref_z);
			prefix[0] = ptr_ref[0].aabb;
			for (int i = 1;i < n;i++) prefix[i] = prefix[i - 1] + ptr_ref[i].aabb;
			suffix[n - 1] = ptr_ref[n - 1].aabb;
			for (int i = n - 2;i >= 0;i--) suffix[i] = suffix[i + 1] + ptr_ref[i].aabb;
			for (int i = 0;i < n - 1;i++) {
				float sah = COST_INNER + prefix[i].area() / suffix[0].area()*i*COST_LEAF + suffix[i + 1].area() / suffix[0].area()*(n - i)*COST_LEAF;
				if (sah < best_split.SAH) {
					best_split.SAH = sah;
					best_split.sort_dim = sort_dim;
					best_split.split_index = i;
				}
			}
		}
		return best_split;
	}

	void split_object(int ref_l, int ref_r, ObjectSplit split, int &res_l, int &res_m, int &res_r) {
		ObjectRef *ptr_ref = &ref_list[ref_l];
		int n = ref_r - ref_l;
		if (split.sort_dim == 0)		sort(ptr_ref, ptr_ref + n, cmp_ref_x);
		else if (split.sort_dim == 1)	sort(ptr_ref, ptr_ref + n, cmp_ref_y);
		else							sort(ptr_ref, ptr_ref + n, cmp_ref_z);
		res_l = ref_l;
		res_m = ref_l + split.split_index + 1;
		res_r = ref_r;
	}

	BVHHostNode * make_leaf(int ref_l, int ref_r) {
		AABB s_aabb;
		int start_id = obj_id_list.size();
		for (int i = ref_l;i < ref_r;i++) {
			s_aabb += ref_list.back().aabb;
			obj_id_list.push_back(ref_list.back().obj->object_id);
			ref_list.pop_back();
		}
		int end_id = obj_id_list.size();
		return new BVHHostNode(s_aabb, 0, 0, start_id, end_id);
	}

	BVHHostNode * build(int ref_l, int ref_r, int max_depth) {
		num_nodes++;

		ObjectRef *ptr_ref = &ref_list[ref_l];
		int n = ref_r - ref_l;

		float sah_leaf = n*COST_LEAF;

		ObjectSplit object_split = find_object_split(ref_l, ref_r);
		float sah_object = object_split.SAH;
		if (!max_depth || sah_leaf < sah_object)
			return make_leaf(ref_l, ref_r);

		int split_l, split_m, split_r;
		split_object(ref_l, ref_r, object_split, split_l, split_m, split_r);

		BVHHostNode *cr = build(split_m, split_r, max_depth - 1);
		BVHHostNode *cl = build(split_l, split_m, max_depth - 1);
		return new BVHHostNode(cl->aabb + cr->aabb, cl, cr, 0, 0);
	}

	BVHHost(Object **obj_list, int num_objects) {
		this->num_objects = num_objects;
		ref_list.resize(num_objects);
		for (int i = 0;i < num_objects;i++) {
			ref_list[i].obj = obj_list[i];
			ref_list[i].aabb = obj_list[i]->get_AABB();
		}
		prefix.resize(num_objects);
		suffix.resize(num_objects);
		bins.resize(NUM_BINS);
		num_nodes = 0;
		root = build(0, num_objects, MAX_BVH_DEPTH - 1);
		traverse_pos = 0;
	}

	int traverse(BVHHostNode *node, BVHNode *node_list) {
		int pos = traverse_pos++;
		node_list[pos].p_min = node->aabb.p_min;
		node_list[pos].p_max = node->aabb.p_max;
		if (node->child_l) {
			node_list[pos].index.x = -traverse(node->child_l, node_list);
			node_list[pos].index.y = -traverse(node->child_r, node_list);
		} else {
			node_list[pos].index.x = node->obj_id_l;
			node_list[pos].index.y = node->obj_id_r;
		}
		return pos;
	}
};
