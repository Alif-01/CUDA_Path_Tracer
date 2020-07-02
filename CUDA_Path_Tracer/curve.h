#pragma once
#include "util.h"
#include <vector>

using namespace std;

namespace Curve {
	__host__ __device__ __inline__ void print_poly(float *a, int k) {
		for (int i = k;i;i--)
			printf("%.6f * x^%d + ", a[i], i);
		printf("%.6f\n", a[0]);
	}

	__host__ __device__ __inline__ float calc_poly(float *a, int k, float x) {
		float s = 0;
		for (int i = k;i >= 0;i--) s = s*x + a[i];
		return s;
	}

	__host__ __device__ __inline__ void poly_diff(float *a, int k) {
		for (int i = 0;i < k;i++) a[i] = a[i + 1] * (i + 1);
		a[k] = 0;
	}

	__host__ __device__ __inline__ float get_range_root(float *a, int k, float l, float r) {
		float fl = calc_poly(a, k, l), fr = calc_poly(a, k, r);
		while (r - l > 1e-6) {
			float mid = (l + r) / 2;
			float fmid = 0;
			for (int i = k;i >= 0;i--) fmid = fmid*mid + a[i];
			if (fl*fmid > 0) l = mid, fl = fmid;
			else r = mid, fr = fmid;
		}
		float x = l, y = fl, dy = 0;
		for (int i = k;i;i--) dy = dy*x + a[i] * i;
		for(int t=0;t<5;t++){
			x -= y / dy;
			for (int i = k;i;i--) y = y*x + a[i], dy = dy*x + a[i] * i;
			y = y*x + a[0];
		}
		return x;
	}

	__host__ __device__ __inline__ void _get_root(float *a, float *rt, int k) {
		int nrt = 0;
		for (int i = 0;i < k;i++) rt[i] = 0;
		static float ta[7], rt0[7];
		for (int l = 1;l <= k;l++) {
			for (int i = 0;i <= k;i++) ta[i] = a[i];
			for (int i = k;i > k;i--) poly_diff(ta, i);
			int nrt0 = 0;
			for (int i = 0;i < nrt;i++) rt0[nrt0++] = rt[i];
			rt0[nrt0++] = 1;
			float lastx = 0, lasty = a[0];
			for (int i = 0;i < nrt0;i++) {
				float x = rt0[i];
				float y = 0;
				for (int j = k;j >= 0;j--) y = y*x + a[j];
				if (fabsf(y) < 1e-6) rt[nrt++] = x;
				else if (fabs(lasty) > 1e-6 && lasty*y < 0) rt[nrt++] = get_range_root(a, k, lastx, x);
				lastx = x;
				lasty = y;
				if (nrt == l) break;
			}
			if (debug_pos(630, 360)) {
				printf("\n%d ", l);
				for (int i = 0;i <= l;i++)
					printf("%f ", a[i]);
				//float ty = 0;
				//for (int i = l;i >= 0;i--)
				//	ty = ty*0.875 + a[i];
				//printf("%f", ty);
			}
		}
	}

	__host__ __device__ __inline__ void get_root(float *a, float *rt, int k) {
		int test_cnt = 0;
		float init_value = 0;
		for (int t = k - 1;t >= 0;t--) {
			float x = init_value;
			float temp = 1, y = 0, y1 = 0;
			for (int i = k;i >= 0;i--) y = y*x + a[i];
			for (int i = k;i;i--) y1 = y1*x + a[i] * i;
			int cnt = 0;
			while (fabsf(y) > 1e-6 && (++cnt) <= 20) {
				x -= y / y1;
				y = y1 = 0;
				for (int i = k;i;i--) {
					y = y*x + a[i];
					y1 = y1*x + a[i] * i;
				}
				y = y*x + a[0];
			}
			if (fabsf(y) > 1e-3) {
				if (++test_cnt > 10) break;
				init_value = (float)test_cnt / 10;
				t++;
				continue;
			}
			test_cnt = 0;
			init_value = 0;
			rt[t] = x;
			for (int i = k - 1;i >= 0;i--) a[i] += a[i + 1] * x;
			for (int i = 0;i < k;i++) a[i] = a[i + 1];
			a[k] = 0;
			k--;
		}
		while (k) rt[--k] = 0;
	}

	__host__ __inline__ float2 get_bound(float *a, int k, float x1, float x2) {
		float *b = new float[k], *c = new float[k - 1];
		for (int i = 1;i <= k;i++)
			b[i - 1] = a[i] * i;
		get_root(b, c, k - 1);
		float s1 = fminf(calc_poly(a, k, x1), calc_poly(a, k, x2));
		float s2 = fmaxf(calc_poly(a, k, x1), calc_poly(a, k, x2));
		for (int i = 0;i < k - 1;i++)
			if (c[i] > x1 && c[i] < x2) {
				float t = calc_poly(a, k, c[i]);
				if (t < s1) s1 = t;
				if (t > s2) s2 = t;
			}
		delete[] b;
		delete[] c;
		return make_float2(s1, s2);
	}

	__host__ __inline__ vector<double> calcBasis(vector<float> &t, int seg, int i, int k) {
		if (k == 0) return i == seg ? vector<double>(1, 1) : vector<double>(1, 0);
		vector<double> t1 = calcBasis(t, seg, i, k - 1), t2 = calcBasis(t, seg, i + 1, k - 1);
		vector<double> s(k + 1, 0);
		double k0 = -(double)t[i] / (t[i + k] - t[i]), k1 = 1.0 / (t[i + k] - t[i]);
		double k2 = (double)t[i + k + 1] / (t[i + k + 1] - t[i + 1]), k3 = -1.0 / (t[i + k + 1] - t[i + 1]);
		for (int i = 0;i < k;i++) {
			s[i + 1] += t1[i] * k1 + t2[i] * k3;
			s[i] += t1[i] * k0 + t2[i] * k2;
		}
		return s;
	}

	__host__ __inline__ vector<double2> createBSplineSegment(vector<float> &t, vector<float2> &p, int i, int k) {
		if (i - k < 0 || i + k + 1 >= t.size()) {
			printf("B-spline error: i-k<0 or i+k+1>n");
			return vector<double2>();
		}
		vector<double2> res(k + 1, make_double2(0, 0));
		for (int j = i - k;j <= i;j++) {
			vector<double> b = calcBasis(t, i, j, k);
			for (int t = 0;t <= k;t++) {
				res[t].x += b[t] * p[j].x;
				res[t].y += b[t] * p[j].y;
			}
		}
		return res;
	}
}
