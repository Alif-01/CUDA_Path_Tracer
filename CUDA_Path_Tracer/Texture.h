#pragma once
#include "util.h"

class Texture {
public:
	Texture(const char *file_name, int filter_mode, bool load_hdr);
	~Texture();

	int width, height, channels;
	cudaArray_t arr;
	cudaTextureObject_t texture_obj;
};

