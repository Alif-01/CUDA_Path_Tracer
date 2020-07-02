#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Texture::Texture(const char * file_name, int filter_mode, bool load_hdr) {
	float* data;
	if (load_hdr) {
		float *raw_data = stbi_loadf(file_name, &width, &height, &channels, 0);
		data = new float[width*height*channels];
		for (int i = 0;i < width*height;i++)
			for (int j = 0;j < channels;j++)
				data[j*width*height + i] = raw_data[i*channels + j];
		stbi_image_free(raw_data);
	} else {
		uchar *raw_data = stbi_load(file_name, &width, &height, &channels, 0);
		data = new float[width*height*channels];
		for (int i = 0;i < width*height;i++)
			for (int j = 0;j < channels;j++)
				data[j*width*height + i] = (float)raw_data[i*channels + j] / 255;
		stbi_image_free(raw_data);
	}

	auto extent = make_cudaExtent(width, height, channels);

	auto formatDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_CHECK(cudaMalloc3DArray(&arr, &formatDesc, extent, cudaArrayLayered));

	cudaMemcpy3DParms params;
	memset(&params, 0, sizeof(params));
	params.srcPos = params.dstPos = make_cudaPos(0, 0, 0);
	params.srcPtr = make_cudaPitchedPtr(data, width * sizeof(float), width, height);
	params.dstArray = arr;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&params));

	delete[] data;

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = arr;
	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeWrap;
	tex_desc.addressMode[2] = cudaAddressModeWrap;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = true;
	CUDA_CHECK(cudaCreateTextureObject(&texture_obj, &res_desc, &tex_desc, NULL));
}

Texture::~Texture() {
	CUDA_CHECK(cudaDestroyTextureObject(texture_obj));
	CUDA_CHECK(cudaFreeArray(arr));
}
