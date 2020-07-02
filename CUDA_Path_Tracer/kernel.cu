#include "kernel.h"
#include "cutil_math.h"
#include <ctime>

#include "util.h"
#include "Renderer.h"
#include "Object.h"

__global__ void render_thread(Renderer renderer) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	renderer.update(x, y);
}

__global__ void init_state_thread(Renderer renderer) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*renderer.screen_width + x;

	if (x < renderer.screen_width && y < renderer.screen_height)
		curand_init(0, offset, 0, &renderer.curand_state[offset]);
}

const int INIT_STATE_SIZE = 256;

__global__ void copy_state_thread(Renderer renderer) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*renderer.screen_width + x;

	if (x >= renderer.screen_width || y >= renderer.screen_height) return;
	if (x >= INIT_STATE_SIZE || y >= INIT_STATE_SIZE)
		renderer.curand_state[offset] = renderer.curand_state[(y % INIT_STATE_SIZE)*renderer.screen_width + (x % INIT_STATE_SIZE)];
}

__global__ void clear_thread(Renderer renderer) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*renderer.screen_width + x;
	if (x < renderer.screen_width && y < renderer.screen_height)
		renderer.acc_buffer[offset] = make_float3(0);
}

__host__ void kernel_clear(Renderer & renderer) {
	dim3 grid_size((renderer.screen_width + 15) / 16, (renderer.screen_height + 15) / 16);
	dim3 block_size(16, 16);

	renderer.sample_count = 0;
	clear_thread <<< grid_size, block_size >>> (renderer);
}

void kernel_init_curand_state(Renderer &renderer) {
	dim3 grid_size(INIT_STATE_SIZE/16 , INIT_STATE_SIZE/16);
	dim3 block_size(16, 16);

	init_state_thread <<< grid_size, block_size >>> (renderer);

	grid_size = dim3((renderer.screen_width + 15) / 16, (renderer.screen_height + 15) / 16);

	copy_state_thread <<< grid_size, block_size >>> (renderer);
}

void kernel_render(Renderer &renderer) {
	dim3 grid_size((renderer.screen_width + 15) / 16, (renderer.screen_height + 15) / 16);
	dim3 block_size(16, 16);

	//if (renderer.sample_count > 50) return;
	renderer.sample_count+=Renderer::SAMPLE_PER_FRAME;
	render_thread <<< grid_size, block_size >>> (renderer);
}
