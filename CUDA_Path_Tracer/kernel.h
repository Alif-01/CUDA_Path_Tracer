#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "Renderer.h"

__host__ void kernel_clear(Renderer &renderer);

__host__ void kernel_init_curand_state(Renderer &renderer);

__host__ void kernel_render(Renderer &renderer);
