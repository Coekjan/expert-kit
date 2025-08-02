#pragma once
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

char * stream_alloc(bool high_priority, c10::DeviceIndex device_index);

char * stream_guard_create(char * stream);

void stream_guard_destroy(char * guard);
