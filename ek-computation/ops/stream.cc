#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

char * stream_alloc(bool high_priority, c10::DeviceIndex device_index) {
    auto stream = new c10::cuda::CUDAStream(at::cuda::getStreamFromPool(high_priority, device_index));
    return reinterpret_cast<char *>(stream);
}

char * stream_guard_create(char * stream) {
    auto guard = new c10::cuda::CUDAStreamGuard(*reinterpret_cast<c10::cuda::CUDAStream *>(stream));
    return reinterpret_cast<char *>(guard);
}

void stream_guard_destroy(char * guard_ptr) {
    delete reinterpret_cast<c10::cuda::CUDAStreamGuard *>(guard_ptr);
}
