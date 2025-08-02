use std::{
    collections::HashMap,
    os::raw::c_char, sync::{Arc, LazyLock, Mutex},
};

use tch::Device;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("ek-computation/ops/stream.h");

        pub unsafe fn stream_alloc(high_priority: bool, device: i8) -> *mut c_char;

        pub unsafe fn stream_guard_create(stream: *mut c_char) -> *mut c_char;

        pub unsafe fn stream_guard_destroy(guard_ptr: *mut c_char);
    }
}

pub struct TorchStream(*mut c_char);

impl TorchStream {
    pub fn new(device: Device) -> Self {
        static STREAMS: LazyLock<Arc<Mutex<HashMap<usize, (usize, Vec<usize>)>>>> = LazyLock::new(|| {
            Arc::new(Mutex::new(HashMap::new()))
        });

        let device = match device {
            Device::Cuda(idx) => idx,
            _ => panic!("Unsupported device type"),
        };

        // Check if the stream already exists for this device
        let mut streams = STREAMS.lock().unwrap();
        let (index, stream_list) = streams.entry(device).or_insert_with(|| {
            let mut stream_list = Vec::new();
            for _ in 0..32 {
                stream_list.push(unsafe { ffi::stream_alloc(false, device as i8) } as _);
            }
            (0, stream_list)
        });
        let stream_ptr = stream_list[*index] as *mut c_char;
        *index = (*index + 1) % stream_list.len();
        TorchStream(stream_ptr)
    }

    pub fn guard(&self) -> TorchStreamGuard {
        let guard = unsafe { ffi::stream_guard_create(self.0) };
        TorchStreamGuard(guard)
    }
}

pub struct TorchStreamGuard(*mut c_char);

impl Drop for TorchStreamGuard {
    fn drop(&mut self) {
        unsafe { ffi::stream_guard_destroy(self.0) };
    }
}
