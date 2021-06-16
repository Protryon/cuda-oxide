use num_enum::TryFromPrimitive;
use std::{ffi::c_void, marker::PhantomData, pin::Pin, ptr::null_mut, rc::Rc};

use crate::*;

/// A stream of asynchronous operations operating in a [`Context`]
pub struct Stream<'a> {
    pub(crate) inner: *mut sys::CUstream_st,
    pub(crate) pending_stores: Vec<Pin<Box<[u8]>>>,
    _p: PhantomData<&'a ()>,
}

/// Wait comparison type for waiting on some condition in [`Stream::wait_32`]/etc
#[derive(Debug, Copy, Clone, TryFromPrimitive)]
#[repr(u32)]
pub enum WaitValueMode {
    /// Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic comparison which ignores wraparound. (Default behavior.)
    Geq = 0x0,
    /// Wait until *addr == value.
    Eq = 0x1,
    /// Wait until (*addr & value) != 0.
    And = 0x2,
    /// Wait until ~(*addr | value) != 0. Support for this operation can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.
    Nor = 0x3,
}

unsafe extern "C" fn host_callback(arg: *mut std::ffi::c_void) {
    let closure: Box<Box<dyn FnOnce() + Send + Sync>> = Box::from_raw(arg as *mut _);
    closure();
}

impl<'a> Stream<'a> {
    /// Creates a new stream for a handle
    pub fn new(_handle: &Rc<Handle<'a>>) -> CudaResult<Self> {
        let mut out = null_mut();
        cuda_error(unsafe {
            sys::cuStreamCreate(
                &mut out as *mut _,
                sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
            )
        })?;
        Ok(Self {
            inner: out,
            pending_stores: vec![],
            _p: PhantomData,
        })
    }

    /// Drives all pending tasks on the stream to completion
    pub fn sync(&mut self) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuStreamSynchronize(self.inner) })?;
        self.pending_stores.clear();
        Ok(())
    }

    /// Returns `Ok(true)` if the stream has finished processing all queued tasks.
    pub fn is_synced(&self) -> CudaResult<bool> {
        match cuda_error(unsafe { sys::cuStreamQuery(self.inner) }) {
            Ok(()) => Ok(true),
            Err(ErrorCode::NotReady) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Wait for a 4-byte value in a specific location to compare to `value` by `mode`.
    pub fn wait_32<'b>(
        &'b mut self,
        addr: &'b DevicePtr<'a>,
        value: u32,
        mode: WaitValueMode,
        flush: bool,
    ) -> CudaResult<()> {
        if addr.len < 4 {
            panic!("overflow in Stream::wait_32");
        }
        let flush = if flush { 1u32 << 30 } else { 0 };
        cuda_error(unsafe {
            sys::cuStreamWaitValue32(self.inner, addr.inner, value, mode as u32 | flush)
        })
    }

    /// Wait for a 8-byte value in a specific location to compare to `value` by `mode`.
    pub fn wait_64<'b>(
        &mut self,
        addr: &'b DevicePtr<'a>,
        value: u64,
        mode: WaitValueMode,
        flush: bool,
    ) -> CudaResult<()> {
        if addr.len < 8 {
            panic!("overflow in Stream::wait_64");
        }
        let flush = if flush { 1u32 << 30 } else { 0 };
        cuda_error(unsafe {
            sys::cuStreamWaitValue64(self.inner, addr.inner, value, mode as u32 | flush)
        })
    }

    /// Writes a 4-byte value to device memory asynchronously
    pub fn write_32<'b>(
        &'b mut self,
        addr: &'b DevicePtr<'a>,
        value: u32,
        no_memory_barrier: bool,
    ) -> CudaResult<()> {
        if addr.len < 4 {
            panic!("overflow in Stream::write_32");
        }
        let no_memory_barrier = if no_memory_barrier { 1u32 } else { 0 };
        cuda_error(unsafe {
            sys::cuStreamWriteValue32(self.inner, addr.inner, value, no_memory_barrier)
        })
    }

    /// Writes a 8-byte value to device memory asynchronously
    pub fn write_64<'b>(
        &'b mut self,
        addr: &'b DevicePtr<'a>,
        value: u64,
        no_memory_barrier: bool,
    ) -> CudaResult<()> {
        if addr.len < 8 {
            panic!("overflow in Stream::write_64");
        }
        let no_memory_barrier = if no_memory_barrier { 1u32 } else { 0 };
        cuda_error(unsafe {
            sys::cuStreamWriteValue64(self.inner, addr.inner, value, no_memory_barrier)
        })
    }

    /// Calls a callback closure function `callback` once all prior tasks in the Stream have been driven to completion.
    /// Note that it is a memory leak to drop the stream before this callback is called.
    /// The callback is not guaranteed to be called if the stream errors out.
    /// Also note that it is erroneous in `libcuda` to make any calls to `libcuda` from this callback.
    /// The callback is called from a CUDA internal thread, however this is an implementation detail of `libcuda` and not guaranteed.
    pub fn callback<F: FnOnce() + Send + Sync>(&mut self, callback: F) -> CudaResult<()> {
        let callback: Box<Box<dyn FnOnce()>> = Box::new(Box::new(callback));
        cuda_error(unsafe {
            sys::cuLaunchHostFunc(
                self.inner,
                Some(host_callback),
                Box::leak(callback) as *mut _ as *mut _,
            )
        })
    }

    /// Launch a CUDA kernel on this [`Stream`] with the given `grid_dim` grid dimensions, `block_dim` block dimensions, `shared_mem_size` allocated shared memory pool, and `parameters` kernel parameters.
    /// It is undefined behavior to pass in `parameters` that do not conform to the passes CUDA kernel. If the argument count is wrong, CUDA will generally throw an error.
    /// If your `parameters` is accurate to the kernel definition, then this function is otherwise safe.
    pub unsafe fn launch<'b, D1: Into<Dim3>, D2: Into<Dim3>, K: KernelParameters>(
        &mut self,
        f: &Function<'a, 'b>,
        grid_dim: D1,
        block_dim: D2,
        shared_mem_size: u32,
        parameters: K,
    ) -> CudaResult<()> {
        let grid_dim = grid_dim.into().0;
        let block_dim = block_dim.into().0;
        let mut kernel_params = vec![];
        parameters.params(&mut kernel_params);
        let mut new_kernel_params = Vec::with_capacity(kernel_params.len());
        for param in &kernel_params {
            new_kernel_params.push(param.as_ptr() as *mut c_void);
        }
        cuda_error(sys::cuLaunchKernel(
            f.inner,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem_size,
            self.inner,
            new_kernel_params.as_mut_ptr(),
            null_mut(),
        ))
    }
}

impl<'a> Drop for Stream<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuStreamDestroy_v2(self.inner) }) {
            eprintln!("CUDA: failed to drop stream: {:?}", e);
        }
    }
}
