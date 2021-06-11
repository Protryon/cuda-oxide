use crate::*;
use num_enum::TryFromPrimitive;
use std::{ptr::null_mut, rc::Rc};

/// A CUDA application context.
/// To start interacting with a device, you want to [`Context::enter`]
#[derive(Debug)]
pub struct Context {
    pub(crate) inner: *mut sys::CUctx_st,
}

impl Context {
    /// Creates a new [`Context`] for a given [`Device`]
    pub fn new(device: &Device) -> CudaResult<Context> {
        let mut inner = null_mut();
        cuda_error(unsafe {
            sys::cuCtxCreate_v2(
                &mut inner as *mut _,
                sys::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC,
                device.handle,
            )
        })?;
        Ok(Context { inner })
    }

    /// Gets the api version of the Context
    pub fn version(&self) -> CudaResult<CudaVersion> {
        let mut out = 0u32;
        cuda_error(unsafe { sys::cuCtxGetApiVersion(self.inner, &mut out as *mut u32) })?;
        Ok(out.into())
    }

    /// Synchronize a Context, running all active handles to completion
    pub fn synchronize(&self) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuCtxSynchronize() })
    }

    /// Set a CUDA context limit
    pub fn set_limit(&mut self, limit: LimitType, value: u64) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuCtxSetLimit(limit as u32, value) })
    }

    /// Get a CUDA context limit
    pub fn get_limit(&self, limit: LimitType) -> CudaResult<u64> {
        let mut out = 0u64;
        cuda_error(unsafe { sys::cuCtxGetLimit(&mut out as *mut u64, limit as u32) })?;
        Ok(out)
    }

    /// Enter a Context, consuming a mutable reference to the context, and allowing thread-local operations to happen.
    pub fn enter<'a>(&'a mut self) -> CudaResult<Rc<Handle<'a>>> {
        cuda_error(unsafe { sys::cuCtxSetCurrent(self.inner) })?;
        Ok(Rc::new(Handle {
            context: self,
            // async_stream_pool: RefCell::new(vec![]),
        }))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuCtxDestroy_v2(self.inner) }) {
            eprintln!("CUDA: failed to destroy cuda context: {:?}", e);
        }
    }
}

/// A CUDA [`Context`] handle for executing thread-local operations.
pub struct Handle<'a> {
    pub(crate) context: &'a mut Context,
    // async_stream_pool: RefCell<Vec<Stream<'a>>>,
}

impl<'a> Handle<'a> {
    /// Get an immutable reference to the source context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    // pub(crate) fn get_async_stream(self: &Rc<Handle<'a>>) -> CudaResult<Stream<'a>> {
    //     let mut pool = self.async_stream_pool.borrow_mut();
    //     if pool.is_empty() {
    //         Stream::new(self)
    //     } else {
    //         Ok(pool.pop().unwrap())
    //     }
    // }

    // pub(crate) fn reset_async_stream(self: &Rc<Handle<'a>>, stream: Stream<'a>) {
    //     let mut pool = self.async_stream_pool.borrow_mut();
    //     pool.push(stream);
    // }
}

impl<'a> Drop for Handle<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuCtxSetCurrent(null_mut()) }) {
            eprintln!("CUDA: error dropping context handle: {:?}", e);
        }
    }
}

/// Context limit types
#[derive(Clone, Copy, Debug, TryFromPrimitive)]
#[repr(u32)]
pub enum LimitType {
    /// GPU thread stack size
    StackSize = 0x00,
    /// GPU printf FIFO size
    PrintfFifoSize = 0x01,
    /// GPU malloc heap size
    MallocHeapSize = 0x02,
    /// GPU device runtime launch synchronize depth
    DevRuntimeSyncDepth = 0x03,
    /// GPU device runtime pending launch count
    DevRuntimePendingLaunchCount = 0x04,
    /// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
    MaxL2FetchGranularity = 0x05,
    /// A size in bytes for L2 persisting lines cache size
    PersistingL2CacheSize = 0x06,
}
