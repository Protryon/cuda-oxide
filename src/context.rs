use std::{cell::RefCell, ptr::null_mut, rc::Rc};

use crate::*;

#[derive(Debug)]
pub struct Context {
    pub(crate) inner: *mut sys::CUctx_st,
}

impl Context {
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

    pub fn version(&self) -> CudaResult<CudaVersion> {
        let mut out = 0u32;
        cuda_error(unsafe { sys::cuCtxGetApiVersion(self.inner, &mut out as *mut u32) })?;
        Ok(out.into())
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuCtxSynchronize() })
    }

    pub fn enter<'a>(&'a mut self) -> CudaResult<Rc<Handle<'a>>> {
        cuda_error(unsafe { sys::cuCtxSetCurrent(self.inner) })?;
        Ok(Rc::new(Handle {
            context: self,
            async_stream_pool: RefCell::new(vec![]),
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

pub struct Handle<'a> {
    pub(crate) context: &'a mut Context,
    async_stream_pool: RefCell<Vec<Stream<'a>>>,
}

impl<'a> Handle<'a> {
    pub fn context(&self) -> &Context {
        &self.context
    }

    pub(crate) fn get_async_stream(self: &Rc<Handle<'a>>) -> CudaResult<Stream<'a>> {
        let mut pool = self.async_stream_pool.borrow_mut();
        if pool.is_empty() {
            Stream::new(self)
        } else {
            Ok(pool.pop().unwrap())
        }
    }

    pub(crate) fn reset_async_stream(self: &Rc<Handle<'a>>, stream: Stream<'a>) {
        let mut pool = self.async_stream_pool.borrow_mut();
        pool.push(stream);
    }
}

impl<'a> Drop for Handle<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuCtxSetCurrent(null_mut()) }) {
            eprintln!("CUDA: error dropping context handle: {:?}", e);
        }
    }
}
