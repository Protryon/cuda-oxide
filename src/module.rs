use std::{ffi::CString, marker::PhantomData, ptr::null_mut, rc::Rc};

use crate::*;

pub struct Module<'a> {
    handle: Rc<Handle<'a>>,
    inner: *mut sys::CUmod_st,
}

impl<'a> Module<'a> {
    pub fn load(handle: &Rc<Handle<'a>>, module: &[u8]) -> CudaResult<Self> {
        let mut inner = null_mut();
        cuda_error(unsafe {
            sys::cuModuleLoadData(&mut inner as *mut _, module.as_ptr() as *const _)
        })?;
        Ok(Module {
            inner,
            handle: handle.clone(),
        })
    }

    pub fn load_fat(handle: &Rc<Handle<'a>>, module: &[u8]) -> CudaResult<Self> {
        let mut inner = null_mut();
        cuda_error(unsafe {
            sys::cuModuleLoadFatBinary(&mut inner as *mut _, module.as_ptr() as *const _)
        })?;
        Ok(Module {
            inner,
            handle: handle.clone(),
        })
    }

    pub fn get_function<'b>(&'b self, name: &str) -> CudaResult<Function<'a, 'b>> {
        let mut inner = null_mut();
        let name = CString::new(name).unwrap();
        cuda_error(unsafe {
            sys::cuModuleGetFunction(&mut inner as *mut _, self.inner, name.as_ptr())
        })?;
        Ok(Function {
            module: self,
            inner,
        })
    }

    pub fn get_global<'b: 'a>(&'b self, name: &str) -> CudaResult<DevicePtr<'b>> {
        let mut out = DevicePtr {
            handle: self.handle.clone(),
            inner: 0,
            len: 0,
            _p: PhantomData,
        };
        let name = CString::new(name).unwrap();
        cuda_error(unsafe {
            sys::cuModuleGetGlobal_v2(
                &mut out.inner,
                &mut out.len as *mut u64 as *mut _,
                self.inner,
                name.as_ptr(),
            )
        })?;
        Ok(out)
    }

    // pub fn get_surface(&self, name: &str) {

    // }

    // pub fn get_texture(&self, name: &str) {

    // }
}

impl<'a> Drop for Module<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuModuleUnload(self.inner) }) {
            eprintln!("CUDA: failed to destroy cuda module: {:?}", e);
        }
    }
}
