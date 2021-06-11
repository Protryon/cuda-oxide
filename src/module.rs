use std::{ffi::CString, ptr::null_mut, rc::Rc};

use crate::*;

/// A loaded CUDA module
pub struct Module<'a> {
    handle: Rc<Handle<'a>>,
    inner: *mut sys::CUmod_st,
}

impl<'a> Module<'a> {
    /// Takes a raw CUDA kernel image and loads the corresponding module module into the current context.
    /// The pointer can be a cubin or PTX or fatbin file as a NULL-terminated text string
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

    /// Same as [`Module::load`] but uses `fatCubin` format.
    pub fn load_fatcubin(handle: &Rc<Handle<'a>>, module: &[u8]) -> CudaResult<Self> {
        let mut inner = null_mut();
        cuda_error(unsafe {
            sys::cuModuleLoadFatBinary(&mut inner as *mut _, module.as_ptr() as *const _)
        })?;
        Ok(Module {
            inner,
            handle: handle.clone(),
        })
    }

    /// Retrieve a reference to a define CUDA kernel within the module.
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

    /// Get a pointer to a global variable defined by a CUDA module.
    pub fn get_global<'b: 'a>(&'b self, name: &str) -> CudaResult<DevicePtr<'b>> {
        let mut out = DevicePtr {
            handle: self.handle.clone(),
            inner: 0,
            len: 0,
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
