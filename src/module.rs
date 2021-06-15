use std::{
    borrow::Cow,
    ffi::{c_void, CString},
    ptr::null_mut,
    rc::Rc,
};

use crate::*;

// Debug must not be derived, see comment on info_buf
/// A CUDA JIT linker context, used to compile device-specific kernels from PTX assembly or link together several precompiled binaries
pub struct Linker<'a> {
    inner: *mut sys::CUlinkState_st,
    info_buf: Vec<u8>, // both info_buf and errors_buf contain uninitialized memory! they should always be NUL terminated strings
    errors_buf: Vec<u8>,
    handle: Rc<Handle<'a>>,
}

/// The type of input to the linker
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinkerInputType {
    Cubin,
    Ptx,
    Fatbin,
}

/// Linker options for CUDA, can generally just be defaulted.
#[derive(Clone, Copy, Debug)]
pub struct LinkerOptions {
    /// Add debug symbols to emitted binary
    pub debug_info: bool,
    /// Collect INFO logs from CUDA build/link, up to 16 MB, then emit to STDOUT
    pub log_info: bool,
    /// Collect ERROR logs from CUDA build/link, up to 16 MB, then emit to STDOUT
    pub log_errors: bool,
    /// Increase log verbosity
    pub verbose_logs: bool,
}

impl Default for LinkerOptions {
    fn default() -> Self {
        LinkerOptions {
            debug_info: false,
            log_info: true,
            log_errors: true,
            verbose_logs: false,
        }
    }
}

impl<'a> Linker<'a> {
    /// Creates a new [`Linker`] for the given context handle, compute capability, and linker options.
    pub fn new(
        handle: &Rc<Handle<'a>>,
        compute_capability: CudaVersion,
        options: LinkerOptions,
    ) -> CudaResult<Self> {
        let mut linker = Linker {
            inner: null_mut(),
            info_buf: if options.log_info {
                let mut buf = Vec::with_capacity(16 * 1024 * 1024);
                buf.push(0);
                unsafe { buf.set_len(buf.capacity()) };
                buf
            } else {
                vec![]
            },
            errors_buf: if options.log_errors {
                let mut buf = Vec::with_capacity(16 * 1024 * 1024);
                buf.push(0);
                unsafe { buf.set_len(buf.capacity()) };
                buf
            } else {
                vec![]
            },
            handle: handle.clone(),
        };
        let log_verbose = if options.verbose_logs { 1u32 } else { 0u32 };
        let debug_info = if options.debug_info { 1u32 } else { 0u32 };

        let mut options = [
            sys::CUjit_option_enum_CU_JIT_INFO_LOG_BUFFER,
            sys::CUjit_option_enum_CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            sys::CUjit_option_enum_CU_JIT_ERROR_LOG_BUFFER,
            sys::CUjit_option_enum_CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            sys::CUjit_option_enum_CU_JIT_TARGET,
            sys::CUjit_option_enum_CU_JIT_LOG_VERBOSE,
            sys::CUjit_option_enum_CU_JIT_GENERATE_DEBUG_INFO,
        ];
        let target = match (compute_capability.major, compute_capability.minor) {
            (2, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_20,
            (2, 1) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_21,
            (3, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_30,
            (3, 2) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_32,
            (3, 5) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_35,
            (3, 7) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_37,
            (5, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_50,
            (5, 2) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_52,
            (5, 3) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_53,
            (6, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_60,
            (6, 1) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_61,
            (6, 2) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_62,
            (7, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_70,
            (7, 2) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_72,
            (7, 5) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_75,
            (8, 0) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_80,
            (8, 6) => sys::CUjit_target_enum_CU_TARGET_COMPUTE_86,
            (_, _) => return Err(ErrorCode::UnsupportedPtxVersion),
        };

        let mut values = [
            linker.info_buf.as_mut_ptr() as *mut c_void,
            linker.info_buf.len() as u32 as u64 as *mut c_void,
            linker.errors_buf.as_mut_ptr() as *mut c_void,
            linker.errors_buf.len() as u32 as u64 as *mut c_void,
            target as u64 as *mut c_void,
            log_verbose as u64 as *mut c_void,
            debug_info as u64 as *mut c_void,
        ];
        cuda_error(unsafe {
            sys::cuLinkCreate_v2(
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
                &mut linker.inner as *mut _,
            )
        })?;
        Ok(linker)
    }

    fn emit_logs(&self) {
        let info_string = self.info_buf.iter().position(|x| *x == 0);
        if let Some(info_string) = info_string {
            let info_string = String::from_utf8_lossy(&self.info_buf[..info_string]);
            if !info_string.is_empty() {
                info_string.split('\n').for_each(|line| {
                    println!("[CUDA INFO] {}", line);
                });
            }
        }
        let error_string = self.errors_buf.iter().position(|x| *x == 0);
        if let Some(error_string) = error_string {
            let error_string = String::from_utf8_lossy(&self.errors_buf[..error_string]);
            if !error_string.is_empty() {
                error_string.split('\n').for_each(|line| {
                    println!("[CUDA ERROR] {}", line);
                });
            }
        }
    }

    /// Add an input file to the linker context. `name` is only used for logs
    pub fn add(self, name: &str, format: LinkerInputType, in_data: &[u8]) -> CudaResult<Self> {
        let mut data = Cow::Borrowed(in_data);
        if format == LinkerInputType::Ptx {
            let mut new_data = Vec::with_capacity(in_data.len() + 1);
            new_data.extend_from_slice(in_data);
            new_data.push(0);
            data = Cow::Owned(new_data)
        }

        let format = match format {
            LinkerInputType::Cubin => sys::CUjitInputType_enum_CU_JIT_INPUT_CUBIN,
            LinkerInputType::Ptx => sys::CUjitInputType_enum_CU_JIT_INPUT_PTX,
            LinkerInputType::Fatbin => sys::CUjitInputType_enum_CU_JIT_INPUT_FATBINARY,
        };
        let name = CString::new(name).unwrap();

        let out = cuda_error(unsafe {
            sys::cuLinkAddData_v2(
                self.inner,
                format,
                data.as_ptr() as *mut u8 as *mut c_void,
                data.len() as u64,
                name.as_ptr(),
                0,
                null_mut(),
                null_mut(),
            )
        });

        if let Err(e) = out {
            self.emit_logs();
            return Err(e);
        }
        Ok(self)
    }

    /// Emit the cubin assembly binary. You probably want [`Linker::build_module`]
    pub fn build(&self) -> CudaResult<&[u8]> {
        let mut cubin_out: *mut c_void = null_mut();
        let mut size_out = 0usize;
        let out = cuda_error(unsafe {
            sys::cuLinkComplete(
                self.inner,
                &mut cubin_out as *mut *mut c_void,
                &mut size_out as *mut usize as *mut u64,
            )
        });
        self.emit_logs();
        if let Err(e) = out {
            return Err(e);
        }
        Ok(unsafe { std::slice::from_raw_parts(cubin_out as *const u8, size_out) })
    }

    /// Build a CUDA module from this [`Linker`].
    pub fn build_module(&self) -> CudaResult<Module<'a>> {
        let built = self.build()?;
        Module::load(&self.handle, built)
    }
}

impl<'a> Drop for Linker<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuLinkDestroy(self.inner) }) {
            eprintln!("CUDA: failed to destroy cuda linker state: {:?}", e);
        }
    }
}

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
