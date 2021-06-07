use crate::*;
use num_enum::TryFromPrimitive;

#[derive(Debug, Copy, Clone, TryFromPrimitive)]
#[repr(u32)]
pub enum FunctionAttribute {
    /// The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.
    MaxThreadsPerBlock = 0,
    /// The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.
    SharedSizeBytes = 1,
    /// The size in bytes of user-allocated constant memory required by this function.
    ConstSizeBytes = 2,
    /// The size in bytes of local memory used by each thread of this function.
    LocalSizeBytes = 3,
    /// The number of registers used by each thread of this function.
    NumRegs = 4,
    /// The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.
    PtxVersion = 5,
    /// The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.
    BinaryVersion = 6,
    /// The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set .
    CacheModeCa = 7,
    /// The maximum size in bytes of dynamically-allocated shared memory that can be used by this function. If the user-specified dynamic shared memory size is larger than this value, the launch will fail. See cuFuncSetAttribute
    MaxDynamicSharedSizeBytes = 8,
    /// On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. Refer to CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR. This is only a hint, and the driver can choose a different ratio if required to execute the function. See cuFuncSetAttribute
    PreferredSharedMemoryCarveout = 9,
}

#[derive(Debug, Copy, Clone, TryFromPrimitive)]
#[repr(u32)]
pub enum FuncCache {
    /// no preference for shared memory or L1 (default)
    PreferNone = 0x00,
    /// prefer larger shared memory and smaller L1 cache
    PreferShared = 0x01,
    /// prefer larger L1 cache and smaller shared memory
    PreferL1 = 0x02,
    /// prefer equal sized L1 cache and shared memory
    PreferEqual = 0x03,
}

#[derive(Debug, Copy, Clone, TryFromPrimitive)]
#[repr(u32)]
pub enum FuncSharedConfig {
    /// set default shared memory bank size
    DefaultBankSize = 0x00,
    /// set shared memory bank width to four bytes
    FourByteBankSize = 0x01,
    /// set shared memory bank width to eight bytes
    EightByteBankSize = 0x02,
}

pub struct Function<'a, 'b> {
    pub(crate) module: &'b Module<'a>,
    pub(crate) inner: *mut sys::CUfunc_st,
}

impl<'a, 'b> Function<'a, 'b> {
    /// Returns a module handle.
    pub fn module(&self) -> &'b Module<'a> {
        self.module
    }

    /// Returns information about a function.
    pub fn get_attribute(&self, attribute: FunctionAttribute) -> CudaResult<i32> {
        let mut out = 0i32;
        cuda_error(unsafe {
            sys::cuFuncGetAttribute(&mut out as *mut i32, attribute as u32, self.inner)
        })?;
        Ok(out)
    }

    /// Sets information about a function.
    pub fn set_attribute(&mut self, attribute: FunctionAttribute, value: i32) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuFuncSetAttribute(self.inner, attribute as u32, value) })
    }

    /// Sets the preferred cache configuration for a device function.
    pub fn set_cache_config(&mut self, func_cache: FuncCache) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuFuncSetCacheConfig(self.inner, func_cache as u32) })
    }

    /// Sets the shared memory configuration for a device function.
    pub fn set_shared_mem_config(&mut self, config: FuncSharedConfig) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuFuncSetSharedMemConfig(self.inner, config as u32) })
    }
}
