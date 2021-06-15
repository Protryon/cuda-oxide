use std::fmt;

use crate::{
    error::{cuda_error, CudaResult},
    sys, Cuda,
};

/// A CUDA device or API version
#[derive(Clone, Copy, Debug)]
pub struct CudaVersion {
    pub major: u32,
    pub minor: u32,
}

impl From<u32> for CudaVersion {
    fn from(version: u32) -> Self {
        CudaVersion {
            major: version as u32 / 1000,
            minor: (version as u32 % 1000) / 10,
        }
    }
}

impl Into<(u32, u32)> for CudaVersion {
    fn into(self) -> (u32, u32) {
        (self.major, self.minor)
    }
}

impl From<(u32, u32)> for CudaVersion {
    fn from(other: (u32, u32)) -> Self {
        CudaVersion {
            major: other.0,
            minor: other.1,
        }
    }
}

impl fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl Cuda {
    /// Gets the local driver version (not to be confused with device compute capability)
    pub fn version() -> CudaResult<CudaVersion> {
        let mut version = 0i32;
        cuda_error(unsafe { sys::cuDriverGetVersion(&mut version as *mut i32) })?;
        Ok((version as u32).into())
    }
}
