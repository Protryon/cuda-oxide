use crate::{
    error::{cuda_error, CudaResult},
    sys, Cuda,
};

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

impl Cuda {
    pub fn version() -> CudaResult<CudaVersion> {
        let mut version = 0i32;
        cuda_error(unsafe { sys::cuDriverGetVersion(&mut version as *mut i32) })?;
        Ok((version as u32).into())
    }
}
