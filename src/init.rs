use std::sync::atomic::{AtomicBool, Ordering};

use crate::{error::*, sys, Cuda};

static CHECK_INIT: AtomicBool = AtomicBool::new(false);
impl Cuda {
    /// Initialize the CUDA library. Can be called repeatedly at no cost.
    pub fn init() -> CudaResult<()> {
        if CHECK_INIT.load(Ordering::SeqCst) {
            return Ok(());
        }
        cuda_error(unsafe { sys::cuInit(0) })?;
        CHECK_INIT.store(true, Ordering::SeqCst);
        Ok(())
    }
}
