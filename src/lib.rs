#[allow(
    non_upper_case_globals,
    non_snake_case,
    improper_ctypes,
    non_camel_case_types
)]
pub mod sys;

pub mod context;
pub mod device;
pub mod dim3;
pub mod error;
pub mod exec;
pub mod func;
pub mod future;
pub mod init;
pub mod kernel_params;
pub mod mem;
pub mod module;
pub mod stream;
pub mod version;

pub struct Cuda;

pub use context::*;
pub use device::*;
pub use dim3::*;
pub use error::*;
pub use func::*;
pub use func::*;
pub use future::*;
pub use kernel_params::*;
pub use mem::*;
pub use module::*;
pub use stream::*;
pub use version::*;
