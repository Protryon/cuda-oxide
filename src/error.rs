use num_enum::TryFromPrimitive;
use std::{
    error::Error,
    fmt::{self, Debug},
};

/// A device-sourced or `libcuda`-sourced error code
#[derive(Debug, Copy, Clone, TryFromPrimitive)]
#[repr(u32)]
pub enum ErrorCode {
    #[doc = "The API call returned with no errors. In the case of query calls, this"]
    #[doc = "also means that the operation being queried is complete (see"]
    #[doc = "::cuEventQuery() and ::cuStreamQuery())."]
    Success = 0,
    #[doc = "This indicates that one or more of the parameters passed to the API call"]
    #[doc = "is not within an acceptable range of values."]
    InvalidValue = 1,
    #[doc = "The API call failed because it was unable to allocate enough memory to"]
    #[doc = "perform the requested operation."]
    OutOfMemory = 2,
    #[doc = "This indicates that the CUDA driver has not been initialized with"]
    #[doc = "::cuInit() or that initialization has failed."]
    NotInitialized = 3,
    #[doc = "This indicates that the CUDA driver is in the process of shutting down."]
    Deinitialized = 4,
    #[doc = "This indicates profiler is not initialized for this run. This can"]
    #[doc = "happen when the application is running with external profiling tools"]
    #[doc = "like visual profiler."]
    ProfilerDisabled = 5,
    #[doc = "\\deprecated"]
    #[doc = "This error return is deprecated as of CUDA 5.0. It is no longer an error"]
    #[doc = "to attempt to enable/disable the profiling via ::cuProfilerStart or"]
    #[doc = "::cuProfilerStop without initialization."]
    ProfilerNotInitialized = 6,
    #[doc = "\\deprecated"]
    #[doc = "This error return is deprecated as of CUDA 5.0. It is no longer an error"]
    #[doc = "to call cuProfilerStart() when profiling is already enabled."]
    ProfilerAlreadyStarted = 7,
    #[doc = "\\deprecated"]
    #[doc = "This error return is deprecated as of CUDA 5.0. It is no longer an error"]
    #[doc = "to call cuProfilerStop() when profiling is already disabled."]
    ProfilerAlreadyStopped = 8,
    #[doc = "This indicates that the CUDA driver that the application has loaded is a"]
    #[doc = "stub library. Applications that run with the stub rather than a real"]
    #[doc = "driver loaded will result in CUDA API returning this error."]
    StubLibrary = 34,
    #[doc = "This indicates that no CUDA-capable devices were detected by the installed"]
    #[doc = "CUDA driver."]
    NoDevice = 100,
    #[doc = "This indicates that the device ordinal supplied by the user does not"]
    #[doc = "correspond to a valid CUDA device."]
    InvalidDevice = 101,
    #[doc = "This error indicates that the Grid license is not applied."]
    DeviceNotLicensed = 102,
    #[doc = "This indicates that the device kernel image is invalid. This can also"]
    #[doc = "indicate an invalid CUDA module."]
    InvalidImage = 200,
    #[doc = "This most frequently indicates that there is no context bound to the"]
    #[doc = "current thread. This can also be returned if the context passed to an"]
    #[doc = "API call is not a valid handle (such as a context that has had"]
    #[doc = "::cuCtxDestroy() invoked on it). This can also be returned if a user"]
    #[doc = "mixes different API versions (i.e. 3010 context with 3020 API calls)."]
    #[doc = "See ::cuCtxGetApiVersion() for more details."]
    InvalidContext = 201,
    #[doc = "This indicated that the context being supplied as a parameter to the"]
    #[doc = "API call was already the active context."]
    #[doc = "\\deprecated"]
    #[doc = "This error return is deprecated as of CUDA 3.2. It is no longer an"]
    #[doc = "error to attempt to push the active context via ::cuCtxPushCurrent()."]
    ContextAlreadyCurrent = 202,
    #[doc = "This indicates that a map or register operation has failed."]
    MapFailed = 205,
    #[doc = "This indicates that an unmap or unregister operation has failed."]
    UnmapFailed = 206,
    #[doc = "This indicates that the specified array is currently mapped and thus"]
    #[doc = "cannot be destroyed."]
    ArrayIsMapped = 207,
    #[doc = "This indicates that the resource is already mapped."]
    AlreadyMapped = 208,
    #[doc = "This indicates that there is no kernel image available that is suitable"]
    #[doc = "for the device. This can occur when a user specifies code generation"]
    #[doc = "options for a particular CUDA source file that do not include the"]
    #[doc = "corresponding device configuration."]
    NoBinaryForGpu = 209,
    #[doc = "This indicates that a resource has already been acquired."]
    AlreadyAcquired = 210,
    #[doc = "This indicates that a resource is not mapped."]
    NotMapped = 211,
    #[doc = "This indicates that a mapped resource is not available for access as an"]
    #[doc = "array."]
    NotMappedAsArray = 212,
    #[doc = "This indicates that a mapped resource is not available for access as a"]
    #[doc = "pointer."]
    NotMappedAsPointer = 213,
    #[doc = "This indicates that an uncorrectable ECC error was detected during"]
    #[doc = "execution."]
    EccUncorrectable = 214,
    #[doc = "This indicates that the ::CUlimit passed to the API call is not"]
    #[doc = "supported by the active device."]
    UnsupportedLimit = 215,
    #[doc = "This indicates that the ::CUcontext passed to the API call can"]
    #[doc = "only be bound to a single CPU thread at a time but is already"]
    #[doc = "bound to a CPU thread."]
    ContextAlreadyInUse = 216,
    #[doc = "This indicates that peer access is not supported across the given"]
    #[doc = "devices."]
    PeerAccessUnsupported = 217,
    #[doc = "This indicates that a PTX JIT compilation failed."]
    InvalidPtx = 218,
    #[doc = "This indicates an error with OpenGL or DirectX context."]
    InvalidGraphicsContext = 219,
    #[doc = "This indicates that an uncorrectable NVLink error was detected during the"]
    #[doc = "execution."]
    NvlinkUncorrectable = 220,
    #[doc = "This indicates that the PTX JIT compiler library was not found."]
    JitCompilerNotFound = 221,
    #[doc = "This indicates that the provided PTX was compiled with an unsupported toolchain."]
    UnsupportedPtxVersion = 222,
    #[doc = "This indicates that the PTX JIT compilation was disabled."]
    JitCompilationDisabled = 223,
    #[doc = "This indicates that the device kernel source is invalid."]
    InvalidSource = 300,
    #[doc = "This indicates that the file specified was not found."]
    FileNotFound = 301,
    #[doc = "This indicates that a link to a shared object failed to resolve."]
    SharedObjectSymbolNotFound = 302,
    #[doc = "This indicates that initialization of a shared object failed."]
    SharedObjectInitFailed = 303,
    #[doc = "This indicates that an OS call failed."]
    OperatingSystem = 304,
    #[doc = "This indicates that a resource handle passed to the API call was not"]
    #[doc = "valid. Resource handles are opaque types like ::CUstream and ::CUevent."]
    InvalidHandle = 400,
    #[doc = "This indicates that a resource required by the API call is not in a"]
    #[doc = "valid state to perform the requested operation."]
    IllegalState = 401,
    #[doc = "This indicates that a named symbol was not found. Examples of symbols"]
    #[doc = "are global/constant variable names, driver function names, texture names,"]
    #[doc = "and surface names."]
    NotFound = 500,
    #[doc = "This indicates that asynchronous operations issued previously have not"]
    #[doc = "completed yet. This result is not actually an error, but must be indicated"]
    #[doc = "differently than ::CUDA_SUCCESS (which indicates completion). Calls that"]
    #[doc = "may return this value include ::cuEventQuery() and ::cuStreamQuery()."]
    NotReady = 600,
    #[doc = "While executing a kernel, the device encountered a"]
    #[doc = "load or store instruction on an invalid memory address."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    IllegalAddress = 700,
    #[doc = "This indicates that a launch did not occur because it did not have"]
    #[doc = "appropriate resources. This error usually indicates that the user has"]
    #[doc = "attempted to pass too many arguments to the device kernel, or the"]
    #[doc = "kernel launch specifies too many threads for the kernel's register"]
    #[doc = "count. Passing arguments of the wrong size (i.e. a 64-bit pointer"]
    #[doc = "when a 32-bit int is expected) is equivalent to passing too many"]
    #[doc = "arguments and can also result in this error."]
    LaunchOutOfResources = 701,
    #[doc = "This indicates that the device kernel took too long to execute. This can"]
    #[doc = "only occur if timeouts are enabled - see the device attribute"]
    #[doc = "::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    LaunchTimeout = 702,
    #[doc = "This error indicates a kernel launch that uses an incompatible texturing"]
    #[doc = "mode."]
    LaunchIncompatibleTexturing = 703,
    #[doc = "This error indicates that a call to ::cuCtxEnablePeerAccess() is"]
    #[doc = "trying to re-enable peer access to a context which has already"]
    #[doc = "had peer access to it enabled."]
    PeerAccessAlreadyEnabled = 704,
    #[doc = "This error indicates that ::cuCtxDisablePeerAccess() is"]
    #[doc = "trying to disable peer access which has not been enabled yet"]
    #[doc = "via ::cuCtxEnablePeerAccess()."]
    PeerAccessNotEnabled = 705,
    #[doc = "This error indicates that the primary context for the specified device"]
    #[doc = "has already been initialized."]
    PrimaryContextActive = 708,
    #[doc = "This error indicates that the context current to the calling thread"]
    #[doc = "has been destroyed using ::cuCtxDestroy, or is a primary context which"]
    #[doc = "has not yet been initialized."]
    ContextIsDestroyed = 709,
    #[doc = "A device-side assert triggered during kernel execution. The context"]
    #[doc = "cannot be used anymore, and must be destroyed. All existing device"]
    #[doc = "memory allocations from this context are invalid and must be"]
    #[doc = "reconstructed if the program is to continue using CUDA."]
    Assert = 710,
    #[doc = "This error indicates that the hardware resources required to enable"]
    #[doc = "peer access have been exhausted for one or more of the devices"]
    #[doc = "passed to ::cuCtxEnablePeerAccess()."]
    TooManyPeers = 711,
    #[doc = "This error indicates that the memory range passed to ::cuMemHostRegister()"]
    #[doc = "has already been registered."]
    HostMemoryAlreadyRegistered = 712,
    #[doc = "This error indicates that the pointer passed to ::cuMemHostUnregister()"]
    #[doc = "does not correspond to any currently registered memory region."]
    HostMemoryNotRegistered = 713,
    #[doc = "While executing a kernel, the device encountered a stack error."]
    #[doc = "This can be due to stack corruption or exceeding the stack size limit."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    HardwareStackError = 714,
    #[doc = "While executing a kernel, the device encountered an illegal instruction."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    IllegalInstruction = 715,
    #[doc = "While executing a kernel, the device encountered a load or store instruction"]
    #[doc = "on a memory address which is not aligned."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    MisalignedAddress = 716,
    #[doc = "While executing a kernel, the device encountered an instruction"]
    #[doc = "which can only operate on memory locations in certain address spaces"]
    #[doc = "(global, shared, or local), but was supplied a memory address not"]
    #[doc = "belonging to an allowed address space."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    InvalidAddressSpace = 717,
    #[doc = "While executing a kernel, the device program counter wrapped its address space."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    InvalidPc = 718,
    #[doc = "An exception occurred on the device while executing a kernel. Common"]
    #[doc = "causes include dereferencing an invalid device pointer and accessing"]
    #[doc = "out of bounds shared memory. Less common cases can be system specific - more"]
    #[doc = "information about these cases can be found in the system specific user guide."]
    #[doc = "This leaves the process in an inconsistent state and any further CUDA work"]
    #[doc = "will return the same error. To continue using CUDA, the process must be terminated"]
    #[doc = "and relaunched."]
    LaunchFailed = 719,
    #[doc = "This error indicates that the number of blocks launched per grid for a kernel that was"]
    #[doc = "launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice"]
    #[doc = "exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor"]
    #[doc = "or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors"]
    #[doc = "as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT."]
    CooperativeLaunchTooLarge = 720,
    #[doc = "This error indicates that the attempted operation is not permitted."]
    NotPermitted = 800,
    #[doc = "This error indicates that the attempted operation is not supported"]
    #[doc = "on the current system or device."]
    NotSupported = 801,
    #[doc = "This error indicates that the system is not yet ready to start any CUDA"]
    #[doc = "work.  To continue using CUDA, verify the system configuration is in a"]
    #[doc = "valid state and all required driver daemons are actively running."]
    #[doc = "More information about this error can be found in the system specific"]
    #[doc = "user guide."]
    SystemNotReady = 802,
    #[doc = "This error indicates that there is a mismatch between the versions of"]
    #[doc = "the display driver and the CUDA driver. Refer to the compatibility documentation"]
    #[doc = "for supported versions."]
    SystemDriverMismatch = 803,
    #[doc = "This error indicates that the system was upgraded to run with forward compatibility"]
    #[doc = "but the visible hardware detected by CUDA does not support this configuration."]
    #[doc = "Refer to the compatibility documentation for the supported hardware matrix or ensure"]
    #[doc = "that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES"]
    #[doc = "environment variable."]
    CompatNotSupportedOnDevice = 804,
    #[doc = "This error indicates that the operation is not permitted when"]
    #[doc = "the stream is capturing."]
    StreamCaptureUnsupported = 900,
    #[doc = "This error indicates that the current capture sequence on the stream"]
    #[doc = "has been invalidated due to a previous error."]
    StreamCaptureInvalidated = 901,
    #[doc = "This error indicates that the operation would have resulted in a merge"]
    #[doc = "of two independent capture sequences."]
    StreamCaptureMerge = 902,
    #[doc = "This error indicates that the capture was not initiated in this stream."]
    StreamCaptureUnmatched = 903,
    #[doc = "This error indicates that the capture sequence contains a fork that was"]
    #[doc = "not joined to the primary stream."]
    StreamCaptureUnjoined = 904,
    #[doc = "This error indicates that a dependency would have been created which"]
    #[doc = "crosses the capture sequence boundary. Only implicit in-stream ordering"]
    #[doc = "dependencies are allowed to cross the boundary."]
    StreamCaptureIsolation = 905,
    #[doc = "This error indicates a disallowed implicit dependency on a current capture"]
    #[doc = "sequence from cudaStreamLegacy."]
    StreamCaptureImplicit = 906,
    #[doc = "This error indicates that the operation is not permitted on an event which"]
    #[doc = "was last recorded in a capturing stream."]
    CapturedEvent = 907,
    #[doc = "A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED"]
    #[doc = "argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a"]
    #[doc = "different thread."]
    StreamCaptureWrongThread = 908,
    #[doc = "This error indicates that the timeout specified for the wait operation has lapsed."]
    Timeout = 909,
    #[doc = "This error indicates that the graph update was not performed because it included"]
    #[doc = "changes which violated constraints specific to instantiated graph update."]
    GraphExecUpdateFailure = 910,
    #[doc = "This indicates that an unknown internal error has occurred."]
    Unknown = 999,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as Debug>::fmt(self, f)
    }
}

impl Error for ErrorCode {}

pub type CudaResult<T> = Result<T, ErrorCode>;

pub(crate) fn cuda_error(input: u32) -> CudaResult<()> {
    if input == 0 {
        Ok(())
    } else {
        Err(ErrorCode::try_from_primitive(input).unwrap_or(ErrorCode::Unknown))
    }
}
