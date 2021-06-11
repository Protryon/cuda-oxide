use crate::*;
use num_enum::TryFromPrimitive;

/// A reference to a CUDA-enabled device
pub struct Device {
    pub(crate) handle: i32,
}

/// Type of native array format
#[derive(Clone, Copy, Debug, TryFromPrimitive)]
#[repr(u32)]
pub enum CudaArrayFormat {
    UnsignedInt8 = 0x01,
    UnsignedInt16 = 0x02,
    UnsignedInt32 = 0x03,
    SignedInt8 = 0x08,
    SignedInt16 = 0x09,
    SignedInt32 = 0x0a,
    Half = 0x10,
    Float = 0x20,
    Nv12 = 0xb0,
}

impl Device {
    /// Fetches a human-readable name from the device
    pub fn name(&self) -> CudaResult<String> {
        let mut buf = [0u8; 256];
        cuda_error(unsafe {
            sys::cuDeviceGetName(&mut buf as *mut u8 as *mut i8, 256, self.handle)
        })?;
        Ok(
            String::from_utf8_lossy(&buf[..buf.iter().position(|x| *x == 0).unwrap_or(0)])
                .into_owned(),
        )
    }

    /// Gets a UUID from the device
    pub fn uuid(&self) -> CudaResult<u128> {
        let mut out = 0u128;
        cuda_error(unsafe { sys::cuDeviceGetUuid(&mut out as *mut u128 as *mut _, self.handle) })?;
        Ok(out)
    }

    /// Gets the total available memory size of the device, in bytes
    pub fn memory_size(&self) -> CudaResult<usize> {
        let mut memory_size = 0usize;
        cuda_error(unsafe {
            sys::cuDeviceTotalMem_v2(&mut memory_size as *mut usize as *mut _, self.handle)
        })?;
        Ok(memory_size)
    }

    /// Gets a current attribute value for the device
    pub fn get_attribute(&self, attribute: DeviceAttribute) -> CudaResult<i32> {
        let mut out = 0i32;
        cuda_error(unsafe {
            sys::cuDeviceGetAttribute(&mut out as *mut i32, attribute as u32, self.handle)
        })?;
        Ok(out)
    }

    /// Calculates the linear max width of 1D textures for a given native array format
    pub fn get_texture_1d_linear_max_width(
        &self,
        format: CudaArrayFormat,
        channels: u32,
    ) -> CudaResult<usize> {
        let mut out = 0usize;
        cuda_error(unsafe {
            sys::cuDeviceGetTexture1DLinearMaxWidth(
                &mut out as *mut usize as *mut _,
                format as u32,
                channels,
                self.handle,
            )
        })?;
        Ok(out)
    }
}

impl Cuda {
    /// List all CUDA-enabled devices on the host
    pub fn list_devices() -> CudaResult<Vec<Device>> {
        let mut count = 0i32;
        cuda_error(unsafe { sys::cuDeviceGetCount(&mut count as *mut i32) })?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut device = Device { handle: 0 };
            cuda_error(unsafe { sys::cuDeviceGet(&mut device.handle as *mut i32, i) })?;
            out.push(device);
        }
        Ok(out)
    }
}

/// A [`Device`]-specific attribute type
#[derive(Clone, Copy, Debug, TryFromPrimitive)]
#[repr(u32)]
pub enum DeviceAttribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    SharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxPitch = 11,
    RegistersPerBlock = 12,
    ClockRate = 13,
    TextureAlignment = 14,
    GpuOverlap = 15,
    MultiprocessorCount = 16,
    KernelExecTimeout = 17,
    Integrated = 18,
    CanMapHostMemory = 19,
    ComputeMode = 20,
    MaximumTexture1dWidth = 21,
    MaximumTexture2dWidth = 22,
    MaximumTexture2dHeight = 23,
    MaximumTexture3dWidth = 24,
    MaximumTexture3dHeight = 25,
    MaximumTexture3dDepth = 26,
    MaximumTexture2dArrayWidth = 27,
    MaximumTexture2dArrayHeight = 28,
    MaximumTexture2dArrayNumslices = 29,
    SurfaceAlignment = 30,
    ConcurrentKernels = 31,
    EccEnabled = 32,
    PciBusId = 33,
    PciDeviceId = 34,
    TccDriver = 35,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    MaximumTexture1dLayeredWidth = 42,
    MaximumTexture1dLayeredLayers = 43,
    CanTex2dGather = 44,
    MaximumTexture2dGatherWidth45,
    MaximumTexture2dGatherHeight = 46,
    MaximumTexture3dWidthAlternate = 47,
    MaximumTexture3dHeightAlternate = 48,
    MaximumTexture3dDepthAlternate = 49,
    PciDomainId = 50,
    TexturePitchAlignment = 51,
    MaximumTexturecubemapWidth = 52,
    MaximumTexturecubemapLayeredWidth = 53,
    MaximumTexturecubemapLayeredLayers = 54,
    MaximumSurface1dWidth = 55,
    MaximumSurface2dWidth = 56,
    MaximumSurface2dHeight = 57,
    MaximumSurface3dWidth = 58,
    MaximumSurface3dHeight = 59,
    MaximumSurface3dDepth = 60,
    MaximumSurface1dLayeredWidth = 61,
    MaximumSurface1dLayeredLayers = 62,
    MaximumSurface2dLayeredWidth = 63,
    MaximumSurface2dLayeredHeight = 64,
    MaximumSurface2dLayeredLayers = 65,
    MaximumSurfacecubemapWidth = 66,
    MaximumSurfacecubemapLayeredWidth = 67,
    MaximumSurfacecubemapLayeredLayers = 68,
    MaximumTexture1dLinearWidth = 69,
    MaximumTexture2dLinearWidth = 70,
    MaximumTexture2dLinearHeight = 71,
    MaximumTexture2dLinearPitch = 72,
    MaximumTexture2dMipmappedWidth = 73,
    MaximumTexture2dMipmappedHeight = 74,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaximumTexture1dMipmappedWidth = 77,
    StreamPrioritiesSupported = 78,
    GlobalL1CacheSupported = 79,
    LocalL1CacheSupported = 80,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    ManagedMemory = 83,
    MultiGpuBoard = 84,
    MultiGpuBoardGroupId = 85,
    HostNativeAtomicSupported = 86,
    SingleToDoublePrecisionPerfRatio = 87,
    PageableMemoryAccess = 88,
    ConcurrentManagedAccess = 89,
    ComputePreemptionSupported = 90,
    CanUseHostPointerForRegisteredMem = 91,
    CanUseStreamMemOps = 92,
    CanUse64BitStreamMemOps = 93,
    CanUseStreamWaitValueNor = 94,
    CooperativeLaunch = 95,
    CooperativeMultiDeviceLaunch = 96,
    MaxSharedMemoryPerBlockOptin = 97,
    CanFlushRemoteWrites = 98,
    HostRegisterSupported = 99,
    PageableMemoryAccessUsesHostPageTables = 100,
    DirectManagedMemAccessFromHost = 101,
    VirtualMemoryManagementSupported = 102,
    HandleTypePosixFileDescriptorSupported = 103,
    HandleTypeWin32HandleSupported = 104,
    HandleTypeWin32KmtHandleSupported = 105,
    MaxBlocksPerMultiprocessor = 106,
    GenericCompressionSupported = 107,
    MaxPersistingL2CacheSize = 108,
    MaxAccessPolicyWindowSize = 109,
    GpuDirectRdmaWithCudaVmmSupported = 110,
    ReservedSharedMemoryPerBlock = 111,
    SparseCudaArraySupported = 112,
    ReadOnlyHostRegisterSupported = 113,
    TimelineSemaphoreInteropSupported = 114,
    MemoryPoolsSupported = 115,
    GpuDirectRdmaSupported = 116,
    GpuDirectRdmaFlushWritesOptions = 117,
    GpuDirectRdmaWritesOrdering = 118,
    MempoolSupportedHandleTypes = 119,
}
