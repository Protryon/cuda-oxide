use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::*;

#[derive(Clone)]
pub struct DevicePtr<'a> {
    pub(crate) handle: Rc<Handle<'a>>,
    pub(crate) inner: u64,
    pub(crate) len: u64,
    pub(crate) _p: PhantomData<&'a ()>,
}

impl<'a> DevicePtr<'a> {
    pub fn copy_to<'b>(&self, target: &mut DevicePtr<'b>) -> CudaResult<()> {
        if self.len > target.len {
            panic!("overflow in DevicePtr::copy_to");
        } else if self.len < target.len {
            panic!("underflow in DevicePtr::copy_to");
        }

        if std::ptr::eq(self.handle.context, target.handle.context) {
            cuda_error(unsafe { sys::cuMemcpy(target.inner, self.inner, self.len as sys::size_t) })
        } else {
            cuda_error(unsafe {
                sys::cuMemcpyPeer(
                    target.inner,
                    target.handle.context.inner,
                    self.inner,
                    self.handle.context.inner,
                    self.len as sys::size_t,
                )
            })
        }
    }

    pub unsafe fn copy_to_stream<'b, 'c: 'b + 'a>(
        &self,
        target: &mut DevicePtr<'b>,
        stream: &mut Stream<'c>,
    ) -> CudaResult<()>
    where
        'a: 'b,
    {
        if self.len > target.len {
            panic!("overflow in DevicePtr::copy_to");
        } else if self.len < target.len {
            panic!("underflow in DevicePtr::copy_to");
        }

        if std::ptr::eq(self.handle.context, target.handle.context) {
            cuda_error(sys::cuMemcpyAsync(
                target.inner,
                self.inner,
                self.len as sys::size_t,
                stream.inner,
            ))
        } else {
            cuda_error(sys::cuMemcpyPeerAsync(
                target.inner,
                target.handle.context.inner,
                self.inner,
                self.handle.context.inner,
                self.len as sys::size_t,
                stream.inner,
            ))
        }
    }

    pub fn copy_to_async<'b>(&self, target: &mut DevicePtr<'b>) -> CudaResult<CudaFuture<'a, ()>>
    where
        'a: 'b,
    {
        let mut stream = self.handle.get_async_stream()?;
        unsafe { self.copy_to_stream(target, &mut stream) }?;
        Ok(CudaFuture::new(self.handle.clone(), stream))
    }

    pub fn copy_from<'b>(&mut self, source: &DevicePtr<'b>) -> CudaResult<()> {
        source.copy_to(self)
    }

    pub unsafe fn copy_from_stream<'b: 'a, 'c: 'a + 'b>(
        &mut self,
        source: &DevicePtr<'b>,
        stream: &mut Stream<'c>,
    ) -> CudaResult<()> {
        source.copy_to_stream(self, stream)
    }

    pub fn subslice(&self, from: u64, to: u64) -> Self {
        if from > self.len || from > to || to > self.len {
            panic!("overflow in DevicePtr::subslice");
        }
        Self {
            handle: self.handle.clone(),
            inner: self.inner + from,
            len: to - from,
            _p: PhantomData,
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn load(&self) -> CudaResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.len as usize);
        cuda_error(unsafe {
            sys::cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, self.inner, self.len)
        })?;
        unsafe { buf.set_len(self.len as usize) };
        Ok(buf)
    }

    pub unsafe fn load_stream(&self, stream: &mut Stream<'a>) -> CudaResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.len as usize);
        cuda_error(sys::cuMemcpyDtoHAsync_v2(
            buf.as_mut_ptr() as *mut _,
            self.inner,
            self.len,
            stream.inner,
        ))?;
        buf.set_len(self.len as usize);
        Ok(buf)
    }

    pub fn store(&mut self, data: &[u8]) -> CudaResult<()> {
        if data.len() > self.len as usize {
            panic!("overflow in DevicePtr::store");
        } else if data.len() < self.len as usize {
            panic!("underflow in DevicePtr::store");
        }
        cuda_error(unsafe {
            sys::cuMemcpyHtoD_v2(self.inner, data.as_ptr() as *const _, self.len)
        })?;
        Ok(())
    }

    pub unsafe fn store_stream(&mut self, data: &[u8], stream: &mut Stream<'a>) -> CudaResult<()> {
        if data.len() > self.len as usize {
            panic!("overflow in DevicePtr::store");
        } else if data.len() < self.len as usize {
            panic!("underflow in DevicePtr::store");
        }
        cuda_error(sys::cuMemcpyHtoDAsync_v2(
            self.inner,
            data.as_ptr() as *const _,
            self.len,
            stream.inner,
        ))?;
        Ok(())
    }

    pub fn memset_d8(&mut self, data: u8) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuMemsetD8_v2(self.inner, data, self.len) })
    }

    pub unsafe fn memset_d8_stream(&mut self, data: u8, stream: &mut Stream<'a>) -> CudaResult<()> {
        cuda_error(sys::cuMemsetD8Async(
            self.inner,
            data,
            self.len,
            stream.inner,
        ))
    }

    pub fn memset_d16(&mut self, data: u16) -> CudaResult<()> {
        if self.len % 2 != 0 {
            panic!("alignment failure in DevicePtr::memset_d16");
        }
        cuda_error(unsafe { sys::cuMemsetD16_v2(self.inner, data, self.len / 2) })
    }

    pub unsafe fn memset_d16_stream(
        &mut self,
        data: u16,
        stream: &mut Stream<'a>,
    ) -> CudaResult<()> {
        if self.len % 2 != 0 {
            panic!("alignment failure in DevicePtr::memset_d16_stream");
        }
        cuda_error(sys::cuMemsetD16Async(
            self.inner,
            data,
            self.len / 2,
            stream.inner,
        ))
    }

    pub fn memset_d32(&mut self, data: u32) -> CudaResult<()> {
        if self.len % 4 != 0 {
            panic!("alignment failure in DevicePtr::memset_d32");
        }
        cuda_error(unsafe { sys::cuMemsetD32_v2(self.inner, data, self.len / 4) })
    }

    pub unsafe fn memset_d32_stream(
        &mut self,
        data: u32,
        stream: &mut Stream<'a>,
    ) -> CudaResult<()> {
        if self.len % 4 != 0 {
            panic!("alignment failure in DevicePtr::memset_d32_stream");
        }
        cuda_error(sys::cuMemsetD32Async(
            self.inner,
            data,
            self.len / 4,
            stream.inner,
        ))
    }

    pub fn handle(&self) -> &Rc<Handle<'a>> {
        &self.handle
    }
}

pub struct DeviceBox<'a> {
    pub(crate) inner: DevicePtr<'a>,
}

impl<'a> DeviceBox<'a> {
    pub unsafe fn alloc(handle: &Rc<Handle<'a>>, size: u64) -> CudaResult<Self> {
        let mut out = 0u64;
        cuda_error(sys::cuMemAlloc_v2(&mut out as *mut u64, size))?;
        Ok(DeviceBox {
            inner: DevicePtr {
                handle: handle.clone(),
                inner: out,
                len: size,
                _p: PhantomData,
            },
        })
    }

    pub fn new(handle: &Rc<Handle<'a>>, input: &[u8]) -> CudaResult<Self> {
        let mut buf = unsafe { Self::alloc(handle, input.len() as u64) }?;
        buf.store(input)?;
        Ok(buf)
    }
}

impl<'a> Drop for DeviceBox<'a> {
    fn drop(&mut self) {
        if let Err(e) = cuda_error(unsafe { sys::cuMemFree_v2(self.inner.inner) }) {
            eprintln!("CUDA: failed freeing device buffer: {:?}", e);
        }
    }
}

impl<'a> AsRef<DevicePtr<'a>> for DeviceBox<'a> {
    fn as_ref(&self) -> &DevicePtr<'a> {
        &self.inner
    }
}

impl<'a> Deref for DeviceBox<'a> {
    type Target = DevicePtr<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> DerefMut for DeviceBox<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/*

CUresult cuMemAllocHost ( void** pp, size_t bytesize )
Allocates page-locked host memory.
CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags )
Allocates memory that will be automatically managed by the Unified Memory system.
CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes )
Allocates pitched device memory.

CUresult cuMemFreeHost ( void* p )
Frees page-locked host memory.

CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )
Get information on memory allocations.
CUresult cuMemGetInfo ( size_t* free, size_t* total )
Gets free and total memory.
CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags )
Allocates page-locked host memory.
CUresult cuMemHostGetDevicePointer ( CUdeviceptr* pdptr, void* p, unsigned int  Flags )
Passes back device pointer of mapped pinned memory.
CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p )
Passes back flags that were used for a pinned allocation.
CUresult cuMemHostRegister ( void* p, size_t bytesize, unsigned int  Flags )
Registers an existing host memory range for use by CUDA.
CUresult cuMemHostUnregister ( void* p )
Unregisters a memory range that was registered with cuMemHostRegister.
CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )
Copies memory.
CUresult cuMemcpy2D ( const CUDA_MEMCPY2D* pCopy )
Copies memory for 2D arrays.
CUresult cuMemcpy2DAsync ( const CUDA_MEMCPY2D* pCopy, CUstream hStream )
Copies memory for 2D arrays.
CUresult cuMemcpy2DUnaligned ( const CUDA_MEMCPY2D* pCopy )
Copies memory for 2D arrays.
CUresult cuMemcpy3D ( const CUDA_MEMCPY3D* pCopy )
Copies memory for 3D arrays.
CUresult cuMemcpy3DAsync ( const CUDA_MEMCPY3D* pCopy, CUstream hStream )
Copies memory for 3D arrays.
CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy )
Copies memory between contexts.
CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream )
Copies memory between contexts asynchronously.
CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream )
Copies memory asynchronously.
CUresult cuMemcpyAtoA ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount )
Copies memory from Array to Array.
CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )
Copies memory from Array to Device.
CUresult cuMemcpyAtoH ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount )
Copies memory from Array to Host.
CUresult cuMemcpyAtoHAsync ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream )
Copies memory from Array to Host.
CUresult cuMemcpyDtoA ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount )
Copies memory from Device to Array.
CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )
Copies memory from Device to Device.
CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )
Copies memory from Device to Device.
CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount )
Copies memory from Device to Host.
CUresult cuMemcpyDtoHAsync ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )
Copies memory from Device to Host.
CUresult cuMemcpyHtoA ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount )
Copies memory from Host to Array.
CUresult cuMemcpyHtoAAsync ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream )
Copies memory from Host to Array.
CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount )
Copies memory from Host to Device.
CUresult cuMemcpyHtoDAsync ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream )
Copies memory from Host to Device.
CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )
Copies device memory between two contexts.
CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )
Copies device memory between two contexts asynchronously.
CUresult cuMemsetD16 ( CUdeviceptr dstDevice, unsigned short us, size_t N )
Initializes device memory.
CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream )
Sets device memory.
CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height )
Initializes device memory.
CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream )
Sets device memory.
CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height )
Initializes device memory.
CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream )
Sets device memory.
CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height )
Initializes device memory.
CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream )
Sets device memory.
CUresult cuMemsetD32 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N )
Initializes device memory.
CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream )
Sets device memory.
CUresult cuMemsetD8 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N )
Initializes device memory.
CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream )
Sets device memory.
CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )
Creates a CUDA mipmapped array.
CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )
Destroys a CUDA mipmapped array.
CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level )
Gets a mipmap level of a CUDA mipmapped array.
CUresult cuMipmappedArrayGetSparseProperties ( CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap )
Returns the layout properties of a sparse CUDA mipmapped array.
*/
