use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::*;

/// A slice into the device memory.
#[derive(Clone)]
pub struct DevicePtr<'a> {
    pub(crate) handle: Rc<Handle<'a>>,
    pub(crate) inner: u64,
    pub(crate) len: u64,
}

impl<'a> DevicePtr<'a> {
    pub fn as_raw(&self) -> u64 {
        self.inner
    }

    pub unsafe fn from_raw_parts(handle: Rc<Handle<'a>>, ptr: u64, len: u64) -> Self {
        Self {
            handle,
            inner: ptr,
            len,
        }
    }

    /// Synchronously copies data from `self` to `target`. Panics if length is not equal.
    pub fn copy_to<'b>(&self, target: &DevicePtr<'b>) -> CudaResult<()> {
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

    /// Asynchronously copies data from `self` to `target`. Panics if length is not equal.
    pub fn copy_to_stream<'b, 'c: 'b + 'a>(
        &self,
        target: &DevicePtr<'b>,
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
            cuda_error(unsafe {
                sys::cuMemcpyAsync(
                    target.inner,
                    self.inner,
                    self.len as sys::size_t,
                    stream.inner,
                )
            })
        } else {
            cuda_error(unsafe {
                sys::cuMemcpyPeerAsync(
                    target.inner,
                    target.handle.context.inner,
                    self.inner,
                    self.handle.context.inner,
                    self.len as sys::size_t,
                    stream.inner,
                )
            })
        }
    }

    // pub fn copy_to_async<'b>(&self, target: &DevicePtr<'b>) -> CudaResult<CudaFuture<'a, ()>>
    // where
    //     'a: 'b,
    // {
    //     let mut stream = self.handle.get_async_stream()?;
    //     unsafe { self.copy_to_stream(target, &mut stream) }?;
    //     Ok(CudaFuture::new(self.handle.clone(), stream))
    // }

    /// Synchronously copies data from `source` to `self`. Panics if length is not equal.
    pub fn copy_from<'b>(&self, source: &DevicePtr<'b>) -> CudaResult<()> {
        source.copy_to(self)
    }

    /// Asynchronously copies data from `source` to `self`. Panics if length is not equal.
    pub fn copy_from_stream<'b: 'a, 'c: 'a + 'b>(
        &self,
        source: &DevicePtr<'b>,
        stream: &mut Stream<'c>,
    ) -> CudaResult<()> {
        source.copy_to_stream(self, stream)
    }

    /// Gets a subslice of this slice from `[from:to]`
    pub fn subslice(&self, from: u64, to: u64) -> Self {
        if from > self.len || from > to || to > self.len {
            panic!("overflow in DevicePtr::subslice");
        }
        Self {
            handle: self.handle.clone(),
            inner: self.inner + from,
            len: to - from,
        }
    }

    /// Gets the length of this slice
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Check if the slice's length is 0
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Synchronously loads the data from this slice into a local buffer
    pub fn load(&self) -> CudaResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.len as usize);
        cuda_error(unsafe {
            sys::cuMemcpyDtoH_v2(buf.as_mut_ptr() as *mut _, self.inner, self.len as sys::size_t)
        })?;
        unsafe { buf.set_len(self.len as usize) };
        Ok(buf)
    }

    /// Asynchronously loads the data from this slice into a local buffer.
    /// The contents of the buffer are undefined until `stream.sync` is called.
    /// The output must not be dropped until the stream is synced.
    pub unsafe fn load_stream(&self, stream: &mut Stream<'a>) -> CudaResult<Vec<u8>> {
        let mut buf = Vec::with_capacity(self.len as usize);
        cuda_error(sys::cuMemcpyDtoHAsync_v2(
            buf.as_mut_ptr() as *mut _,
            self.inner,
            self.len as sys::size_t,
            stream.inner,
        ))?;
        buf.set_len(self.len as usize);
        Ok(buf)
    }

    /// Synchronously stores host data from `data` to `self`.
    pub fn store(&self, data: &[u8]) -> CudaResult<()> {
        if data.len() > self.len as usize {
            panic!("overflow in DevicePtr::store");
        } else if data.len() < self.len as usize {
            panic!("underflow in DevicePtr::store");
        }
        cuda_error(unsafe {
            sys::cuMemcpyHtoD_v2(self.inner, data.as_ptr() as *const _, self.len as sys::size_t)
        })?;
        Ok(())
    }

    /// Asynchronously stores host data from `data` to `self`.
    /// The `data` must not be dropped or mutated until `stream.sync` is called.
    pub unsafe fn store_stream(&self, data: &[u8], stream: &mut Stream<'a>) -> CudaResult<()> {
        if data.len() > self.len as usize {
            panic!("overflow in DevicePtr::store");
        } else if data.len() < self.len as usize {
            panic!("underflow in DevicePtr::store");
        }
        cuda_error(sys::cuMemcpyHtoDAsync_v2(
            self.inner,
            data.as_ptr() as *const _,
            self.len as sys::size_t,
            stream.inner,
        ))?;
        Ok(())
    }

    /// Synchronously set the contents of `self` to `data` repeated to fill length
    pub fn memset_d8(&self, data: u8) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuMemsetD8_v2(self.inner, data, self.len as sys::size_t) })
    }

    /// Asynchronously set the contents of `self` to `data` repeated to fill length
    pub fn memset_d8_stream(&self, data: u8, stream: &mut Stream<'a>) -> CudaResult<()> {
        cuda_error(unsafe { sys::cuMemsetD8Async(self.inner, data, self.len as sys::size_t, stream.inner) })
    }

    /// Synchronously set the contents of `self` to `data` repeated to fill length.
    /// Panics if [`Self::len`] is not a multiple of 2.
    pub fn memset_d16(&self, data: u16) -> CudaResult<()> {
        if self.len % 2 != 0 {
            panic!("alignment failure in DevicePtr::memset_d16");
        }
        cuda_error(unsafe { sys::cuMemsetD16_v2(self.inner, data, self.len as sys::size_t / 2) })
    }

    /// Asynchronously set the contents of `self` to `data` repeated to fill length.
    /// Panics if [`Self::len`] is not a multiple of 2.
    pub fn memset_d16_stream(&self, data: u16, stream: &mut Stream<'a>) -> CudaResult<()> {
        if self.len % 2 != 0 {
            panic!("alignment failure in DevicePtr::memset_d16_stream");
        }
        cuda_error(unsafe { sys::cuMemsetD16Async(self.inner, data, self.len as sys::size_t / 2, stream.inner) })
    }

    /// Synchronously set the contents of `self` to `data` repeated to fill length.
    /// Panics if [`Self::len`] is not a multiple of 4.
    pub fn memset_d32(&self, data: u32) -> CudaResult<()> {
        if self.len % 4 != 0 {
            panic!("alignment failure in DevicePtr::memset_d32");
        }
        cuda_error(unsafe { sys::cuMemsetD32_v2(self.inner, data, self.len as sys::size_t / 4) })
    }

    /// Asynchronously set the contents of `self` to `data` repeated to fill length.
    /// Panics if [`Self::len`] is not a multiple of 4.
    pub fn memset_d32_stream(&self, data: u32, stream: &mut Stream<'a>) -> CudaResult<()> {
        if self.len % 4 != 0 {
            panic!("alignment failure in DevicePtr::memset_d32_stream");
        }
        cuda_error(unsafe { sys::cuMemsetD32Async(self.inner, data, self.len as sys::size_t / 4, stream.inner) })
    }

    /// Gets a reference to the owning handle
    pub fn handle(&self) -> &Rc<Handle<'a>> {
        &self.handle
    }
}

/// An owned device-allocated buffer
pub struct DeviceBox<'a> {
    pub(crate) inner: DevicePtr<'a>,
}

impl<'a> DeviceBox<'a> {
    /// Allocate an uninitialized buffer of size `size` on the device
    pub fn alloc(handle: &Rc<Handle<'a>>, size: u64) -> CudaResult<Self> {
        let mut out = 0u64;
        cuda_error(unsafe { sys::cuMemAlloc_v2(&mut out as *mut u64, size as sys::size_t) })?;
        Ok(DeviceBox {
            inner: DevicePtr {
                handle: handle.clone(),
                inner: out,
                len: size,
            },
        })
    }

    /// Allocate a new initialized buffer on the device matching the size and content of `input`.
    pub fn new(handle: &Rc<Handle<'a>>, input: &[u8]) -> CudaResult<Self> {
        let buf = Self::alloc(handle, input.len() as u64)?;
        buf.store(input)?;
        Ok(buf)
    }

    /// Allocates a new uninitialized buffer on the device, then synchronously fills it with `input`.
    /// `input` must not be dropped or mutated until `stream.sync` is called.
    /// Does not allocate the memory asynchronously.
    pub unsafe fn new_stream(
        handle: &Rc<Handle<'a>>,
        input: &[u8],
        stream: &mut Stream<'a>,
    ) -> CudaResult<Self> {
        let buf = Self::alloc(handle, input.len() as u64)?;
        buf.store_stream(input, stream)?;
        Ok(buf)
    }

    /// Allocates a new initialized buffer on the device matching the size and content of `input`.
    /// Note that memory is directly copied, so [`T`] must be [`Sized`] should not contain any pointers, references, unsized types, or other non-FFI safe types.
    pub fn new_ffi<T>(handle: &Rc<Handle<'a>>, input: &[T]) -> CudaResult<Self> {
        let raw = unsafe {
            std::slice::from_raw_parts(
                input.as_ptr() as *const u8,
                input.len() * std::mem::size_of::<T>(),
            )
        };
        let buf = Self::alloc(handle, raw.len() as u64)?;
        buf.store(raw)?;
        Ok(buf)
    }

    /// Allocates a new uninitialized buffer on the device, then synchronously fills it with `input`.
    /// Note that memory is directly copied, so [`T`] must be [`Sized`] should not contain any pointers, references, unsized types, or other non-FFI safe types.
    /// `input` must not be dropped or mutated until `stream.sync` is called.
    /// Does not allocate the memory asynchronously.
    pub unsafe fn new_ffi_stream<T>(
        handle: &Rc<Handle<'a>>,
        input: &[T],
        stream: &mut Stream<'a>,
    ) -> CudaResult<Self> {
        let raw = std::slice::from_raw_parts(
            input.as_ptr() as *const u8,
            input.len() * std::mem::size_of::<T>(),
        );
        let buf = Self::alloc(handle, raw.len() as u64)?;
        buf.store_stream(raw, stream)?;
        Ok(buf)
    }

    /// Leaks the DeviceBox, similar to [`Box::leak`].
    pub fn leak(self) {
        std::mem::forget(self);
    }

    /// Constructs a [`DeviceBox`] from a device pointer.
    pub unsafe fn from_raw(raw: DevicePtr<'a>) -> Self {
        Self { inner: raw }
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
