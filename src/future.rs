use std::{
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use crate::*;

enum InteriorState<T> {
    Waiting,
    Some(T),
    Taken,
}

pub struct CudaFuture<'a, T> {
    interior: Arc<Mutex<InteriorState<CudaResult<T>>>>,
    active_stream: Option<Stream<'a>>,
    handle: Rc<Handle<'a>>,
}

impl<'a> Unpin for CudaFuture<'a, ()> {}

impl<'a> Future for CudaFuture<'a, ()> {
    type Output = CudaResult<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut interior = self.interior.lock().unwrap();
        match &mut *interior {
            InteriorState::Taken => panic!("over polled CudaFuture (do you need to fuse?)"),
            x @ InteriorState::Some(_) => {
                let mut state = InteriorState::Taken;
                std::mem::swap(x, &mut state);
                if let InteriorState::Some(x) = state {
                    Poll::Ready(x)
                } else {
                    unimplemented!()
                }
            }
            InteriorState::Waiting => {
                let waker = cx.waker().clone();
                drop(interior);
                let new_self = self.interior.clone();
                match self.active_stream.as_mut().unwrap().callback(move || {
                    let mut inner = new_self.lock().unwrap();
                    if matches!(&*inner, InteriorState::Waiting) {
                        *inner = InteriorState::Some(Ok(()));
                    }
                    waker.wake()
                }) {
                    Ok(()) => Poll::Pending,
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
        }
    }
}

impl<'a> CudaFuture<'a, ()> {
    pub(crate) fn new(handle: Rc<Handle<'a>>, stream: Stream<'a>) -> Self {
        CudaFuture {
            interior: Arc::new(Mutex::new(InteriorState::Waiting)),
            active_stream: Some(stream),
            handle,
        }
    }
}

impl<'a, T> Drop for CudaFuture<'a, T> {
    fn drop(&mut self) {
        self.handle
            .reset_async_stream(self.active_stream.take().unwrap());
    }
}
