use std::ops::{Deref, DerefMut};

pub struct Dim3(pub (u32, u32, u32));

impl Deref for Dim3 {
    type Target = (u32, u32, u32);

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Dim3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<(u32, u32, u32)> for Dim3 {
    fn into(self) -> (u32, u32, u32) {
        self.0
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(inner: (u32, u32, u32)) -> Self {
        Self(inner)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self((x, y, 1))
    }
}

impl From<(u32,)> for Dim3 {
    fn from((x,): (u32,)) -> Self {
        Self((x, 1, 1))
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self((x, 1, 1))
    }
}
