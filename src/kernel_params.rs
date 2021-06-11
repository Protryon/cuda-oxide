use crate::{DeviceBox, DevicePtr};

/// Some data able to represent one or more kernel parameters
pub trait KernelParameters {
    fn params(&self, out: &mut Vec<Vec<u8>>);
}

impl KernelParameters for u8 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(vec![*self]);
    }
}

impl KernelParameters for u16 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for u32 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for u64 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for usize {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for i8 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for i16 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for i32 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for i64 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for f32 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

impl KernelParameters for f64 {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_le_bytes().to_vec());
    }
}

/// WARNING: this is unsafe!
impl<'a> KernelParameters for DevicePtr<'a> {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.inner.to_le_bytes().to_vec());
    }
}

/// WARNING: this is unsafe!
impl<'a, 'b> KernelParameters for &'b DeviceBox<'a> {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.inner.inner.to_le_bytes().to_vec());
    }
}

impl KernelParameters for &[u8] {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.to_vec());
    }
}

impl KernelParameters for Vec<u8> {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        out.push(self.clone());
    }
}

impl<T: KernelParameters + Default + Copy, const N: usize> KernelParameters for [T; N] {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        for x in self {
            x.params(out);
        }
    }
}

impl<T: KernelParameters> KernelParameters for Box<T> {
    fn params(&self, out: &mut Vec<Vec<u8>>) {
        (&**self).params(out);
    }
}

impl KernelParameters for () {
    fn params(&self, _out: &mut Vec<Vec<u8>>) {}
}

macro_rules! tuple_impls {
    ($($len:expr => ($($n:tt $name:ident)+))+) => {
        $(
            impl<$($name: KernelParameters),+> KernelParameters for ($($name,)+) {
                fn params(&self, out: &mut Vec<Vec<u8>>) {
                    $(
                        $name::params(&self.$n, out);
                    )+
                }
            }
        )+
    }
}

tuple_impls! {
    1 => (0 T0)
    2 => (0 T0 1 T1)
    3 => (0 T0 1 T1 2 T2)
    4 => (0 T0 1 T1 2 T2 3 T3)
    5 => (0 T0 1 T1 2 T2 3 T3 4 T4)
    6 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5)
    7 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6)
    8 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7)
    9 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8)
    10 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9)
    11 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10)
    12 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10 11 T11)
    13 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10 11 T11 12 T12)
    14 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10 11 T11 12 T12 13 T13)
    15 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10 11 T11 12 T12 13 T13 14 T14)
    16 => (0 T0 1 T1 2 T2 3 T3 4 T4 5 T5 6 T6 7 T7 8 T8 9 T9 10 T10 11 T11 12 T12 13 T13 14 T14 15 T15)
}
