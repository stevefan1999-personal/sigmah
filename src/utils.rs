#[cfg(feature = "simd")]
pub const unsafe fn pad_zeroes_slice_unchecked<const N: usize>(arr: &[u8]) -> [u8; N] {
    let mut arr_: [u8; N] = [0; N];
    let n = arr.len();
    core::ptr::copy_nonoverlapping(arr.as_ptr(), arr_.as_mut_ptr(), if N > n { n } else { N });
    arr_
}

pub trait Bits {
    const BITS: u32;
}

impl Bits for u8 {
    const BITS: u32 = u8::BITS;
}

impl Bits for u16 {
    const BITS: u32 = u16::BITS;
}

impl Bits for u32 {
    const BITS: u32 = u32::BITS;
}

impl Bits for u64 {
    const BITS: u32 = u64::BITS;
}

impl Bits for usize {
    const BITS: u32 = usize::BITS;
}
