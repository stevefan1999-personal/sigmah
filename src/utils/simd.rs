use core::{
    fmt::Debug,
    ops::{BitOr, Shl},
};

// Unfortunately, where LaneCount<{ Self::LANES }>: SupportedLaneCount does not work, we do the best we can
pub trait SimdBits:
    Shl<usize, Output = Self> + BitOr<Output = Self> + Sized + Sync + Copy + Debug
{
    const LANES: usize;
    const ONE: Self;
    const ZERO: Self;

    fn to_u64(self) -> u64;
}

impl SimdBits for u8 {
    const LANES: usize = u8::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;

    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u16 {
    const LANES: usize = u16::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u32 {
    const LANES: usize = u32::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u64 {
    const LANES: usize = u64::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for usize {
    const LANES: usize = usize::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}
