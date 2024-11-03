use crate::utils::{const_get_unchecked, const_set_unchecked};
use bitvec::prelude::*;
use derive_more::{Deref, From, Into};

#[derive(Debug, Copy, Clone, From, Into, Deref)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct ConciseBitArray<const N: usize>(pub BitArray<[u8; N.div_ceil(u8::BITS as usize)]>)
where
    [(); N.div_ceil(u8::BITS as usize)]:;

impl<const N: usize> ConciseBitArray<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    pub const MAX: Self = Self({
        let mut arr: BitArray<[u8; N.div_ceil(u8::BITS as usize)]> = BitArray::ZERO;
        let mut i = 0;
        while i < N {
            let (idx, bit_pos) = unsafe { Self::get_storage_idx_and_bit_pos_unchecked(i) };
            unsafe {
                let data = const_get_unchecked(&arr.data, idx);
                const_set_unchecked(&mut arr.data, idx, data | (1 << bit_pos));
            }
            i += 1;
        }
        arr
    });

    pub const ZERO: Self = Self(BitArray::ZERO);

    #[inline(always)]
    pub const fn get_storage_idx_and_bit_pos(i: usize) -> Option<(usize, usize)> {
        if i < N {
            Some(unsafe { Self::get_storage_idx_and_bit_pos_unchecked(i) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub const unsafe fn get_storage_idx_and_bit_pos_unchecked(i: usize) -> (usize, usize) {
        const BITS: usize = u8::BITS as usize;
        (i / BITS, i % BITS)
    }

    #[inline(always)]
    pub const fn is_exact(&self) -> bool {
        let mut i = 0;
        while i < N {
            if !unsafe { self.get_unchecked(i) } {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline(always)]
    pub const fn get(&self, i: usize) -> Option<bool> {
        if i < N {
            Some(unsafe { self.get_unchecked(i) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub const unsafe fn get_unchecked(&self, i: usize) -> bool {
        let (idx, bit_pos) = unsafe { Self::get_storage_idx_and_bit_pos_unchecked(i) };
        (const_get_unchecked(&self.0.data, idx) & (1 << bit_pos)) != 0
    }

    #[inline(always)]
    pub const fn get_or_false_if_idx_greater(&self, i: usize) -> bool {
        matches!(self.get(i), Some(false) | None)
    }
}

impl<const N: usize> ConciseBitArray<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn from_bool_array(pattern: [bool; N]) -> Self {
        Self::from_bool_slice(&pattern)
    }

    #[inline(always)]
    pub const fn from_bool_slice(pattern: &[bool; N]) -> Self {
        let mut arr = Self::ZERO;
        let mut i = 0;
        while i < pattern.len() {
            let (idx, bit_pos) = unsafe { Self::get_storage_idx_and_bit_pos_unchecked(i) };
            unsafe {
                let bit_storage = const_get_unchecked(&arr.0.data, idx);
                const_set_unchecked(
                    &mut arr.0.data,
                    idx,
                    bit_storage
                        | if const_get_unchecked(pattern, i) {
                            1 << bit_pos
                        } else {
                            0
                        },
                );
            }
            i += 1;
        }
        arr
    }

    #[inline(always)]
    pub const fn to_bool_array(&self) -> [bool; N] {
        let mut arr: [bool; N] = [false; N];
        let mut i = 0;
        while i < N {
            let (idx, bit_pos) = unsafe { Self::get_storage_idx_and_bit_pos_unchecked(i) };
            unsafe {
                const_set_unchecked(
                    &mut arr,
                    i,
                    (const_get_unchecked(&self.0.data, idx) & (1 << bit_pos)) != 0,
                );
            }
            i += 1;
        }
        arr
    }
}
