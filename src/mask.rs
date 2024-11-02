use crate::concise_bitvec::ConciseBitArray;
use crate::utils::{const_get_unchecked, const_set_unchecked};
use derive_more::derive::{From, Into};

#[derive(Debug, Copy, Clone, From, Into)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct Mask<const N: usize>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    pub inner: ConciseBitArray<N>,
}

impl<const N: usize> Mask<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    pub const MAX: Self = Self {
        inner: ConciseBitArray::MAX,
    };
    pub const ZERO: Self = Self {
        inner: ConciseBitArray::ZERO,
    };

    #[inline(always)]
    pub const fn from_bit_array(arr: ConciseBitArray<N>) -> Self {
        Self { inner: arr }
    }

    #[inline(always)]
    pub const fn from_byte_array_or_panic(mask: [u8; N]) -> Self {
        Self::from_byte_slice_or_panic(&mask)
    }

    #[inline(always)]
    pub const fn from_byte_slice_or_panic(mask: &[u8; N]) -> Self {
        match Self::try_from_byte_slice_to_bitarr(mask) {
            Ok(x) => Self { inner: x },
            Err(e) => panic!("{}", e),
        }
    }

    #[inline(always)]
    pub const fn try_from_byte_array_to_bitarr(
        mask: [u8; N],
    ) -> Result<ConciseBitArray<N>, &'static str> {
        Self::try_from_byte_slice_to_bitarr(&mask)
    }

    #[inline(always)]
    pub const fn try_from_byte_slice_to_bitarr(
        mask: &[u8; N],
    ) -> Result<ConciseBitArray<N>, &'static str> {
        let mut pattern_bool: [bool; N] = [false; N];
        let mut i = 0;
        while i < N {
            unsafe {
                const_set_unchecked(
                    &mut pattern_bool,
                    i,
                    match const_get_unchecked(mask, i) {
                        b'x' => true,
                        b'?' => false,
                        _ => return Err("unknown character in mask"),
                    },
                )
            };
            i += 1;
        }
        Ok(ConciseBitArray::from_bool_slice(&pattern_bool))
    }

    #[inline(always)]
    pub const fn to_byte_array(&self) -> [u8; N] {
        // Unfortunately, self.to_bool_array().map(|x| if x { b'x' } else { b'?' }).collect() wouldn't work in const, at least for now
        let base = self.to_bool_array();
        let mut arr: [u8; N] = [b'?'; N];
        let mut i = 0;
        while i < N {
            unsafe {
                const_set_unchecked(
                    &mut arr,
                    i,
                    if const_get_unchecked(&base, i) {
                        b'x'
                    } else {
                        b'?'
                    },
                )
            };

            i += 1;
        }
        arr
    }

    #[inline(always)]
    pub const fn to_bool_array(&self) -> [bool; N] {
        self.inner.to_bool_array()
    }
}
