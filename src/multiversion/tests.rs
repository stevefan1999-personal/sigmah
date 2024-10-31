use crate::utils::{are_all_elements_equal, pad_zeroes_slice_unchecked};

#[test]
fn basic_sanity_test() {
    let mut haystack: [u8; 128] = core::array::from_fn(|i| i as u8);
    haystack[0] = 125;

    let haystack_after_first = unsafe { pad_zeroes_slice_unchecked::<128>(&haystack[1..]) };
    assert!(are_all_elements_equal(&[
        super::equal_then_find_first_position_naive(haystack[0], &haystack_after_first),
        super::simd::equal_then_find_first_position_simd::<u8>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u16>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u32>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u64>(haystack[0], &haystack_after_first,),
    ]));

    haystack[0] = 129;
    assert!(are_all_elements_equal(&[
        super::equal_then_find_first_position_naive(haystack[0], &haystack_after_first),
        super::simd::equal_then_find_first_position_simd::<u8>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u16>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u32>(haystack[0], &haystack_after_first,),
        super::simd::equal_then_find_first_position_simd::<u64>(haystack[0], &haystack_after_first,),
    ]));
}
