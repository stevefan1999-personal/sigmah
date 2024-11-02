#[inline(always)]
pub const unsafe fn pad_zeroes_slice_unchecked<const N: usize>(arr: &[u8]) -> [u8; N] {
    let mut arr_: [u8; N] = [0; N];
    let n = arr.len();
    core::ptr::copy_nonoverlapping(arr.as_ptr(), arr_.as_mut_ptr(), if N > n { n } else { N });
    arr_
}

#[cfg(test)]
#[inline(always)]
pub fn are_all_elements_equal<T: PartialEq>(elems: &[T]) -> bool {
    let [head, tail @ ..] = elems else {
        return false;
    };

    tail.iter().all(|x| x == head)
}

#[inline(always)]
pub const unsafe fn const_set_unchecked<T: Copy, const N: usize>(
    arr: &mut [T; N],
    idx: usize,
    elem: T,
) {
    *arr.as_mut_ptr().add(idx) = elem;
}

#[inline(always)]
pub const unsafe fn const_get_unchecked<T: Copy, const N: usize>(arr: &[T; N], idx: usize) -> T {
    *arr.as_ptr().add(idx)
}

pub(crate) mod simd;
