//! Sets of lanes: which invocations of a run are active. A pixel run is a quad or a few invocations in lockstep, its
//! set an integer's bits; a thread group of up to 1024 invocations needs [`Wide`].

/// A set of the lanes of a run.
pub trait Mask: Copy + PartialEq + std::fmt::Debug {
    /// The empty set.
    fn none() -> Self;
    /// Lanes `0..n`.
    fn first(n: usize) -> Self;
    /// Whether `lane` is in the set.
    fn has(self, lane: usize) -> bool;
    /// Add `lane`; a lane past the width of the set changes nothing.
    fn insert(&mut self, lane: usize);
    /// Remove `lane`; a lane past the width of the set changes nothing.
    fn remove(&mut self, lane: usize);
    /// The lanes in both sets.
    fn and(self, other: Self) -> Self;
    /// The lanes in either set.
    fn or(self, other: Self) -> Self;
    /// The lanes of `self` that are not in `other`.
    fn minus(self, other: Self) -> Self;
    /// Whether the set has no lane.
    fn is_empty(self) -> bool;
}

/// The shift count of a lane: a lane past `u32` is past every mask.
fn shift(lane: usize) -> u32 {
    u32::try_from(lane).unwrap_or(u32::MAX)
}

/// A set of up to as many lanes as the integer has bits. A lane past the width is in no set, and inserting it or removing
/// it changes none.
macro_rules! bits {
    ($($t:ty),*) => {$(
        impl Mask for $t {
            fn none() -> Self {
                0
            }
            fn first(n: usize) -> Self {
                (1 as $t).checked_shl(shift(n)).map_or(<$t>::MAX, |one| one.wrapping_sub(1))
            }
            fn has(self, lane: usize) -> bool {
                self.checked_shr(shift(lane)).is_some_and(|rest| rest & 1 != 0)
            }
            fn insert(&mut self, lane: usize) {
                if let Some(bit) = (1 as $t).checked_shl(shift(lane)) {
                    *self |= bit;
                }
            }
            fn remove(&mut self, lane: usize) {
                if let Some(bit) = (1 as $t).checked_shl(shift(lane)) {
                    *self &= !bit;
                }
            }
            fn and(self, other: Self) -> Self {
                self & other
            }
            fn or(self, other: Self) -> Self {
                self | other
            }
            fn minus(self, other: Self) -> Self {
                self & !other
            }
            fn is_empty(self) -> bool {
                self == 0
            }
        }
    )*};
}
bits!(u8, u16, u32);

/// Up to 1024 lanes. A lane past them is in no set, and inserting it or removing it changes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wide([u64; 16]);

impl Wide {
    /// The word of `lane` and the bit in it.
    fn place(lane: usize) -> (usize, u64) {
        (lane / 64, 1u64.checked_shl(shift(lane % 64)).unwrap_or(0))
    }
    /// Each word of `self` combined with the word of `other` at the same position.
    fn zip(self, other: Self, combine: impl Fn(u64, u64) -> u64) -> Self {
        let mut words = self.0;
        for (word, other) in words.iter_mut().zip(other.0) {
            *word = combine(*word, other);
        }
        Wide(words)
    }
}

impl Mask for Wide {
    fn none() -> Self {
        Wide([0; 16])
    }
    fn first(n: usize) -> Self {
        let mut w = [0u64; 16];
        for (i, word) in w.iter_mut().enumerate() {
            // the lanes of this word that are among the first n
            let taken = n.saturating_sub(i.saturating_mul(64));
            *word = 1u64
                .checked_shl(shift(taken))
                .map_or(u64::MAX, |one| one.wrapping_sub(1));
        }
        Wide(w)
    }
    fn has(self, lane: usize) -> bool {
        let (word, bit) = Self::place(lane);
        self.0.get(word).is_some_and(|w| w & bit != 0)
    }
    fn insert(&mut self, lane: usize) {
        let (word, bit) = Self::place(lane);
        if let Some(w) = self.0.get_mut(word) {
            *w |= bit;
        }
    }
    fn remove(&mut self, lane: usize) {
        let (word, bit) = Self::place(lane);
        if let Some(w) = self.0.get_mut(word) {
            *w &= !bit;
        }
    }
    fn and(self, other: Self) -> Self {
        self.zip(other, |a, b| a & b)
    }
    fn or(self, other: Self) -> Self {
        self.zip(other, |a, b| a | b)
    }
    fn minus(self, other: Self) -> Self {
        self.zip(other, |a, b| a & !b)
    }
    fn is_empty(self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
}
