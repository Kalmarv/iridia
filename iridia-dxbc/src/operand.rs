//! What both lifts ask of an instruction's decoded operands before they read them.

use dxbc_reader::decode::{Index, Operand, Register, Selection};

/// The component of its register a source gives destination lane `lane`, as the lifts' `read_lanes` pick it.
fn component(op: &Operand, lane: u8) -> Option<u8> {
    match op.selection {
        Selection::Swizzle(s) => s.get(usize::from(lane)).copied(),
        Selection::Scalar(c) => Some(c),
        Selection::Mask(_) if op.register == Register::Immediate => Some(lane),
        _ => None,
    }
}

fn same_index(a: &Index, b: &Index) -> bool {
    a.base == b.base
        && match (&a.relative, &b.relative) {
            (None, None) => true,
            // a relative index is its register's one selected component, read as the first lane
            (Some(a), Some(b)) => same_source(a, b, &[0]),
            _ => false,
        }
}

/// Whether two sources of one instruction are one value read twice in the destination lanes `lanes`: the same register
/// at the same indices (a relative index the same source too), the same component for every lane of `lanes`, the same
/// modifier and minimum precision, and for a literal the same words. An instruction reads its sources before it writes
/// its destination and a read changes nothing, so two such operands cannot differ. The components a swizzle names for
/// lanes the destination does not write are not read, and do not count: `r0.xyzx` and `r0.xyzw` are the same source of
/// a destination `.xyz`.
pub(crate) fn same_source(a: &Operand, b: &Operand, lanes: &[u8]) -> bool {
    a.register == b.register
        && a.modifier == b.modifier
        && a.precision == b.precision
        && a.immediate == b.immediate
        && a.indices.len() == b.indices.len()
        && a.indices.iter().zip(&b.indices).all(|(a, b)| same_index(a, b))
        && lanes.iter().all(|&lane| {
            let read = component(a, lane);
            read.is_some() && read == component(b, lane)
        })
}
