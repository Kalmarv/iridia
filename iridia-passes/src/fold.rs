//! Constant folding: a value whose operands are all literals becomes the literal it computes, where every target
//! computes it the same way.
//!
//! Folded: integer and bool arithmetic, comparisons and bitwise operations (wrapping, as the operations state), the
//! float operations that touch only the sign bit (`fneg`, `fabs`) or test the bits (`isnan`), `select`, and the lane
//! moves (`compose`, swizzle, `splat`, a `bitcast` that keeps the lanes). Float arithmetic and float comparisons are
//! not: their results on a target depend on its denormal handling and precision, which a fold here would fix to one
//! answer. A division by zero whose rule is `Undef` is left for the target.

use alloc::vec::Vec;

use iridia_ir::{Binary, Cmp, Function, OnZero, Op, Scalar, Type, Unary, ValueId};

/// A lane's bits, masked to its scalar's width.
fn mask(s: Scalar) -> u64 {
    match s.bits() {
        64 => u64::MAX,
        b => (1u64 << b).wrapping_sub(1),
    }
}

/// A lane as a signed number, sign-extended from its width.
fn signed(s: Scalar, bits: u64) -> i64 {
    let shift = 64u32.saturating_sub(s.bits());
    (bits.wrapping_shl(shift).cast_signed()).wrapping_shr(shift)
}

/// The literal lanes of `v` and its type, where it is a literal.
fn literal(f: &Function, v: ValueId) -> Option<([u64; 4], Type)> {
    let d = f.value(v)?;
    match d.op {
        Op::Literal(bits) => Some((bits, d.ty)),
        _ => None,
    }
}

fn lanes(t: Type) -> usize {
    usize::from(t.lanes)
}

/// Each lane of the result, from `each` of the lane index; lanes past the type's are zero.
fn per_lane(t: Type, mut each: impl FnMut(usize) -> Option<u64>) -> Option<[u64; 4]> {
    let mut out = [0u64; 4];
    for (n, o) in out.iter_mut().enumerate().take(lanes(t)) {
        *o = each(n)? & mask(t.scalar);
    }
    Some(out)
}

fn compare(c: Cmp, ordering: core::cmp::Ordering) -> u64 {
    use core::cmp::Ordering::*;
    u64::from(match c {
        Cmp::Eq => ordering == Equal,
        Cmp::Ne => ordering != Equal,
        Cmp::Lt => ordering == Less,
        Cmp::Le => ordering != Greater,
        Cmp::Gt => ordering == Greater,
        Cmp::Ge => ordering != Less,
    })
}

fn sign_bit(s: Scalar) -> Option<u64> {
    s.is_float().then(|| 1u64 << s.bits().saturating_sub(1))
}

fn unary(op: Unary, a: [u64; 4], at: Type, t: Type) -> Option<[u64; 4]> {
    let lane = |n: usize| a.get(n).copied();
    match op {
        Unary::FNeg => {
            let sign = sign_bit(at.scalar)?;
            per_lane(t, |n| Some(lane(n)? ^ sign))
        }
        Unary::FAbs => {
            let sign = sign_bit(at.scalar)?;
            per_lane(t, |n| Some(lane(n)? & !sign))
        }
        Unary::IsNan => {
            let sign = sign_bit(at.scalar)?;
            let magnitude = sign.wrapping_sub(1);
            let exponent = match at.scalar {
                Scalar::F16 => 0x7c00,
                Scalar::F32 => 0x7f80_0000,
                _ => 0x7ff0_0000_0000_0000,
            };
            per_lane(t, |n| Some(u64::from(lane(n)? & magnitude > exponent)))
        }
        Unary::Not if at.scalar == Scalar::Bool => per_lane(t, |n| Some(lane(n)? ^ 1)),
        Unary::INeg if at.scalar.is_integer() => per_lane(t, |n| Some(lane(n)?.wrapping_neg())),
        Unary::Any => Some([u64::from(a.iter().take(lanes(at)).any(|&l| l != 0)), 0, 0, 0]),
        Unary::All => Some([u64::from(a.iter().take(lanes(at)).all(|&l| l != 0)), 0, 0, 0]),
        _ => None,
    }
}

fn binary(op: Binary, a: [u64; 4], b: [u64; 4], at: Type, t: Type) -> Option<[u64; 4]> {
    let s = at.scalar;
    if s.is_float() {
        return None;
    }
    let pair = |n: usize| Some((*a.get(n)?, *b.get(n)?));
    match op {
        Binary::SCmp(c) => per_lane(t, |n| {
            let (x, y) = pair(n)?;
            Some(compare(c, signed(s, x).cmp(&signed(s, y))))
        }),
        Binary::UCmp(c) => per_lane(t, |n| {
            let (x, y) = pair(n)?;
            Some(compare(c, x.cmp(&y)))
        }),
        Binary::IAdd => per_lane(t, |n| pair(n).map(|(x, y)| x.wrapping_add(y))),
        Binary::ISub => per_lane(t, |n| pair(n).map(|(x, y)| x.wrapping_sub(y))),
        Binary::IMul => per_lane(t, |n| pair(n).map(|(x, y)| x.wrapping_mul(y))),
        Binary::UDiv(zero) | Binary::URem(zero) => per_lane(t, |n| {
            let (x, y) = pair(n)?;
            match (x.checked_div(y), x.checked_rem(y), zero) {
                (Some(q), _, _) if matches!(op, Binary::UDiv(_)) => Some(q),
                (_, Some(r), _) if matches!(op, Binary::URem(_)) => Some(r),
                (_, _, OnZero::AllOnes) => Some(u64::MAX),
                (_, _, OnZero::ByOne) if matches!(op, Binary::UDiv(_)) => Some(x),
                (_, _, OnZero::ByOne) => Some(0),
                _ => None,
            }
        }),
        Binary::SMin => per_lane(t, |n| {
            pair(n).map(|(x, y)| if signed(s, x) <= signed(s, y) { x } else { y })
        }),
        Binary::SMax => per_lane(t, |n| {
            pair(n).map(|(x, y)| if signed(s, x) >= signed(s, y) { x } else { y })
        }),
        Binary::And => per_lane(t, |n| pair(n).map(|(x, y)| x & y)),
        Binary::Or => per_lane(t, |n| pair(n).map(|(x, y)| x | y)),
        Binary::Xor => per_lane(t, |n| pair(n).map(|(x, y)| x ^ y)),
        _ => None,
    }
}

/// The literal value `v` computes, where its operands are literals and its operation folds.
fn evaluate(f: &Function, v: ValueId) -> Option<[u64; 4]> {
    let d = f.value(v)?;
    let t = d.ty;
    match &d.op {
        Op::Unary(op, a) => {
            let (a, at) = literal(f, *a)?;
            unary(*op, a, at, t)
        }
        Op::Binary(op, a, b) => {
            let (a, at) = literal(f, *a)?;
            let (b, _) = literal(f, *b)?;
            binary(*op, a, b, at, t)
        }
        Op::Select { cond, accept, reject } => {
            let (c, ct) = literal(f, *cond)?;
            let (x, _) = literal(f, *accept)?;
            let (y, _) = literal(f, *reject)?;
            per_lane(t, |n| {
                let which = if ct.lanes == 1 { c.first()? } else { c.get(n)? };
                if *which != 0 {
                    x.get(n).copied()
                } else {
                    y.get(n).copied()
                }
            })
        }
        Op::Compose(parts) => {
            let mut all = Vec::new();
            for p in parts {
                let (bits, pt) = literal(f, *p)?;
                all.extend(bits.into_iter().take(lanes(pt)));
            }
            per_lane(t, |n| all.get(n).copied())
        }
        Op::Swizzle(a, picks) => {
            let (a, _) = literal(f, *a)?;
            per_lane(t, |n| a.get(usize::from(*picks.get(n)?)).copied())
        }
        Op::Splat(a) => {
            let (a, _) = literal(f, *a)?;
            per_lane(t, |_| a.first().copied())
        }
        Op::Bitcast(a) => {
            let (a, at) = literal(f, *a)?;
            // a bitcast between scalars of one width keeps each lane's bits; one that repacks lanes is left
            (at.lanes == t.lanes && at.scalar.bits() == t.scalar.bits()).then_some(a)
        }
        _ => None,
    }
}

/// Fold every value of `f` that computes a literal, until none is left; the number folded.
pub fn fold(f: &mut Function) -> usize {
    let mut folded = 0usize;
    // a value's operands may be defined after it in the list: each sweep folds what the last made possible
    loop {
        let mut changed = false;
        for n in 0..f.values.len() {
            let v = ValueId(u32::try_from(n).unwrap_or(u32::MAX));
            if matches!(f.value(v).map(|d| &d.op), Some(Op::Literal(_))) {
                continue;
            }
            if let Some(bits) = evaluate(f, v)
                && let Some(d) = f.values.get_mut(n)
            {
                d.op = Op::Literal(bits);
                folded = folded.saturating_add(1);
                changed = true;
            }
        }
        if !changed {
            return folded;
        }
    }
}
