//! Runs a program of iridia's shader IR ([`iridia_ir::Program`]) on the CPU, one invocation or a pixel quad, by the
//! IR's own definition of each operation: the reference the lowerings are checked against.
//!
//! The arithmetic is the reference `iridia-exec` uses: IEEE 754 `f32` for the basic operations, the transcendental
//! functions (`exp2`, `log2`, `sin`, `cos`, `sqrt`, `inverseSqrt`, `pow`) in `f64` rounded once, a dot product summed
//! left to right with each step rounded. Denormals are kept, or flushed to zero (sign kept) on the inputs and result
//! of each float operation but the sign operations and the moves, by the program's float environment or, where it
//! leaves them open, by the run's [`Bound::denormals`]; a double never is. This subset runs 32-bit scalars and
//! vectors and 64-bit ones of up to two lanes ([`runs`]), both control phases (every construct of a `Regions` body,
//! the blocks of a `Cfg` one), constant buffer reads, pipeline state, samples (answered by the caller), kill and
//! demotion; it refuses the rest by name.
//!
//! [`run`] runs one invocation, which has no screen derivatives: it refuses a derivative and leaves an implicit-LOD
//! sample's level to the sampler. [`run_quad`] runs a pixel program's 2 x 2 quad, whose derivatives are differences
//! between its lanes.
#![deny(missing_docs)]
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )
)]

mod wide;

use iridia_ir::{
    Axis, Binary, Block, Body, Cmp, Denormals, End, Fineness, FloatToInt, Function, LabelId, Level,
    MinMaxNan, MulZero, Node, OnZero, Op, OutOfBounds, Program, Region, Rounding, SaturateNan, Scalar,
    Terminator, Ternary, Type, Unary, ValueId,
    interface::{ResourceId, ResourceKind},
};

/// A value's lanes as 32-bit words (a bool is 0 or 1).
pub type Lanes = [u32; 4];

/// How a sample picks its level of detail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lod {
    /// From the derivatives, which a single invocation does not have: the sampler decides.
    Auto,
    /// From the derivatives, which a single invocation does not have, plus a bias.
    Bias(f32),
    /// An explicit level.
    Exact(f32),
    /// From derivatives of the coordinate along x and y (the coordinate's lanes of each, zero past them): the
    /// program's own, or for an implicit level in a quad the quad's.
    Grad([f32; 3], [f32; 3]),
    /// An implicit level in a quad: from the quad's derivatives of the coordinate, as `Grad`, plus a bias.
    GradBias([f32; 3], [f32; 3], f32),
}

/// A sample request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// The texture.
    pub texture: ResourceId,
    /// The sampler.
    pub sampler: ResourceId,
    /// The coordinate's lanes (two for 2D, three for cube and volume).
    pub coord: [f32; 3],
    /// How many lanes `coord` has.
    pub lanes: u8,
    /// The level of detail.
    pub lod: Lod,
}

/// What a run gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    /// Each output slot's lanes in varying order, each as the program stored it last (zero for a lane it never
    /// stored); `None` for a slot it did not store at all.
    pub outputs: Vec<Option<Lanes>>,
    /// Whether the invocation was killed or demoted (its outputs then are those stored before).
    pub killed: bool,
}

/// The resources and inputs of a run.
#[allow(missing_debug_implementations)] // the sampler is a closure
pub struct Bound<'a> {
    /// Each interface input slot's lanes.
    pub inputs: &'a [Lanes],
    /// Each constant buffer's rows.
    pub buffers: &'a [(ResourceId, &'a [Lanes])],
    /// Each pipeline state resource's value (a bool is 0 or 1).
    pub state: &'a [(ResourceId, Lanes)],
    /// Answers each sample with the texel's four lanes.
    pub sampler: &'a mut dyn FnMut(Sample) -> Result<[f32; 4], String>,
    /// The `f32` denormal rule of a program whose float environment leaves it open ([`Denormals::Any`]): kept for
    /// `Any` and `Preserve`, flushed for `Flush`. A program that states a rule runs by it.
    pub denormals: Denormals,
}

/// The most loop iterations a run takes before it is refused.
pub const STEP_LIMIT: u64 = 1 << 24;

/// The most derivatives and implicit-LOD samples a quad's lanes meet before the quad run is refused.
pub const MEETING_LIMIT: usize = 1 << 12;

/// Why a lane's run stopped before its end.
enum Stop {
    /// It failed.
    Fail(String),
    /// A lane of a quad reached an operation whose neighbours' operands it does not know yet.
    Meet,
}

impl From<String> for Stop {
    fn from(e: String) -> Self {
        Stop::Fail(e)
    }
}

impl From<&str> for Stop {
    fn from(e: &str) -> Self {
        Stop::Fail(e.into())
    }
}

type Step<T> = Result<T, Stop>;

/// Where a lane met an operation that reads its quad: the operation, how many loop iterations or block executions the
/// lane had run, and how many times it had run the operation before. Lanes that meet at different places diverge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    value: ValueId,
    steps: u64,
    before: u32,
}

/// What a lane of a quad knows of the meetings before: each lane's operand at each meeting, the lanes in quad order.
struct Quad<'q> {
    /// This lane: 0 and 1 the top row, left to right, 2 and 3 the bottom.
    lane: usize,
    known: &'q [[Lanes; 4]],
    /// How many meetings this run of the lane has passed.
    met: usize,
    /// How many times this run has met each value's operation.
    times: Vec<u32>,
    /// The meeting the run stopped at, and its operand there.
    waiting: Option<(Place, Lanes)>,
}

/// How a region ended.
enum Exit {
    Yield(Vec<ValueId>),
    Br(LabelId, Vec<ValueId>),
    Continue(LabelId, Vec<ValueId>),
    Return,
    Kill,
}

struct Machine<'a, 'b> {
    program: &'a Program,
    f: &'a Function,
    values: Vec<Option<Lanes>>,
    /// Each variable's elements (one for a variable that is not an array), zero where it starts undefined.
    vars: Vec<Vec<Lanes>>,
    bound: &'a mut Bound<'b>,
    outputs: Vec<Option<Lanes>>,
    steps: u64,
    /// Whether `f32` denormals are flushed.
    flush: bool,
    /// The lane's quad, where the run is one.
    quad: Option<Quad<'a>>,
    /// Whether the invocation was demoted: a helper from there, whose output stores have no effect.
    demoted: bool,
}

/// The two lanes of a quad whose difference is lane `lane`'s derivative: (from, to). A coarse derivative (and
/// `Any`'s, as `iridia-exec` takes it) is the top row's or the left column's for every lane, a fine one the lane's own
/// row's or column's.
fn pair(axis: Axis, fineness: Fineness, lane: usize) -> (usize, usize) {
    let fine = fineness == Fineness::Fine;
    match axis {
        Axis::X if fine && lane >= 2 => (2, 3),
        Axis::X => (0, 1),
        Axis::Y if fine && lane % 2 == 1 => (1, 3),
        Axis::Y => (0, 2),
    }
}

/// An `f32` lane with a denormal made a zero of its sign.
fn flushed(w: u32) -> u32 {
    if w & 0x7f80_0000 == 0 { w & 0x8000_0000 } else { w }
}

/// The lanes a mask names, in ascending order.
fn masked(mask: u8) -> impl Iterator<Item = usize> {
    (0..4).filter(move |n| mask >> n & 1 != 0)
}

fn narrow(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // the rounding is the operation
    let r = v as f32;
    r
}
/// An operation's name, for a refusal.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::LoadInputAt(..) => "a read of one vertex's input",
        Op::LoadOutputAt(..) => "a read of one control point's output",
        Op::LoadPatch(..) => "a patch constant read",
        Op::Builtin(_) => "a stage builtin",
        Op::Gather { .. } => "a gather",
        Op::LoadTexel { .. } => "a texel load",
        Op::Query(..) => "a resource query",
        Op::Atomic { .. } => "an atomic",
        Op::Counter(..) => "a buffer counter",
        Op::ReportHit { .. } => "a reported hit",
        Op::Wave(_) => "a wave operation",
        _ => "the operation",
    }
}

/// `count` bits of `a` from bit `offset` (both modulo 32), cut at bit 31, extended by sign or zero.
fn bit_extract(a: u32, offset: u32, count: u32, signed: bool) -> u32 {
    let (offset, count) = (offset % 32, count % 32);
    if count == 0 {
        return 0;
    }
    // the field's width, 1 to 31 bits: moved to the top, then back down with the sign or zeros
    let width = count.min(32u32.wrapping_sub(offset));
    let top = (a >> offset).wrapping_shl(32u32.wrapping_sub(width));
    let down = 32u32.wrapping_sub(width);
    if signed {
        top.cast_signed().wrapping_shr(down).cast_unsigned()
    } else {
        top.wrapping_shr(down)
    }
}

/// `count` low bits of `insert` placed into `base` from bit `offset` (both modulo 32), cut at bit 31.
fn bit_insert(base: u32, insert: u32, offset: u32, count: u32) -> u32 {
    let (offset, count) = (offset % 32, count % 32);
    let field = 1u32.wrapping_shl(count).wrapping_sub(1).wrapping_shl(offset);
    base & !field | insert.wrapping_shl(offset) & field
}

/// The binary16 nearest an `f32`, ties to even, in the low 16 bits: a NaN is the quiet NaN 0x7e00, a value past the
/// range an infinity, and a value below the smallest normal a denormal or a zero. IEEE 754 leaves the sign and payload
/// of an operation's NaN open (an operation on two NaNs gives either), so the half keeps neither: what the conversion
/// gives depends on the value alone, and a pass that reorders an operation's operands leaves it.
fn f32_to_f16(w: u32) -> u32 {
    let sign = w >> 16 & 0x8000;
    let exp = w >> 23 & 0xff;
    let man = w & 0x7f_ffff;
    if exp == 0xff {
        return if man == 0 { sign | 0x7c00 } else { 0x7e00 };
    }
    // binary16's biased exponent is f32's less 112
    if exp >= 143 {
        return sign | 0x7c00;
    }
    if exp < 102 {
        return sign;
    }
    // a normal half keeps the exponent and drops 13 bits; below one, the implicit bit joins a significand shifted
    // further, a denormal
    let (sig, shift, high) = if exp > 112 {
        (man, 13u32, exp.wrapping_sub(112) << 10)
    } else {
        (man | 0x80_0000, 126u32.wrapping_sub(exp), 0)
    };
    let half = sig >> shift;
    let rest = sig & 1u32.wrapping_shl(shift).wrapping_sub(1);
    let mid = 1u32.wrapping_shl(shift.wrapping_sub(1));
    let up = u32::from(rest > mid || (rest == mid && half & 1 == 1));
    // a carry out of the significand moves into the exponent, to an infinity at the top
    sign | (high | half).wrapping_add(up)
}

/// The `f32` of the binary16 in the low 16 bits.
fn f16_to_f32(w: u32) -> u32 {
    let sign = (w & 0x8000) << 16;
    let exp = w >> 10 & 0x1f;
    let man = w & 0x3ff;
    match exp {
        0 if man == 0 => sign,
        0 => {
            // a denormal, normalised: its leading bit becomes the implicit one
            let lead = man.leading_zeros().wrapping_sub(21);
            let e = 113u32.wrapping_sub(lead);
            sign | e << 23 | (man.wrapping_shl(lead) & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | man << 13,
        _ => sign | exp.wrapping_add(112) << 23 | man << 13,
    }
}

fn fb(x: u32) -> f32 {
    f32::from_bits(x)
}

/// Each lane of `n` through `op`.
fn each(n: u8, a: &Lanes, op: impl Fn(u32) -> u32) -> Lanes {
    let mut r = [0; 4];
    for (o, x) in r.iter_mut().zip(a).take(usize::from(n)) {
        *o = op(*x);
    }
    r
}
fn each2(n: u8, a: &Lanes, b: &Lanes, op: impl Fn(u32, u32) -> u32) -> Lanes {
    let mut r = [0; 4];
    for ((o, x), y) in r.iter_mut().zip(a).zip(b).take(usize::from(n)) {
        *o = op(*x, *y);
    }
    r
}
fn float(n: u8, a: &Lanes, op: impl Fn(f32) -> f32) -> Lanes {
    each(n, a, |x| op(fb(x)).to_bits())
}
fn float2(n: u8, a: &Lanes, b: &Lanes, op: impl Fn(f32, f32) -> f32) -> Lanes {
    each2(n, a, b, |x, y| op(fb(x), fb(y)).to_bits())
}
fn compare(c: Cmp) -> impl Fn(core::cmp::Ordering) -> bool {
    use core::cmp::Ordering::*;
    move |o| match c {
        Cmp::Eq => o == Equal,
        Cmp::Ne => o != Equal,
        Cmp::Lt => o == Less,
        Cmp::Le => o != Greater,
        Cmp::Gt => o == Greater,
        Cmp::Ge => o != Less,
    }
}
fn is_nan(x: f32) -> bool {
    x.to_bits() & 0x7fff_ffff > 0x7f80_0000
}
fn min_max(a: f32, b: f32, nan: MinMaxNan, max: bool) -> f32 {
    match (nan, is_nan(a), is_nan(b)) {
        (MinMaxNan::Propagate, true, _) | (MinMaxNan::Propagate, _, true) => f32::NAN,
        _ if max => a.max(b),
        _ => a.min(b),
    }
}

/// Whether values of `t` are run: a bool, a 32-bit scalar or vector, or a 64-bit one of at most two lanes (two words
/// a lane, the low first). Values of 8 or 16 bits are not.
pub fn runs(t: Type) -> bool {
    match t.scalar.bits() {
        32 => true,
        64 => t.lanes <= 2,
        _ => t.scalar == Scalar::Bool,
    }
}

/// The words of a literal of type `t`: each 64-bit lane in two words, the low first; each other lane in its word.
pub fn literal_words(bits: &[u64; 4], t: Type) -> Lanes {
    let mut out = [0u32; 4];
    if t.scalar.bits() == 64 {
        for ([lo, hi], b) in out.as_chunks_mut::<2>().0.iter_mut().zip(bits) {
            *lo = (b & 0xffff_ffff) as u32;
            *hi = (b >> 32) as u32;
        }
    } else {
        for (o, b) in out.iter_mut().zip(bits) {
            *o = (b & 0xffff_ffff) as u32;
        }
    }
    out
}

/// What an operation reads: its operands' lanes and types.
pub trait Operands {
    /// The lanes of value `v`.
    fn get(&self, v: ValueId) -> Result<Lanes, String>;
    /// The type of value `v`.
    fn ty(&self, v: ValueId) -> Result<Type, String>;
    /// An operand of an arithmetic operation: its lanes, flushed where `flush` holds and it is an `f32`.
    fn arg(&self, v: ValueId, flush: bool) -> Result<Lanes, String> {
        let x = self.get(v)?;
        Ok(if flush && self.ty(v)?.scalar == Scalar::F32 {
            x.map(flushed)
        } else {
            x
        })
    }
}

/// The lanes of a pure operation of type `t` (one that reads only its operands: the arithmetic, `select`, the lane
/// moves, `dot`, the conversions and bitcasts), by the IR's definition; `None` for an operation that is not pure (a
/// literal, a read, a derivative). Where `flush` holds, the `f32` operands and result of each float operation (the sign
/// operations among them, as a Direct3D `mov` with a source modifier) are flushed to zero (sign kept); a move, select
/// or bitcast keeps its words. A 64-bit value (at most two lanes) is two words a
/// lane, the low first; values of 8 or 16 bits are refused.
pub fn eval(op: &Op, t: Type, ops: &dyn Operands, flush: bool) -> Result<Option<Lanes>, String> {
    // the operand whose type tells a 64-bit operation from a 32-bit one (the operands of each are of one width, but
    // for a conversion's and a bitcast's result)
    let first = match op {
        Op::Unary(_, a)
        | Op::Binary(_, a, _)
        | Op::Ternary(_, a, _, _)
        | Op::Select { accept: a, .. }
        | Op::Swizzle(a, _)
        | Op::Splat(a)
        | Op::Dot(a, _)
        | Op::Convert(a, _)
        | Op::Bitcast(a) => Some(*a),
        Op::Compose(parts) => parts.first().copied(),
        _ => None,
    };
    if wide::wide(t) || first.map(|a| ops.ty(a)).transpose()?.is_some_and(wide::wide) {
        return wide::eval(op, t, ops, flush);
    }
    if !runs(t) {
        return Err(format!("{t} values are not run yet"));
    }
    let n = t.lanes;
    let out = match op {
        Op::Unary(u, a) => {
            let x = ops.arg(*a, flush)?;
            let at = ops.ty(*a)?;
            let m = at.lanes;
            match u {
                Unary::FNeg => each(m, &x, |w| w ^ 0x8000_0000),
                Unary::FAbs => each(m, &x, |w| w & 0x7fff_ffff),
                Unary::FFract => float(m, &x, |f| f - f.floor()),
                Unary::FRound(Rounding::Floor) => float(m, &x, f32::floor),
                Unary::FRound(Rounding::Ceil) => float(m, &x, f32::ceil),
                Unary::FRound(Rounding::Trunc) => float(m, &x, f32::trunc),
                Unary::FRound(Rounding::NearestEven) => float(m, &x, f32::round_ties_even),
                // Rust's `round` is to the nearest, halves away from zero
                Unary::FRound(Rounding::NearestAway) => float(m, &x, f32::round),
                Unary::FSqrt => float(m, &x, |f| narrow(f64::from(f).sqrt())),
                Unary::FInverseSqrt => float(m, &x, |f| narrow(1.0 / f64::from(f).sqrt())),
                Unary::FExp2 => float(m, &x, |f| narrow(f64::from(f).exp2())),
                Unary::FLog2 => float(m, &x, |f| narrow(f64::from(f).log2())),
                Unary::FSin => float(m, &x, |f| narrow(f64::from(f).sin())),
                Unary::FCos => float(m, &x, |f| narrow(f64::from(f).cos())),
                Unary::FTan => float(m, &x, |f| narrow(f64::from(f).tan())),
                Unary::FAsin => float(m, &x, |f| narrow(f64::from(f).asin())),
                Unary::FAcos => float(m, &x, |f| narrow(f64::from(f).acos())),
                Unary::FAtan => float(m, &x, |f| narrow(f64::from(f).atan())),
                Unary::FSinh => float(m, &x, |f| narrow(f64::from(f).sinh())),
                Unary::FCosh => float(m, &x, |f| narrow(f64::from(f).cosh())),
                Unary::FTanh => float(m, &x, |f| narrow(f64::from(f).tanh())),
                // a zero gives itself, NaN 0
                Unary::FSign => float(m, &x, |f| {
                    if is_nan(f) {
                        0.0
                    } else if f == 0.0 {
                        f
                    } else {
                        f.signum()
                    }
                }),
                Unary::Saturate(SaturateNan::Zero | SaturateNan::Any) => {
                    float(m, &x, |f| if is_nan(f) || f <= 0.0 { 0.0 } else { f.min(1.0) })
                }
                // NaN gives 0 under either rule (`Any` allows it)
                Unary::ClampRange { lo, hi, .. } => {
                    let (lo, hi) = (fb(*lo), fb(*hi));
                    float(m, &x, |f| if is_nan(f) { 0.0 } else { f.clamp(lo, hi) })
                }
                Unary::Quantize { frac_bits, rounding } => {
                    let scale = f32::from_bits(127u32.saturating_add(u32::from(*frac_bits)) << 23);
                    // from 2^(23 - frac_bits) on every f32 is on the grid already
                    let grid = f32::from_bits(150u32.saturating_sub(u32::from(*frac_bits)) << 23);
                    let round = match rounding {
                        Rounding::Floor => f32::floor,
                        Rounding::Ceil => f32::ceil,
                        Rounding::Trunc => f32::trunc,
                        Rounding::NearestEven => f32::round_ties_even,
                        Rounding::NearestAway => f32::round,
                    };
                    float(m, &x, |f| {
                        if f.abs() >= grid {
                            f
                        } else {
                            round(f * scale) / scale
                        }
                    })
                }
                // a derivative reads the quad, which is the run's
                Unary::Derivative(..) => return Ok(None),
                Unary::Not if at.scalar == Scalar::Bool => each(m, &x, |w| u32::from(w == 0)),
                Unary::Not => each(m, &x, |w| !w),
                Unary::INeg => each(m, &x, u32::wrapping_neg),
                Unary::Any => [u32::from(x.iter().take(usize::from(m)).any(|&w| w != 0)), 0, 0, 0],
                Unary::All => [u32::from(x.iter().take(usize::from(m)).all(|&w| w != 0)), 0, 0, 0],
                Unary::IsNan => each(m, &x, |w| u32::from(is_nan(fb(w)))),
                Unary::IsInf => each(m, &x, |w| u32::from(w & 0x7fff_ffff == 0x7f80_0000)),
                Unary::CountOnes => each(m, &x, u32::count_ones),
                Unary::BitReverse => each(m, &x, u32::reverse_bits),
                Unary::FirstBitHigh => each(m, &x, |w| if w == 0 { u32::MAX } else { w.leading_zeros() }),
                Unary::FirstBitHighSigned => each(m, &x, |w| {
                    let w = if w.cast_signed() < 0 { !w } else { w };
                    if w == 0 { u32::MAX } else { w.leading_zeros() }
                }),
                Unary::FirstBitLow => each(m, &x, |w| if w == 0 { u32::MAX } else { w.trailing_zeros() }),
                Unary::F32ToF16 => each(m, &x, f32_to_f16),
                Unary::F16ToF32 => each(m, &x, f16_to_f32),
            }
        }
        Op::Binary(b, l, r) => {
            let (x, y) = (ops.arg(*l, flush)?, ops.arg(*r, flush)?);
            let at = ops.ty(*l)?;
            let m = at.lanes;
            let signed = at.scalar.is_signed();
            let order = |p: u32, q: u32| {
                if at.scalar.is_float() {
                    fb(p).partial_cmp(&fb(q))
                } else if signed {
                    Some(p.cast_signed().cmp(&q.cast_signed()))
                } else {
                    Some(p.cmp(&q))
                }
            };
            match b {
                Binary::FAdd => float2(m, &x, &y, |p, q| p + q),
                Binary::FSub => float2(m, &x, &y, |p, q| p - q),
                Binary::FMul(MulZero::Ieee) => float2(m, &x, &y, |p, q| p * q),
                Binary::FMul(MulZero::ZeroWins) => {
                    float2(m, &x, &y, |p, q| if p == 0.0 || q == 0.0 { 0.0 } else { p * q })
                }
                Binary::FDiv => float2(m, &x, &y, |p, q| p / q),
                // C's fmod, exact
                Binary::FRem => float2(m, &x, &y, |p, q| p % q),
                Binary::FMin(nan) => float2(m, &x, &y, |p, q| min_max(p, q, *nan, false)),
                Binary::FMax(nan) => float2(m, &x, &y, |p, q| min_max(p, q, *nan, true)),
                Binary::FPow => float2(m, &x, &y, |p, q| narrow(f64::from(p).powf(f64::from(q)))),
                Binary::FCmp(c) | Binary::SCmp(c) | Binary::UCmp(c) => {
                    let c = compare(*c);
                    let ne = matches!(
                        b,
                        Binary::FCmp(Cmp::Ne) | Binary::SCmp(Cmp::Ne) | Binary::UCmp(Cmp::Ne)
                    );
                    each2(m, &x, &y, |p, q| u32::from(order(p, q).map_or(ne, &c)))
                }
                Binary::IAdd => each2(m, &x, &y, u32::wrapping_add),
                Binary::ISub => each2(m, &x, &y, u32::wrapping_sub),
                Binary::IMul => each2(m, &x, &y, u32::wrapping_mul),
                Binary::UMulHigh => each2(m, &x, &y, |p, q| {
                    let wide = u64::from(p).wrapping_mul(u64::from(q));
                    u32::try_from(wide >> 32).unwrap_or(0)
                }),
                Binary::SMulHigh => each2(m, &x, &y, |p, q| {
                    let wide = i64::from(p.cast_signed()).wrapping_mul(i64::from(q.cast_signed()));
                    u32::try_from(wide.cast_unsigned() >> 32).unwrap_or(0)
                }),
                Binary::UAddCarry => each2(m, &x, &y, |p, q| u32::from(p.checked_add(q).is_none())),
                Binary::USubBorrow => each2(m, &x, &y, |p, q| u32::from(p < q)),
                Binary::SDiv(z) => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u32::MAX,
                    (0, _) => p,
                    // only the most negative value by -1 overflows: it wraps to itself
                    _ => p
                        .cast_signed()
                        .checked_div(q.cast_signed())
                        .map_or(p, i32::cast_unsigned),
                }),
                Binary::SRem(z) => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u32::MAX,
                    (0, _) => 0,
                    _ => p
                        .cast_signed()
                        .checked_rem(q.cast_signed())
                        .map_or(0, i32::cast_unsigned),
                }),
                Binary::UMin => each2(m, &x, &y, u32::min),
                Binary::UMax => each2(m, &x, &y, u32::max),
                Binary::Shl => each2(m, &x, &y, |p, q| p.wrapping_shl(q)),
                Binary::ShrU => each2(m, &x, &y, |p, q| p.wrapping_shr(q)),
                Binary::ShrS => each2(m, &x, &y, |p, q| p.cast_signed().wrapping_shr(q).cast_unsigned()),
                Binary::UDiv(z) => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u32::MAX,
                    (0, _) => p,
                    _ => p.checked_div(q).unwrap_or(0),
                }),
                Binary::URem(z) => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u32::MAX,
                    (0, _) => 0,
                    _ => p.checked_rem(q).unwrap_or(0),
                }),
                Binary::SMin => each2(m, &x, &y, |p, q| {
                    p.cast_signed().min(q.cast_signed()).cast_unsigned()
                }),
                Binary::SMax => each2(m, &x, &y, |p, q| {
                    p.cast_signed().max(q.cast_signed()).cast_unsigned()
                }),
                Binary::And => each2(m, &x, &y, |p, q| p & q),
                Binary::Or => each2(m, &x, &y, |p, q| p | q),
                Binary::Xor => each2(m, &x, &y, |p, q| p ^ q),
            }
        }
        Op::Ternary(op, a, b, c) => {
            let (x, y, z) = (ops.arg(*a, flush)?, ops.arg(*b, flush)?, ops.arg(*c, flush)?);
            let m = ops.ty(*a)?.lanes;
            let mut out = [0; 4];
            for (k, o) in out.iter_mut().enumerate().take(usize::from(m)) {
                let lane = |l: &Lanes| l.get(k).copied().unwrap_or(0);
                *o = match op {
                    Ternary::Fma => fb(lane(&x)).mul_add(fb(lane(&y)), fb(lane(&z))).to_bits(),
                    Ternary::BitExtract { signed } => bit_extract(lane(&x), lane(&y), lane(&z), *signed),
                };
            }
            out
        }
        Op::BitInsert {
            base,
            insert,
            offset,
            count,
        } => {
            let (b, i, o, c) = (
                ops.get(*base)?,
                ops.get(*insert)?,
                ops.get(*offset)?,
                ops.get(*count)?,
            );
            let mut out = [0; 4];
            for (k, r) in out.iter_mut().enumerate().take(usize::from(n)) {
                let lane = |l: &Lanes| l.get(k).copied().unwrap_or(0);
                *r = bit_insert(lane(&b), lane(&i), lane(&o), lane(&c));
            }
            out
        }
        Op::Select { cond, accept, reject } => {
            let (c, a, r) = (ops.get(*cond)?, ops.get(*accept)?, ops.get(*reject)?);
            let wide = ops.ty(*cond)?.lanes > 1;
            let mut out = [0; 4];
            for (i, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                let pick = if wide {
                    c.get(i).copied()
                } else {
                    c.first().copied()
                }
                .unwrap_or(0)
                    != 0;
                *o = if pick { a.get(i) } else { r.get(i) }.copied().unwrap_or(0);
            }
            out
        }
        Op::Compose(parts) => {
            let mut out = [0; 4];
            let mut at = 0usize;
            for p in parts {
                let x = ops.get(*p)?;
                for w in x.iter().take(usize::from(ops.ty(*p)?.lanes)) {
                    *out.get_mut(at).ok_or("a compose of more than four lanes")? = *w;
                    at = at.saturating_add(1);
                }
            }
            out
        }
        Op::Swizzle(a, lanes) => {
            let x = ops.get(*a)?;
            let mut out = [0; 4];
            for (o, &l) in out.iter_mut().zip(lanes).take(usize::from(n)) {
                *o = *x.get(usize::from(l)).ok_or("a swizzle of lane past four")?;
            }
            out
        }
        Op::Splat(a) => {
            let x = ops.get(*a)?;
            let mut out = [0; 4];
            for o in out.iter_mut().take(usize::from(n)) {
                *o = x[0];
            }
            out
        }
        Op::Dot(a, b) => {
            let (x, y) = (ops.arg(*a, flush)?, ops.arg(*b, flush)?);
            let m = ops.ty(*a)?.lanes;
            // each step is rounded, and flushed where the run flushes
            let step = |f: f32| if flush { fb(flushed(f.to_bits())) } else { f };
            let mut acc = 0.0f32;
            for (i, (p, q)) in x.iter().zip(&y).take(usize::from(m)).enumerate() {
                let prod = step(fb(*p) * fb(*q));
                acc = if i == 0 { prod } else { step(acc + prod) };
            }
            [acc.to_bits(), 0, 0, 0]
        }
        Op::Convert(a, rule) => {
            let x = ops.arg(*a, flush)?;
            let from = ops.ty(*a)?;
            match (from.scalar, t.scalar) {
                // Rust's `as` truncates toward zero, saturates and gives 0 for NaN: D3D's ftoi and ftou
                (Scalar::F32, Scalar::I32) if *rule == FloatToInt::Saturate => {
                    each(n, &x, |w| {
                        #[allow(clippy::cast_possible_truncation)] // the saturation is the operation
                        let r = fb(w) as i32;
                        r.cast_unsigned()
                    })
                }
                (Scalar::F32, Scalar::U32) if *rule == FloatToInt::Saturate => each(n, &x, |w| {
                    #[allow(clippy::cast_possible_truncation)]
                    let r = fb(w) as u32;
                    r
                }),
                (Scalar::I32, Scalar::F32) => each(n, &x, |w| {
                    #[allow(clippy::cast_precision_loss)] // to nearest even, the conversion's rule
                    let r = w.cast_signed() as f32;
                    r.to_bits()
                }),
                (Scalar::U32, Scalar::F32) => each(n, &x, |w| {
                    #[allow(clippy::cast_precision_loss)]
                    let r = w as f32;
                    r.to_bits()
                }),
                (Scalar::I32, Scalar::U32) | (Scalar::U32, Scalar::I32) => x,
                (s, d) => {
                    return Err(format!(
                        "a conversion of {} to {} ({rule:?}) is not run yet",
                        s.name(),
                        d.name()
                    ));
                }
            }
        }
        Op::Bitcast(a) => ops.get(*a)?,
        _ => return Ok(None),
    };
    // the float operations whose inputs and result a flushing run flushes: not the sign operations and moves
    let arithmetic = match op {
        Op::Unary(u, _) => !matches!(u, Unary::FNeg | Unary::FAbs),
        Op::Binary(..) | Op::Ternary(..) | Op::Dot(..) | Op::Convert(..) => true,
        _ => false,
    };
    Ok(Some(if flush && arithmetic && t.scalar == Scalar::F32 {
        out.map(flushed)
    } else {
        out
    }))
}

impl Operands for Machine<'_, '_> {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        Machine::get(self, v)
    }
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        Machine::ty(self, v)
    }
}

impl Machine<'_, '_> {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        self.values
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("%{} has no value here", v.0))
    }
    fn set(&mut self, v: ValueId, x: Lanes) -> Result<(), String> {
        *self
            .values
            .get_mut(v.0 as usize)
            .ok_or_else(|| format!("%{} is out of range", v.0))? = Some(x);
        Ok(())
    }
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        self.f.ty(v).ok_or_else(|| format!("%{} has no type", v.0))
    }
    fn buffer(&self, id: ResourceId) -> Result<(&[Lanes], u32), String> {
        let rows = match self
            .program
            .interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .map(|r| &r.kind)
        {
            Some(ResourceKind::ConstantBuffer { layout }) => layout.size.div_ceil(16),
            _ => return Err(format!("resource {} is not a constant buffer", id.0)),
        };
        let data = self
            .bound
            .buffers
            .iter()
            .find(|(r, _)| *r == id)
            .map(|(_, d)| *d)
            .ok_or_else(|| format!("constant buffer {} is not bound", id.0))?;
        Ok((data, rows))
    }
    /// Meet the quad at value `v`'s operation with this lane's `operand`: every lane's operand in quad order where the
    /// lanes have all met here before, a stop where they have not, `None` where the run is not a quad's.
    fn meet(&mut self, v: ValueId, operand: Lanes) -> Step<Option<[Lanes; 4]>> {
        let steps = self.steps;
        let Some(q) = self.quad.as_mut() else {
            return Ok(None);
        };
        let times = q.times.get_mut(v.0 as usize).ok_or("a value out of range")?;
        let before = *times;
        *times = times.saturating_add(1);
        if let Some(known) = q.known.get(q.met) {
            q.met = q.met.saturating_add(1);
            return Ok(Some(*known));
        }
        q.waiting = Some((
            Place {
                value: v,
                steps,
                before,
            },
            operand,
        ));
        Err(Stop::Meet)
    }
    fn op(&mut self, v: ValueId) -> Step<Lanes> {
        let def = self
            .f
            .value(v)
            .ok_or_else(|| format!("%{} is not a value", v.0))?;
        let t = def.ty;
        if !runs(t) {
            return Err(format!("{t} values are not run yet").into());
        }
        self.compute(v, &def.op, t)
    }
    /// An operand of an arithmetic operation: its lanes, flushed where the run flushes and it is a float.
    fn arg(&self, v: ValueId) -> Result<Lanes, String> {
        Operands::arg(self, v, self.flush)
    }
    fn compute(&mut self, v: ValueId, op: &Op, t: Type) -> Step<Lanes> {
        if let Some(x) = eval(op, t, &*self, self.flush)? {
            return Ok(x);
        }
        let n = t.lanes;
        Ok(match op {
            Op::Literal(bits) => literal_words(bits, t),
            Op::Undef => [0; 4],
            Op::Param => return Err(format!("parameter %{} has no value here", v.0).into()),
            Op::Unary(Unary::Derivative(axis, fineness), a) => {
                let x = self.arg(*a)?;
                let m = self.ty(*a)?.lanes;
                let quad = self
                    .meet(v, x)?
                    .ok_or("a derivative needs the invocation's quad, which a single run does not have")?;
                let (from, to) = pair(*axis, *fineness, self.quad.as_ref().map_or(0, |q| q.lane));
                let lane = |l: usize| quad.get(l).copied().unwrap_or_default();
                let d = float2(m, &lane(to), &lane(from), |p, q| p - q);
                if self.flush { d.map(flushed) } else { d }
            }
            Op::LoadInput(slot, mask) => {
                let all = self
                    .bound
                    .inputs
                    .get(*slot as usize)
                    .ok_or_else(|| format!("input {slot} is not given"))?;
                let mut x = [0; 4];
                for (to, from) in x.iter_mut().zip(masked(*mask)) {
                    *to = all.get(from).copied().unwrap_or(0);
                }
                x
            }
            Op::LoadVar(var) => self
                .vars
                .get(var.0 as usize)
                .and_then(|e| e.first())
                .copied()
                .ok_or("a load of no variable")?,
            Op::LoadElement(var, index, oob) => {
                let i = self.get(*index)?[0] as usize;
                match self
                    .vars
                    .get(var.0 as usize)
                    .ok_or("a load of no variable")?
                    .get(i)
                {
                    Some(x) => *x,
                    None if *oob == OutOfBounds::Zero => [0; 4],
                    None => return Err("an undefined read past an array's end".into()),
                }
            }
            Op::LoadInputAt(..)
            | Op::LoadOutputAt(..)
            | Op::LoadPatch(..)
            | Op::Builtin(_)
            | Op::Gather { .. }
            | Op::LoadTexel { .. }
            | Op::Query(..)
            | Op::Atomic { .. }
            | Op::Counter(..)
            | Op::ReportHit { .. }
            | Op::Wave(_) => {
                return Err(format!("{} is not run here: iridia-exec runs it", op_name(op)).into());
            }
            // the pure operations, which `eval` took above
            Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::BitInsert { .. }
            | Op::Select { .. }
            | Op::Compose(_)
            | Op::Swizzle(..)
            | Op::Splat(_)
            | Op::Dot(..)
            | Op::Convert(..)
            | Op::Bitcast(_) => return Err("a pure operation eval did not take".into()),
            Op::LoadState(r) => self
                .bound
                .state
                .iter()
                .find(|(id, _)| id == r)
                .map(|(_, x)| *x)
                .ok_or_else(|| format!("state {} is not given", r.0))?,
            Op::LoadBuffer {
                resource,
                address,
                oob,
            } => {
                if resource.index.is_some() {
                    return Err("an element of an array of constant buffers is not run here".into());
                }
                let (data, rows) = self.buffer(resource.id)?;
                // the address in wrapping 32-bit arithmetic, its low two bits ignored
                let base = match address.index {
                    None => address.offset,
                    Some(i) => self.get(i)?[0]
                        .wrapping_mul(address.stride)
                        .wrapping_add(address.offset),
                } / 4;
                let words = rows.saturating_mul(4);
                let mut out = [0; 4];
                for (k, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                    let word = base
                        .checked_add(u32::try_from(k).unwrap_or(u32::MAX))
                        .filter(|w| *w < words);
                    *o = match word {
                        Some(w) => data
                            .get(w as usize / 4)
                            .and_then(|r| r.get(w as usize % 4))
                            .copied()
                            .unwrap_or(0),
                        None if *oob == OutOfBounds::Zero => 0,
                        None => return Err("an undefined read outside a buffer".into()),
                    };
                }
                out
            }
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp,
            } => {
                if texture.index.is_some() || sampler.index.is_some() {
                    return Err("a sample of an element of an array of resources is not run here".into());
                }
                if compare.is_some() || clamp.is_some() || *offset != [0; 3] {
                    return Err("a sample with a comparison, offset or level clamp is not run here".into());
                }
                let c = self.get(*coord)?;
                let lanes = self.ty(*coord)?.lanes;
                // a quad's implicit level of detail comes from the coordinate's differences: the right lane's and the
                // lower lane's less the top left's, as `iridia-exec` takes them
                let footprint = match level {
                    Level::Auto | Level::Bias(_) => self.meet(v, c)?.map(|quad| {
                        let d = |l: usize| {
                            let mut r = [0.0; 3];
                            for (k, o) in r.iter_mut().enumerate().take(usize::from(lanes)) {
                                let lane =
                                    |l: usize| quad.get(l).and_then(|q| q.get(k)).copied().unwrap_or(0);
                                *o = fb(lane(l)) - fb(lane(0));
                            }
                            r
                        };
                        (d(1), d(2))
                    }),
                    _ => None,
                };
                let lod = match (level, footprint) {
                    (Level::Auto, Some((dx, dy))) => Lod::Grad(dx, dy),
                    (Level::Bias(b), Some((dx, dy))) => Lod::GradBias(dx, dy, fb(self.get(*b)?[0])),
                    (Level::Auto, None) => Lod::Auto,
                    (Level::Zero, _) => Lod::Exact(0.0),
                    (Level::Bias(b), None) => Lod::Bias(fb(self.get(*b)?[0])),
                    (Level::Exact(l), _) => Lod::Exact(fb(self.get(*l)?[0])),
                    (Level::Grad(x, y), _) => {
                        let (x, y) = (self.get(*x)?, self.get(*y)?);
                        Lod::Grad([fb(x[0]), fb(x[1]), fb(x[2])], [fb(y[0]), fb(y[1]), fb(y[2])])
                    }
                };
                let request = Sample {
                    texture: texture.id,
                    sampler: sampler.id,
                    coord: [fb(c[0]), fb(c[1]), fb(c[2])],
                    lanes,
                    lod,
                };
                (self.bound.sampler)(request)?.map(f32::to_bits)
            }
        })
    }
    /// A node that is not a construct: a value, a variable store or an output store.
    fn plain(&mut self, node: &Node) -> Step<()> {
        match node {
            Node::Value(v) => {
                let x = self.op(*v)?;
                self.set(*v, x)?;
            }
            Node::StoreVar(var, x) => {
                let x = self.get(*x)?;
                *self
                    .vars
                    .get_mut(var.0 as usize)
                    .and_then(|e| e.first_mut())
                    .ok_or("a store to no variable")? = x;
            }
            Node::StoreElement(var, index, x, oob) => {
                let (i, x) = (self.get(*index)?[0] as usize, self.get(*x)?);
                match self
                    .vars
                    .get_mut(var.0 as usize)
                    .ok_or("a store to no variable")?
                    .get_mut(i)
                {
                    Some(e) => *e = x,
                    None if *oob == OutOfBounds::Zero => {}
                    None => return Err("an undefined write past an array's end".into()),
                }
            }
            Node::StoreBuffer { .. }
            | Node::StoreTexel { .. }
            | Node::StorePatch(..)
            | Node::Barrier(_)
            | Node::Emit(_)
            | Node::Cut(_)
            | Node::SetMeshCounts { .. }
            | Node::StoreVertex { .. }
            | Node::StoreIndices { .. }
            | Node::DispatchMesh { .. } => {
                return Err("a buffer, image or patch store, a barrier, a vertex emit, a meshlet store or a mesh launch is not run here: iridia-exec runs it".into());
            }
            Node::TraceRay { .. } | Node::CallShader { .. } => {
                return Err("a traced ray or a shader call is not run here: iridia-exec's ray tracing pipeline runs it".into());
            }
            Node::Demote => self.demoted = true,
            Node::StoreOutput(..) if self.demoted => {}
            Node::StoreOutput(slot, mask, x) => {
                let x = self.get(*x)?;
                let out = self
                    .outputs
                    .get_mut(*slot as usize)
                    .ok_or("a store to no output")?;
                let lanes = out.get_or_insert([0; 4]);
                for (from, to) in x.iter().zip(masked(*mask)) {
                    if let Some(l) = lanes.get_mut(to) {
                        *l = *from;
                    }
                }
            }
            Node::If { .. } | Node::Block { .. } | Node::Loop { .. } | Node::Switch { .. } => {
                return Err("a construct in a graph block".into());
            }
        }
        Ok(())
    }
    /// A graph body from its entry block until it returns or kills: whether it killed.
    fn graph(&mut self, blocks: &[Block]) -> Step<bool> {
        let mut at = 0usize;
        loop {
            self.steps = self.steps.saturating_add(1);
            if self.steps > STEP_LIMIT {
                return Err(format!("more than {STEP_LIMIT} block executions").into());
            }
            let block = blocks
                .get(at)
                .ok_or_else(|| format!("block {at} does not exist"))?;
            for n in &block.nodes {
                self.plain(n)?;
            }
            let target = match &block.term {
                Terminator::Jump(t) => t,
                Terminator::Branch { cond, accept, reject } => {
                    if self.get(*cond)?[0] != 0 {
                        accept
                    } else {
                        reject
                    }
                }
                Terminator::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    let v = self.get(*selector)?[0];
                    cases.iter().find(|(c, _)| *c == v).map_or(default, |(_, t)| t)
                }
                Terminator::Return => return Ok(false),
                Terminator::Kill => return Ok(true),
                Terminator::IgnoreHit | Terminator::AcceptHitAndEndSearch => {
                    return Err("an any-hit program's decision is not run here: iridia-exec runs it".into());
                }
                Terminator::Unreachable => {
                    return Err(format!("reached block {at}'s unreachable end").into());
                }
            };
            let next = blocks
                .get(target.block as usize)
                .ok_or_else(|| format!("block {} does not exist", target.block))?;
            // the parameters take their arguments together: every argument is read before any is set
            let args: Vec<Lanes> = target
                .args
                .iter()
                .map(|a| self.get(*a))
                .collect::<Result<_, _>>()?;
            for (p, a) in next.params.iter().zip(args) {
                self.set(*p, a)?;
            }
            at = target.block as usize;
        }
    }
    fn region(&mut self, r: &Region) -> Step<Exit> {
        for node in &r.nodes {
            match node {
                Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                } => {
                    let v = self.get(*selector)?[0];
                    let arm = cases
                        .iter()
                        .find(|c| c.values.contains(&v))
                        .map_or(default, |c| &c.body);
                    match self.region(arm)? {
                        Exit::Yield(vs) => self.give(results, &vs)?,
                        Exit::Br(l, vs) if l == *label => self.give(results, &vs)?,
                        other => return Ok(other),
                    }
                }
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    let arm = if self.get(*cond)?[0] != 0 { accept } else { reject };
                    match self.region(arm)? {
                        Exit::Yield(vs) => self.give(results, &vs)?,
                        other => return Ok(other),
                    }
                }
                Node::Block { label, body, results } => match self.region(body)? {
                    Exit::Yield(vs) => self.give(results, &vs)?,
                    Exit::Br(l, vs) if l == *label => self.give(results, &vs)?,
                    other => return Ok(other),
                },
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    let mut args: Vec<Lanes> =
                        inits.iter().map(|i| self.get(*i)).collect::<Result<_, _>>()?;
                    loop {
                        self.steps = self.steps.saturating_add(1);
                        if self.steps > STEP_LIMIT {
                            return Err(format!("more than {STEP_LIMIT} loop iterations").into());
                        }
                        for (p, a) in params.iter().zip(&args) {
                            self.set(*p, *a)?;
                        }
                        match self.region(body)? {
                            Exit::Continue(l, vs) if l == *label => {
                                args = vs.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
                            }
                            Exit::Yield(vs) => {
                                self.give(results, &vs)?;
                                break;
                            }
                            Exit::Br(l, vs) if l == *label => {
                                self.give(results, &vs)?;
                                break;
                            }
                            other => return Ok(other),
                        }
                    }
                }
                _ => self.plain(node)?,
            }
        }
        Ok(match &r.end {
            Some(End::Yield(vs)) => Exit::Yield(vs.clone()),
            Some(End::Br(l, vs)) => Exit::Br(*l, vs.clone()),
            Some(End::Continue(l, vs)) => Exit::Continue(*l, vs.clone()),
            Some(End::Return) => Exit::Return,
            Some(End::Kill) => Exit::Kill,
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) => {
                return Err("an any-hit program's decision is not run here: iridia-exec runs it".into());
            }
            Some(End::Unreachable) => return Err("reached an unreachable end".into()),
            None => return Err("a region with no end".into()),
        })
    }
    fn give(&mut self, results: &[ValueId], vs: &[ValueId]) -> Result<(), String> {
        for (r, v) in results.iter().zip(vs) {
            let x = self.get(*v)?;
            self.set(*r, x)?;
        }
        Ok(())
    }
}

/// How a lane's run ended: its outputs, or the meeting it waits at.
enum Ended {
    Ran(Run),
    Waits(Place, Lanes),
}

/// Run a program's entry function for one lane, the quad's meetings known so far where it is a quad's.
fn lane(
    program: &Program,
    bound: &mut Bound<'_>,
    quad: Option<(usize, &[[Lanes; 4]])>,
) -> Result<Ended, String> {
    let f = &program.entry;
    let flush = match program.float.f32 {
        Denormals::Flush => true,
        Denormals::Preserve => false,
        Denormals::Any => bound.denormals == Denormals::Flush,
    };
    let mut m = Machine {
        program,
        f,
        values: vec![None; f.values.len()],
        vars: f
            .vars
            .iter()
            .map(|v| {
                let len = v.len.unwrap_or(1) as usize;
                let mut e: Vec<Lanes> = v
                    .init
                    .iter()
                    .map(|bits| bits.map(|b| u32::try_from(b & 0xffff_ffff).unwrap_or(0)))
                    .collect();
                e.resize(len, [0; 4]);
                e
            })
            .collect(),
        bound,
        outputs: vec![None; program.interface.outputs.len()],
        steps: 0,
        flush,
        quad: quad.map(|(lane, known)| Quad {
            lane,
            known,
            met: 0,
            times: vec![0; f.values.len()],
            waiting: None,
        }),
        demoted: false,
    };
    let ended = match &f.body {
        Body::Regions(body) => match m.region(body) {
            Ok(Exit::Kill) => Ok(true),
            Ok(Exit::Return | Exit::Yield(_)) => Ok(false),
            Ok(Exit::Br(..) | Exit::Continue(..)) => return Err("an exit past the function's body".into()),
            Err(e) => Err(e),
        },
        Body::Cfg(blocks) => m.graph(blocks),
    };
    match ended {
        Ok(killed) => Ok(Ended::Ran(Run {
            outputs: m.outputs,
            killed: killed || m.demoted,
        })),
        Err(Stop::Fail(e)) => Err(e),
        Err(Stop::Meet) => match m.quad.and_then(|q| q.waiting) {
            Some((place, operand)) => Ok(Ended::Waits(place, operand)),
            None => Err("a run stopped at a meeting it did not record".into()),
        },
    }
}

/// Run a program's entry function once, in either control phase.
pub fn run(program: &Program, bound: &mut Bound<'_>) -> Result<Run, String> {
    match lane(program, bound, None)? {
        Ended::Ran(run) => Ok(run),
        Ended::Waits(..) => Err("a single run met its quad".into()),
    }
}

/// Whether a program needs the other lanes of its 2 x 2 quad ([`run_quad`]): it takes a screen derivative, or samples
/// at an implicit level of detail, whose footprint is the quad's. A program that does not runs the same alone as in a
/// quad.
pub fn needs_quad(program: &Program) -> bool {
    program.entry.values.iter().any(|d| {
        matches!(
            d.op,
            Op::Unary(Unary::Derivative(..), _)
                | Op::Sample {
                    level: Level::Auto | Level::Bias(_),
                    ..
                }
        )
    })
}

/// Run a pixel program's entry function for a 2 x 2 quad of invocations, each lane with its own [`Bound`]: lanes 0 and
/// 1 are the top row, left to right, and 2 and 3 the bottom.
///
/// A derivative is a difference between two lanes' operands: along x the right lane's less the left's, along y the
/// lower's less the upper's, of the top row or left column for a coarse derivative (and for [`Fineness::Any`]), of the
/// lane's own for a fine one. An implicit-LOD sample (`Level::Auto`, `Level::Bias`) asks the sampler with the quad's
/// coarse derivatives of its coordinate ([`Lod::Grad`], [`Lod::GradBias`]).
///
/// The lanes run as they would in lockstep, with no threads: each runs until it meets a derivative or implicit-LOD
/// sample whose operands are not all known, and once all four have met it there, they run again knowing them. Each
/// sampler may thus be asked a request again, and must answer it the same way. Lanes must meet each such operation
/// together: at the same operation, after as many loop iterations or block executions, as often before; a lane that
/// ends (returns or kills) while the others go on to one, or that meets one elsewhere, makes the run refused as
/// divergent. So is a run of more than [`MEETING_LIMIT`] meetings.
pub fn run_quad(program: &Program, bounds: [&mut Bound<'_>; 4]) -> Result<[Run; 4], String> {
    let mut bounds = bounds;
    let mut known: Vec<[Lanes; 4]> = Vec::new();
    loop {
        let mut ran = Vec::with_capacity(4);
        let mut waits = Vec::with_capacity(4);
        for (l, bound) in bounds.iter_mut().enumerate() {
            match lane(program, bound, Some((l, &known))).map_err(|e| format!("lane {l}: {e}"))? {
                Ended::Ran(run) => ran.push(run),
                Ended::Waits(place, operand) => waits.push((l, place, operand)),
            }
        }
        if waits.is_empty() {
            return ran
                .try_into()
                .map_err(|_| "a quad of other than four lanes".to_string());
        }
        let Some(&(_, first, _)) = waits.first() else {
            return Err("a quad with no lane waiting".into());
        };
        if waits.len() < 4 || waits.iter().any(|(_, place, _)| *place != first) {
            let at: Vec<String> = (0..4)
                .map(|l| match waits.iter().find(|(w, ..)| *w == l) {
                    Some((_, p, _)) => format!(
                        "lane {l} at %{} (after {} steps, met {} times before)",
                        p.value.0, p.steps, p.before
                    ),
                    None => format!("lane {l} ended"),
                })
                .collect();
            return Err(format!(
                "a quad's lanes diverge at a derivative or an implicit-LOD sample: {}",
                at.join(", ")
            ));
        }
        if known.len() >= MEETING_LIMIT {
            return Err(format!("more than {MEETING_LIMIT} quad meetings"));
        }
        let mut operands = [[0; 4]; 4];
        for (l, _, operand) in &waits {
            if let Some(o) = operands.get_mut(*l) {
                *o = *operand;
            }
        }
        known.push(operands);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every NaN, of either sign and any payload, converts to the one quiet half NaN; an infinity keeps its sign.
    #[test]
    fn every_nan_is_one_half_nan() {
        for sign in [0, 0x8000_0000u32] {
            for man in 1..0x80_0000u32 {
                let w = sign | 0x7f80_0000 | man;
                assert_eq!(f32_to_f16(w), 0x7e00, "{w:#x}");
            }
            assert_eq!(f32_to_f16(sign | 0x7f80_0000), sign >> 16 | 0x7c00);
        }
    }
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
