//! The text form read back: [`parse`] is the inverse of [`print`](super::print) for everything the text states.
//! What it does not state comes back empty: origins, and values the body never places (an id the text skips is an
//! unplaced `undef`). Printing what `parse` gives gives the same text.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use super::{SCHEMA, binary, builtin, ternary, unary};
use crate::{
    Address, AtomicOp, Axis, Barrier, Binary, Block, Body, Builtin, Case, Cmp, CounterOp, End, Fineness,
    FloatToInt, Function, GatherOffset, LabelId, Level, Memory, MinMaxNan, MulZero, Node, OnZero, Op,
    OutOfBounds, Precision, QuadSwap, Query, Ray, Region, Res, Rounding, SaturateNan, Scalar, Scope, Space,
    Target, Terminator, Ternary, Type, Unary, ValueDef, ValueId, Var, VarId, Wave, WaveOp, WaveScan,
    interface::ResourceId,
};

/// Why a text cannot be read: the line (from 1) and what is wrong there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextError {
    /// The line, counting from 1; 0 for the text as a whole.
    pub line: usize,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl core::error::Error for TextError {}

type Result<T> = core::result::Result<T, TextError>;

const ROUNDINGS: [Rounding; 5] = [
    Rounding::Floor,
    Rounding::Ceil,
    Rounding::Trunc,
    Rounding::NearestEven,
    Rounding::NearestAway,
];
const CMPS: [Cmp; 6] = [Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge];
const NANS: [MinMaxNan; 3] = [MinMaxNan::MinNum, MinMaxNan::Propagate, MinMaxNan::Any];
const ON_ZERO: [OnZero; 3] = [OnZero::AllOnes, OnZero::ByOne, OnZero::Undef];

/// Every unary operation, for matching a name against the name the printer gives it.
fn unaries() -> Vec<Unary> {
    let mut all = alloc::vec![
        Unary::FNeg,
        Unary::FAbs,
        Unary::FFract,
        Unary::FSqrt,
        Unary::FInverseSqrt,
        Unary::FExp2,
        Unary::FLog2,
        Unary::FSin,
        Unary::FCos,
        Unary::Saturate(SaturateNan::Zero),
        Unary::Saturate(SaturateNan::Any),
        Unary::Not,
        Unary::INeg,
        Unary::Any,
        Unary::All,
        Unary::IsNan,
        Unary::FTan,
        Unary::FAsin,
        Unary::FAcos,
        Unary::FAtan,
        Unary::FSinh,
        Unary::FCosh,
        Unary::FTanh,
        Unary::FSign,
        Unary::IsInf,
        Unary::CountOnes,
        Unary::BitReverse,
        Unary::FirstBitHigh,
        Unary::FirstBitHighSigned,
        Unary::FirstBitLow,
        Unary::F32ToF16,
        Unary::F16ToF32,
    ];
    all.extend(ROUNDINGS.map(Unary::FRound));
    for axis in [Axis::X, Axis::Y] {
        for fineness in [Fineness::Coarse, Fineness::Fine, Fineness::Any] {
            all.push(Unary::Derivative(axis, fineness));
        }
    }
    for frac_bits in 0..=23 {
        all.extend(ROUNDINGS.map(|rounding| Unary::Quantize { frac_bits, rounding }));
    }
    all
}

/// Every binary operation, likewise.
fn binaries() -> Vec<Binary> {
    let mut all = alloc::vec![
        Binary::FAdd,
        Binary::FSub,
        Binary::FMul(MulZero::Ieee),
        Binary::FMul(MulZero::ZeroWins),
        Binary::FDiv,
        Binary::FPow,
        Binary::IAdd,
        Binary::ISub,
        Binary::IMul,
        Binary::SMin,
        Binary::SMax,
        Binary::And,
        Binary::Or,
        Binary::Xor,
        Binary::FRem,
        Binary::UMulHigh,
        Binary::SMulHigh,
        Binary::UAddCarry,
        Binary::USubBorrow,
        Binary::UMin,
        Binary::UMax,
        Binary::Shl,
        Binary::ShrU,
        Binary::ShrS,
    ];
    all.extend(NANS.map(Binary::FMin));
    all.extend(NANS.map(Binary::FMax));
    all.extend(CMPS.map(Binary::FCmp));
    all.extend(CMPS.map(Binary::SCmp));
    all.extend(CMPS.map(Binary::UCmp));
    all.extend(ON_ZERO.map(Binary::UDiv));
    all.extend(ON_ZERO.map(Binary::URem));
    all.extend(ON_ZERO.map(Binary::SDiv));
    all.extend(ON_ZERO.map(Binary::SRem));
    all
}

const TERNARIES: [Ternary; 3] = [
    Ternary::Fma,
    Ternary::BitExtract { signed: false },
    Ternary::BitExtract { signed: true },
];
const BUILTINS: [Builtin; 5] = [
    Builtin::OutputControlPoint,
    Builtin::PrimitiveId,
    Builtin::GsInstance,
    Builtin::DomainLocation,
    Builtin::PhaseInstance,
];
/// Every wave operation of one value, in the text form's order.
const WAVE_OPS: [WaveOp; 9] = [
    WaveOp::Add,
    WaveOp::Mul,
    WaveOp::Min,
    WaveOp::Max,
    WaveOp::And,
    WaveOp::Or,
    WaveOp::Xor,
    WaveOp::All,
    WaveOp::Any,
];

const WAVE_SCANS: [WaveScan; 3] = [WaveScan::Reduce, WaveScan::Inclusive, WaveScan::Exclusive];

const QUAD_SWAPS: [QuadSwap; 3] = [QuadSwap::X, QuadSwap::Y, QuadSwap::Diagonal];

const ATOMICS: [AtomicOp; 11] = [
    AtomicOp::Add,
    AtomicOp::Sub,
    AtomicOp::And,
    AtomicOp::Or,
    AtomicOp::Xor,
    AtomicOp::SMin,
    AtomicOp::SMax,
    AtomicOp::UMin,
    AtomicOp::UMax,
    AtomicOp::Exchange,
    AtomicOp::CompareExchange,
];

/// `s` cut at the first of each keyword ` key ` in turn: the head and each keyword's text, in the order given; a
/// keyword may be missing.
fn keywords<'s>(s: &'s str, keys: &[&'static str]) -> (&'s str, Vec<(&'static str, &'s str)>) {
    let mut cuts: Vec<(usize, &'static str)> = keys
        .iter()
        .filter_map(|k| s.find(&format!(" {k} ")).map(|at| (at, *k)))
        .collect();
    cuts.sort_unstable();
    let head = cuts.first().map_or(s, |(at, _)| s.get(..*at).unwrap_or(s));
    let mut out = Vec::new();
    for (i, (at, k)) in cuts.iter().enumerate() {
        let start = at.saturating_add(k.len()).saturating_add(2);
        let end = cuts.get(i.saturating_add(1)).map_or(s.len(), |(n, _)| *n);
        out.push((*k, s.get(start..end).unwrap_or("")));
    }
    (head, out)
}

fn lane_of(c: char) -> Option<u8> {
    match c {
        'x' => Some(0),
        'y' => Some(1),
        'z' => Some(2),
        'w' => Some(3),
        _ => None,
    }
}

struct Parser<'a> {
    lines: Vec<(usize, &'a str)>,
    at: usize,
    values: Vec<Option<ValueDef>>,
    vars: Vec<Var>,
    var_names: BTreeMap<String, VarId>,
}

impl<'a> Parser<'a> {
    fn line_no(&self) -> usize {
        self.lines.get(self.at.saturating_sub(1)).map_or(0, |l| l.0)
    }
    fn fail<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(TextError {
            line: self.line_no(),
            message: message.into(),
        })
    }
    fn next(&mut self) -> Result<&'a str> {
        let line = self.lines.get(self.at).map(|l| l.1);
        self.at = self.at.saturating_add(1);
        match line {
            Some(l) => Ok(l),
            None => self.fail("the text ends inside the function"),
        }
    }
    fn peek(&self) -> Option<&'a str> {
        self.lines.get(self.at).map(|l| l.1)
    }

    fn ty(&self, s: &str) -> Result<Type> {
        let s = s.trim();
        let t = match s.strip_prefix("vec").and_then(|r| r.split_once('<')) {
            Some((n, rest)) => {
                let lanes = n.parse::<u8>().ok();
                let scalar = rest.strip_suffix('>').and_then(Scalar::parse);
                match (lanes, scalar) {
                    (Some(l), Some(sc)) => Type::vector(sc, l),
                    _ => return self.fail(format!("not a type: {s}")),
                }
            }
            None => match Scalar::parse(s) {
                Some(sc) => Type::scalar(sc),
                None => return self.fail(format!("not a type: {s}")),
            },
        };
        if t.is_valid() {
            Ok(t)
        } else {
            self.fail(format!("not a type: {s}"))
        }
    }
    fn value_id(&self, s: &str) -> Result<ValueId> {
        match s.trim().strip_prefix('%').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(ValueId(n)),
            None => self.fail(format!("not a value: {s}")),
        }
    }
    fn list(&self, s: &str) -> Result<Vec<ValueId>> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Vec::new());
        }
        s.split(',').map(|v| self.value_id(v)).collect()
    }
    fn resource(&self, s: &str) -> Result<ResourceId> {
        match s.trim().strip_prefix('r').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(ResourceId(n)),
            None => self.fail(format!("not a resource: {s}")),
        }
    }
    /// `rN`, `rN@%i`, `rN@%i!`.
    fn res(&self, s: &str) -> Result<Res> {
        let s = s.trim();
        match s.split_once('@') {
            None => Ok(Res::of(self.resource(s)?)),
            Some((r, i)) => {
                let (i, nonuniform) = match i.strip_suffix('!') {
                    Some(i) => (i, true),
                    None => (i, false),
                };
                Ok(Res {
                    id: self.resource(r)?,
                    index: Some(self.value_id(i)?),
                    nonuniform,
                })
            }
        }
    }
    fn oob(&self, s: &str) -> Result<OutOfBounds> {
        match s.trim() {
            "oob.zero" => Ok(OutOfBounds::Zero),
            "oob.undef" => Ok(OutOfBounds::Undef),
            other => self.fail(format!("no out-of-bounds rule {other}")),
        }
    }
    fn number<T: core::str::FromStr>(&self, s: &str) -> Result<T> {
        s.trim()
            .parse::<T>()
            .map_or_else(|_| self.fail(format!("not a number: {s}")), Ok)
    }
    /// `(a,b,c)`.
    fn offsets(&self, s: &str) -> Result<[i8; 3]> {
        let parts: Vec<&str> = s
            .trim()
            .strip_prefix('(')
            .and_then(|x| x.strip_suffix(')'))
            .map(|x| x.split(',').collect())
            .unwrap_or_default();
        match parts.as_slice() {
            [a, b, c] => Ok([self.number(a)?, self.number(b)?, self.number(c)?]),
            _ => self.fail(format!("not offsets: {s}")),
        }
    }
    /// `slot.lanes`.
    fn slot(&self, s: &str) -> Result<(u32, u8)> {
        let Some((slot, lanes)) = s.trim().split_once('.') else {
            return self.fail("not slot.lanes");
        };
        Ok((self.number(slot)?, self.mask(lanes)?))
    }
    /// `[%i]`.
    fn bracketed(&self, s: &str) -> Result<ValueId> {
        match s.trim().strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            Some(v) => self.value_id(v),
            None => self.fail(format!("not [%n]: {s}")),
        }
    }
    /// `[%i * stride] + offset` or ` + offset` after a resource.
    fn address(&self, s: &str) -> Result<Address> {
        let Some((head, offset)) = s
            .rsplit_once(" + ")
            .or_else(|| s.strip_prefix("+ ").map(|o| ("", o)))
        else {
            return self.fail(format!("not an address: {s}"));
        };
        let offset = self.number(offset)?;
        let head = head.trim();
        if head.is_empty() {
            return Ok(Address::fixed(offset));
        }
        let Some((i, stride)) = head
            .strip_prefix('[')
            .and_then(|x| x.strip_suffix(']'))
            .and_then(|x| x.split_once(" * "))
        else {
            return self.fail("a buffer index without a stride");
        };
        Ok(Address {
            index: Some(self.value_id(i)?),
            stride: self.number(stride)?,
            offset,
        })
    }
    /// A resource and the address after it: `rN@%i[%j * s] + o`.
    fn res_address(&self, s: &str) -> Result<(Res, Address)> {
        let s = s.trim();
        let cut = s.find(['[', ' ']).unwrap_or(s.len());
        let (r, a) = s.split_at(cut);
        Ok((self.res(r)?, self.address(a)?))
    }
    fn memory(&self, s: &str) -> Result<Memory> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix("buffer ") {
            let (r, a) = self.res_address(rest)?;
            return Ok(Memory::Buffer(r, a));
        }
        if let Some(rest) = s.strip_prefix("image ") {
            let Some((r, c)) = rest.split_once(", ") else {
                return self.fail("an image without a coordinate");
            };
            return Ok(Memory::Image(self.res(r)?, self.value_id(c)?));
        }
        if let Some(rest) = s.strip_prefix("var ") {
            return Ok(match rest.split_once('[') {
                Some((v, i)) => Memory::Var(self.var(v)?, Some(self.bracketed(&format!("[{i}"))?)),
                None => Memory::Var(self.var(rest)?, None),
            });
        }
        self.fail(format!("not memory: {s}"))
    }
    fn label(&self, s: &str) -> Result<LabelId> {
        match s.trim().strip_prefix('\'').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(LabelId(n)),
            None => self.fail(format!("not a label: {s}")),
        }
    }
    fn mask(&self, s: &str) -> Result<u8> {
        let mut mask = 0u8;
        let mut last = None;
        for c in s.chars() {
            let Some(l) = lane_of(c) else {
                return self.fail(format!("not lanes: {s}"));
            };
            if last.is_some_and(|p| p >= l) {
                return self.fail(format!("lanes out of order: {s}"));
            }
            last = Some(l);
            mask |= 1 << l;
        }
        if mask == 0 {
            self.fail("no lanes")
        } else {
            Ok(mask)
        }
    }
    fn define(&mut self, v: ValueId, def: ValueDef) -> Result<()> {
        let i = v.0 as usize;
        if self.values.len() <= i {
            self.values.resize(i.saturating_add(1), None);
        }
        match self.values.get_mut(i) {
            Some(slot @ None) => {
                *slot = Some(def);
                Ok(())
            }
            _ => self.fail(format!("%{} is defined twice", v.0)),
        }
    }
    /// `%n: T, %m: U`: values a construct defines.
    fn params(&mut self, s: &str) -> Result<Vec<ValueId>> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for p in s.split(", ") {
            let Some((v, t)) = p.split_once(':') else {
                return self.fail(format!("not a parameter: {p}"));
            };
            let (v, ty) = (self.value_id(v)?, self.ty(t)?);
            self.define(
                v,
                ValueDef {
                    ty,
                    op: Op::Param,
                    origin: None,
                    precision: Precision::Full,
                },
            )?;
            out.push(v);
        }
        Ok(out)
    }
    fn var(&self, name: &str) -> Result<VarId> {
        match self.var_names.get(name.trim()) {
            Some(v) => Ok(*v),
            None => self.fail(format!("no variable {}", name.trim())),
        }
    }

    fn literal(&self, t: Type, s: &str) -> Result<Op> {
        let Some(inner) = s.strip_prefix(&format!("{t}(")).and_then(|r| r.strip_suffix(')')) else {
            return self.fail(format!("a literal of another type than {t}: {s}"));
        };
        let parts: Vec<&str> = inner.split(", ").collect();
        if parts.len() != usize::from(t.lanes) {
            return self.fail(format!("{} lanes for {t}", parts.len()));
        }
        let mut bits = [0u64; 4];
        for (b, p) in bits.iter_mut().zip(&parts) {
            let hex = p.strip_prefix("0x").and_then(|h| u64::from_str_radix(h, 16).ok());
            *b = match t.scalar {
                Scalar::F32 => match hex {
                    Some(h) => h,
                    None => match p.parse::<f32>() {
                        Ok(x) => u64::from(x.to_bits()),
                        Err(_) => return self.fail(format!("not an f32: {p}")),
                    },
                },
                Scalar::F64 => match hex {
                    Some(h) => h,
                    None => match p.parse::<f64>() {
                        Ok(x) => x.to_bits(),
                        Err(_) => return self.fail(format!("not an f64: {p}")),
                    },
                },
                Scalar::Bool => match *p {
                    "true" => 1,
                    "false" => 0,
                    _ => return self.fail(format!("not a bool: {p}")),
                },
                sc if sc.is_signed() => match p.parse::<i64>() {
                    // kept in the low bits of the word, as the builder keeps them
                    Ok(x) => x.cast_unsigned() & (u64::MAX >> 64u32.saturating_sub(sc.bits())),
                    Err(_) => return self.fail(format!("not an integer: {p}")),
                },
                _ => match p.parse::<u64>() {
                    Ok(x) => x,
                    Err(_) => return self.fail(format!("not an integer: {p}")),
                },
            };
        }
        Ok(Op::Literal(bits))
    }

    fn op(&self, t: Type, s: &str) -> Result<Op> {
        let s = s.trim();
        if s.starts_with(&format!("{t}(")) {
            return self.literal(t, s);
        }
        match s {
            "undef" => return Ok(Op::Undef),
            "param" => return Ok(Op::Param),
            _ => {}
        }
        if let Some(rest) = s.strip_prefix('%') {
            // a swizzle: `%n.lanes`, in the result's lane count
            let Some((v, names)) = rest.split_once('.') else {
                return self.fail(format!("not an operation: {s}"));
            };
            let v = self.value_id(&format!("%{v}"))?;
            let mut lanes = [0u8; 4];
            if names.len() != usize::from(t.lanes) {
                return self.fail(format!("a swizzle of {} lanes to {t}", names.len()));
            }
            for (l, c) in lanes.iter_mut().zip(names.chars()) {
                *l = lane_of(c).map_or_else(|| self.fail(format!("not a lane: {c}")), Ok)?;
            }
            return Ok(Op::Swizzle(v, lanes));
        }
        let (word, rest) = s.split_once(' ').unwrap_or((s, ""));
        Ok(match word {
            "select" => match self.list(rest)?.as_slice() {
                [c, a, b] => Op::Select {
                    cond: *c,
                    accept: *a,
                    reject: *b,
                },
                _ => return self.fail("select takes three values"),
            },
            "compose" => Op::Compose(self.list(rest)?),
            "splat" => Op::Splat(self.value_id(rest)?),
            "dot" => match self.list(rest)?.as_slice() {
                [a, b] => Op::Dot(*a, *b),
                _ => return self.fail("dot takes two values"),
            },
            "convert.saturate" => Op::Convert(self.value_id(rest)?, FloatToInt::Saturate),
            "convert.undef" => Op::Convert(self.value_id(rest)?, FloatToInt::Undef),
            "bitcast" => Op::Bitcast(self.value_id(rest)?),
            "input" | "output" => match rest.split_once(' ') {
                Some((at, index)) => {
                    let (slot, mask) = self.slot(at)?;
                    let index = self.bracketed(index)?;
                    if word == "input" {
                        Op::LoadInputAt(slot, mask, index)
                    } else {
                        Op::LoadOutputAt(slot, mask, index)
                    }
                }
                None if word == "input" => {
                    let (slot, mask) = self.slot(rest)?;
                    Op::LoadInput(slot, mask)
                }
                None => return self.fail("an output read without its control point"),
            },
            "patch" => {
                let (slot, mask) = self.slot(rest)?;
                Op::LoadPatch(slot, mask)
            }
            "builtin" => match BUILTINS.into_iter().find(|b| builtin(*b) == rest) {
                Some(b) => Op::Builtin(b),
                None => return self.fail(format!("no builtin {rest}")),
            },
            "load" => match rest.split_once('[') {
                Some((v, tail)) => {
                    let Some((i, rule)) = tail.split_once("] ") else {
                        return self.fail("an element load without its rule");
                    };
                    Op::LoadElement(self.var(v)?, self.value_id(i)?, self.oob(rule)?)
                }
                None => Op::LoadVar(self.var(rest)?),
            },
            "bitinsert" => match self.list(rest)?.as_slice() {
                [base, insert, offset, count] => Op::BitInsert {
                    base: *base,
                    insert: *insert,
                    offset: *offset,
                    count: *count,
                },
                _ => return self.fail("bitinsert takes four values"),
            },
            "texel" => self.texel(rest)?,
            name if name.starts_with("gather.") => self.gather(name.trim_start_matches("gather."), rest)?,
            name if name.starts_with("query.") => self.query(name.trim_start_matches("query."), rest)?,
            name if name.starts_with("atomic.") => self.atomic(name.trim_start_matches("atomic."), rest)?,
            name if name.starts_with("wave.") => self.wave(name, rest)?,
            "reporthit" => {
                let (t, keys) = keywords(rest, &["kind", "attributes"]);
                let found = |name: &str| keys.iter().find(|(k, _)| *k == name).map(|(_, x)| *x);
                let (Some(kind), Some(attributes)) = (found("kind"), found("attributes")) else {
                    return self.fail("a reported hit without its kind and attributes");
                };
                Op::ReportHit {
                    t: self.value_id(t)?,
                    kind: self.value_id(kind)?,
                    attributes: self.var(attributes)?,
                }
            }
            "counter.increment" => Op::Counter(self.res(rest)?, CounterOp::Increment),
            "counter.decrement" => Op::Counter(self.res(rest)?, CounterOp::Decrement),
            "state" => Op::LoadState(self.resource(rest)?),
            "buffer" => self.buffer(rest)?,
            "sample" => self.sample(rest)?,
            name if name.starts_with("clamp.") => Op::Unary(self.clamp(name)?, self.value_id(rest)?),
            name => {
                if let Some(t) = TERNARIES.into_iter().find(|t| ternary(t) == name) {
                    match self.list(rest)?.as_slice() {
                        [a, b, c] => Op::Ternary(t, *a, *b, *c),
                        _ => return self.fail(format!("{name} takes three values")),
                    }
                } else if let Some(u) = unaries().into_iter().find(|u| unary(u) == name) {
                    Op::Unary(u, self.value_id(rest)?)
                } else if let Some(b) = binaries().into_iter().find(|b| binary(b) == name) {
                    match self.list(rest)?.as_slice() {
                        [a, c] => Op::Binary(b, *a, *c),
                        _ => return self.fail(format!("{name} takes two values")),
                    }
                } else {
                    return self.fail(format!("no operation {name}"));
                }
            }
        })
    }
    /// `clamp.nan(lo,hi)`, the bounds as the printer writes floats.
    fn clamp(&self, name: &str) -> Result<Unary> {
        let parsed = name
            .strip_prefix("clamp.")
            .and_then(|r| r.strip_suffix(')'))
            .and_then(|r| r.split_once('('))
            .and_then(|(nan, bounds)| {
                let nan = match nan {
                    "zero" => SaturateNan::Zero,
                    "any" => SaturateNan::Any,
                    _ => return None,
                };
                let (lo, hi) = bounds.split_once(',')?;
                Some((nan, lo.parse::<f32>().ok()?, hi.parse::<f32>().ok()?))
            });
        match parsed {
            Some((nan, lo, hi)) => Ok(Unary::ClampRange {
                lo: lo.to_bits(),
                hi: hi.to_bits(),
                nan,
            }),
            None => self.fail(format!("not a clamp: {name}")),
        }
    }
    /// `rN[%i * stride] + offset oob.rule`, the index part optional.
    fn buffer(&self, s: &str) -> Result<Op> {
        let Some((head, oob)) = s.rsplit_once(" oob.") else {
            return self.fail("a buffer load without oob");
        };
        let oob = match oob {
            "zero" => OutOfBounds::Zero,
            "undef" => OutOfBounds::Undef,
            _ => return self.fail(format!("no out-of-bounds rule {oob}")),
        };
        let Some((head, offset)) = head.rsplit_once(" + ") else {
            return self.fail("a buffer load without offset");
        };
        let offset = offset
            .parse::<u32>()
            .map_or_else(|_| self.fail(format!("not an offset: {offset}")), Ok)?;
        let (resource, index, stride) = match head.split_once('[') {
            Some((r, idx)) => {
                let Some((i, stride)) = idx.strip_suffix(']').and_then(|x| x.split_once(" * ")) else {
                    return self.fail("a buffer index without a stride");
                };
                let stride = stride
                    .parse::<u32>()
                    .map_or_else(|_| self.fail(format!("not a stride: {stride}")), Ok)?;
                (self.res(r)?, Some(self.value_id(i)?), stride)
            }
            // a fixed address states no stride: the printer writes none, so it is zero
            None => (self.res(head)?, None, 0),
        };
        Ok(Op::LoadBuffer {
            resource,
            address: Address {
                index,
                stride,
                offset,
            },
            oob,
        })
    }
    /// `rT, rS, %c` and an optional `level 0`, `bias %x`, `level %x` or `grad %x, %y`, then optional `compare %x`,
    /// `offset (a,b,c)` and `clamp %x`.
    fn sample(&self, s: &str) -> Result<Op> {
        let (head, keys) = keywords(s, &["level", "bias", "grad", "compare", "offset", "clamp"]);
        let parts: Vec<&str> = head.split(", ").collect();
        let [texture, sampler, coord] = parts.as_slice() else {
            return self.fail("sample takes texture, sampler, coordinate");
        };
        let (mut level, mut compare, mut offset, mut clamp) = (Level::Auto, None, [0; 3], None);
        for (k, x) in keys {
            match k {
                "level" if x == "0" => level = Level::Zero,
                "level" => level = Level::Exact(self.value_id(x)?),
                "bias" => level = Level::Bias(self.value_id(x)?),
                "grad" => match self.list(x)?.as_slice() {
                    [dx, dy] => level = Level::Grad(*dx, *dy),
                    _ => return self.fail("a gradient takes two values"),
                },
                "compare" => compare = Some(self.value_id(x)?),
                "offset" => offset = self.offsets(x)?,
                _ => clamp = Some(self.value_id(x)?),
            }
        }
        Ok(Op::Sample {
            texture: self.res(texture)?,
            sampler: self.res(sampler)?,
            coord: self.value_id(coord)?,
            level,
            compare,
            offset,
            clamp,
        })
    }
    /// `gather.c rT, rS, %c` then optional `compare %x` and `offset (a,b,c)` or `offset %v`.
    fn gather(&self, component: &str, s: &str) -> Result<Op> {
        let component = match component.chars().next().and_then(lane_of) {
            Some(c) if component.len() == 1 => c,
            _ => return self.fail(format!("not a lane: {component}")),
        };
        let (head, keys) = keywords(s, &["compare", "offset"]);
        let parts: Vec<&str> = head.split(", ").collect();
        let [texture, sampler, coord] = parts.as_slice() else {
            return self.fail("gather takes texture, sampler, coordinate");
        };
        let (mut compare, mut offset) = (None, GatherOffset::Fixed([0; 3]));
        for (k, x) in keys {
            match k {
                "compare" => compare = Some(self.value_id(x)?),
                _ if x.starts_with('%') => offset = GatherOffset::Dynamic(self.value_id(x)?),
                _ => offset = GatherOffset::Fixed(self.offsets(x)?),
            }
        }
        Ok(Op::Gather {
            texture: self.res(texture)?,
            sampler: self.res(sampler)?,
            coord: self.value_id(coord)?,
            component,
            compare,
            offset,
        })
    }
    /// `texel rT, %c` then optional `level %l`, `sample %s`, `offset (a,b,c)`.
    fn texel(&self, s: &str) -> Result<Op> {
        let (head, keys) = keywords(s, &["level", "sample", "offset"]);
        let Some((texture, coord)) = head.split_once(", ") else {
            return self.fail("texel takes a resource and a coordinate");
        };
        let (mut level, mut sample, mut offset) = (None, None, [0; 3]);
        for (k, x) in keys {
            match k {
                "level" => level = Some(self.value_id(x)?),
                "sample" => sample = Some(self.value_id(x)?),
                _ => offset = self.offsets(x)?,
            }
        }
        Ok(Op::LoadTexel {
            texture: self.res(texture)?,
            coord: self.value_id(coord)?,
            level,
            sample,
            offset,
        })
    }
    /// A wave operation: its name (as [`super::wave`] prints it) and its values.
    fn wave(&self, name: &str, s: &str) -> Result<Op> {
        let values = self.list(s)?;
        let one = |w: fn(ValueId) -> Wave| match values.as_slice() {
            [a] => Ok(Op::Wave(w(*a))),
            _ => self.fail(format!("{name} takes one value")),
        };
        let two = |w: fn(ValueId, ValueId) -> Wave| match values.as_slice() {
            [a, b] => Ok(Op::Wave(w(*a, *b))),
            _ => self.fail(format!("{name} takes two values")),
        };
        match name {
            "wave.ballot" => return one(Wave::Ballot),
            "wave.first" => return one(Wave::First),
            "wave.lane" => return two(Wave::Lane),
            "wave.quadlane" => return two(Wave::QuadLane),
            _ => {}
        }
        let value = match values.as_slice() {
            [a] => *a,
            _ => return self.fail(format!("{name} takes one value")),
        };
        if let Some(s) = QUAD_SWAPS
            .into_iter()
            .find(|s| super::wave(&Wave::QuadSwap(value, *s)) == name)
        {
            return Ok(Op::Wave(Wave::QuadSwap(value, s)));
        }
        for op in WAVE_OPS {
            for scan in WAVE_SCANS {
                let w = Wave::Collective { op, scan, value };
                if super::wave(&w) == name {
                    return Ok(Op::Wave(w));
                }
            }
        }
        self.fail(format!("no wave operation {name}"))
    }
    fn query(&self, what: &str, s: &str) -> Result<Op> {
        let (head, keys) = keywords(s, &["level", "index"]);
        let r = self.res(head)?;
        let arg = keys.first().map(|(_, x)| self.value_id(x)).transpose()?;
        let q = match (what, arg) {
            ("size", l) => Query::Size(l),
            ("layers", None) => Query::Layers,
            ("levels", None) => Query::Levels,
            ("samples", None) => Query::Samples,
            ("sampleposition", Some(i)) => Query::SamplePosition(i),
            ("elements", None) => Query::Elements,
            _ => return self.fail(format!("no query {what}")),
        };
        Ok(Op::Query(r, q))
    }
    fn atomic(&self, name: &str, s: &str) -> Result<Op> {
        let Some(op) = ATOMICS
            .into_iter()
            .find(|a| format!("{a:?}").to_lowercase() == name)
        else {
            return self.fail(format!("no atomic operation {name}"));
        };
        let (head, keys) = keywords(s, &["value", "compare"]);
        let mut value = None;
        let mut compare = None;
        for (k, x) in keys {
            match k {
                "value" => value = Some(self.value_id(x)?),
                _ => compare = Some(self.value_id(x)?),
            }
        }
        let Some(value) = value else {
            return self.fail("an atomic operation without a value");
        };
        Ok(Op::Atomic {
            memory: self.memory(head)?,
            op,
            value,
            compare,
        })
    }

    /// A node line (not a construct's), or `None` for a line that is not one.
    fn plain(&mut self, line: &str) -> Result<Option<Node>> {
        if let Some(rest) = line.strip_prefix('%') {
            let Some((v, rest)) = rest.split_once(": ") else {
                return self.fail("a value without a type");
            };
            let v = self.value_id(&format!("%{v}"))?;
            let Some((ty, op)) = rest.split_once(" = ") else {
                return self.fail("a value without an operation");
            };
            let (ty, precision) = match ty.strip_suffix(" pp") {
                Some(t) => (t, Precision::AtLeast16),
                None => (ty, Precision::Full),
            };
            let ty = self.ty(ty)?;
            let op = self.op(ty, op)?;
            self.define(
                v,
                ValueDef {
                    ty,
                    op,
                    origin: None,
                    precision,
                },
            )?;
            return Ok(Some(Node::Value(v)));
        }
        if line == "demote" {
            return Ok(Some(Node::Demote));
        }
        if let Some(rest) = line.strip_prefix("output ") {
            let Some((at, v)) = rest.split_once(" = ") else {
                return self.fail("an output store without a value");
            };
            let Some((slot, lanes)) = at.split_once('.') else {
                return self.fail("an output store without lanes");
            };
            let slot = slot
                .parse::<u32>()
                .map_or_else(|_| self.fail(format!("not a slot: {slot}")), Ok)?;
            return Ok(Some(Node::StoreOutput(
                slot,
                self.mask(lanes)?,
                self.value_id(v)?,
            )));
        }
        if let Some(rest) = line.strip_prefix("store buffer ") {
            let Some((at, v)) = rest.rsplit_once(" = ") else {
                return self.fail("a buffer store without a value");
            };
            let (resource, address) = self.res_address(at)?;
            return Ok(Some(Node::StoreBuffer {
                resource,
                address,
                value: self.value_id(v)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("store vertex ") {
            let Some((at, v)) = rest.split_once(" = ") else {
                return self.fail("a vertex store without a value");
            };
            let Some((vertex, at)) = at.split_once(' ') else {
                return self.fail("a vertex store without its output");
            };
            let (slot, mask) = self.slot(at)?;
            return Ok(Some(Node::StoreVertex {
                vertex: self.value_id(vertex)?,
                slot,
                mask,
                value: self.value_id(v)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("store indices ") {
            let Some((primitive, v)) = rest.split_once(" = ") else {
                return self.fail("an index store without a value");
            };
            return Ok(Some(Node::StoreIndices {
                primitive: self.value_id(primitive)?,
                indices: self.value_id(v)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("dispatch mesh ") {
            let Some((groups, payload)) = rest.split_once(", ") else {
                return self.fail("a mesh launch without its payload");
            };
            return Ok(Some(Node::DispatchMesh {
                groups: self.value_id(groups)?,
                payload: self.var(payload)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("trace ray ") {
            let names = [
                "flags",
                "mask",
                "offset",
                "stride",
                "miss",
                "origin",
                "tmin",
                "direction",
                "tmax",
                "payload",
            ];
            let (scene, keys) = keywords(rest, &names);
            let found = |name: &str| keys.iter().find(|(k, _)| *k == name).map(|(_, x)| *x);
            let value = |name: &str| match found(name) {
                Some(x) => self.value_id(x),
                None => self.fail(format!("a ray without its {name}")),
            };
            let Some(payload) = found("payload") else {
                return self.fail("a ray without its payload");
            };
            return Ok(Some(Node::TraceRay {
                scene: self.res(scene)?,
                ray: Ray {
                    flags: value("flags")?,
                    mask: value("mask")?,
                    hit_group_offset: value("offset")?,
                    hit_group_stride: value("stride")?,
                    miss_index: value("miss")?,
                    origin: value("origin")?,
                    t_min: value("tmin")?,
                    direction: value("direction")?,
                    t_max: value("tmax")?,
                },
                payload: self.var(payload)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("call shader ") {
            let Some((index, parameter)) = rest.split_once(" parameter ") else {
                return self.fail("a shader call without its parameter");
            };
            return Ok(Some(Node::CallShader {
                index: self.value_id(index)?,
                parameter: self.var(parameter)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("mesh counts ") {
            let Some((vertices, primitives)) = rest.split_once(", ") else {
                return self.fail("mesh counts without both counts");
            };
            return Ok(Some(Node::SetMeshCounts {
                vertices: self.value_id(vertices)?,
                primitives: self.value_id(primitives)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("store texel ") {
            let Some((at, v)) = rest.rsplit_once(" = ") else {
                return self.fail("a texel store without a value");
            };
            let Some((r, c)) = at.split_once(", ") else {
                return self.fail("a texel store without a coordinate");
            };
            return Ok(Some(Node::StoreTexel {
                image: self.res(r)?,
                coord: self.value_id(c)?,
                value: self.value_id(v)?,
            }));
        }
        if let Some(rest) = line.strip_prefix("patch ") {
            let Some((at, v)) = rest.split_once(" = ") else {
                return self.fail("a patch constant store without a value");
            };
            let (slot, mask) = self.slot(at)?;
            return Ok(Some(Node::StorePatch(slot, mask, self.value_id(v)?)));
        }
        if let Some(rest) = line.strip_prefix("barrier") {
            let mut b = Barrier::default();
            for w in rest.split_whitespace() {
                match w {
                    "execution" => b.execution = true,
                    "workgroup" => b.workgroup = true,
                    "storage.workgroup" => b.storage = Some(Scope::Workgroup),
                    "storage.device" => b.storage = Some(Scope::Device),
                    _ => return self.fail(format!("not a barrier: {line}")),
                }
            }
            return Ok(Some(Node::Barrier(b)));
        }
        if let Some(n) = line.strip_prefix("emit ") {
            return Ok(Some(Node::Emit(self.number(n)?)));
        }
        if let Some(n) = line.strip_prefix("cut ") {
            return Ok(Some(Node::Cut(self.number(n)?)));
        }
        if let Some((name, v)) = line.split_once(" = ") {
            if let Some((var, i)) = name.split_once('[') {
                let Some((v, rule)) = v.split_once(' ') else {
                    return self.fail("an element store without its rule");
                };
                return Ok(Some(Node::StoreElement(
                    self.var(var)?,
                    self.bracketed(&format!("[{i}"))?,
                    self.value_id(v)?,
                    self.oob(rule)?,
                )));
            }
            return Ok(Some(Node::StoreVar(self.var(name)?, self.value_id(v)?)));
        }
        Ok(None)
    }

    /// A region's end line, or `None` for a line that is not one.
    fn end(&self, line: &str) -> Result<Option<End>> {
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        Ok(Some(match word {
            "yield" => End::Yield(self.list(rest)?),
            "br" | "continue" => {
                let (l, vs) = rest.split_once(' ').unwrap_or((rest, ""));
                let (l, vs) = (self.label(l)?, self.list(vs)?);
                if word == "br" {
                    End::Br(l, vs)
                } else {
                    End::Continue(l, vs)
                }
            }
            "return" if rest.is_empty() => End::Return,
            "kill" if rest.is_empty() => End::Kill,
            "ignorehit" if rest.is_empty() => End::IgnoreHit,
            "accepthit" if rest.is_empty() => End::AcceptHitAndEndSearch,
            "unreachable" if rest.is_empty() => End::Unreachable,
            _ => return Ok(None),
        }))
    }

    /// A construct's closing line: `}` or `} -> %n: T, ...`.
    fn close(&mut self) -> Result<Vec<ValueId>> {
        let line = self.next()?;
        match line.strip_prefix('}') {
            Some("") => Ok(Vec::new()),
            Some(rest) => match rest.strip_prefix(" -> ") {
                Some(ps) => self.params(ps),
                None => self.fail(format!("not a closing line: {line}")),
            },
            None => self.fail(format!("not a closing line: {line}")),
        }
    }

    fn region(&mut self) -> Result<Region> {
        let mut r = Region::default();
        loop {
            let line = self.next()?;
            if let Some(end) = self.end(line)? {
                r.end = Some(end);
                return Ok(r);
            }
            if let Some(c) = line.strip_prefix("if ").and_then(|x| x.strip_suffix(" {")) {
                let cond = self.value_id(c)?;
                let accept = self.region()?;
                if self.next()? != "} else {" {
                    return self.fail("an if without its else");
                }
                let reject = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                });
            } else if let Some(l) = line.strip_prefix("block ").and_then(|x| x.strip_suffix(" {")) {
                let label = self.label(l)?;
                let body = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::Block { label, body, results });
            } else if let Some(rest) = line.strip_prefix("loop ").and_then(|x| x.strip_suffix(") {")) {
                let Some((l, rest)) = rest.split_once(" (") else {
                    return self.fail("a loop without parameters");
                };
                let Some((ps, inits)) = rest.split_once(") = (") else {
                    return self.fail("a loop without initial values");
                };
                let label = self.label(l)?;
                let params = self.params(ps)?;
                let inits = self.list(inits)?;
                let body = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                });
            } else if let Some(rest) = line.strip_prefix("switch ").and_then(|x| x.strip_suffix(" {")) {
                let Some((l, sel)) = rest.split_once(' ') else {
                    return self.fail("a switch without a selector");
                };
                let (label, selector) = (self.label(l)?, self.value_id(sel)?);
                let mut cases = Vec::new();
                let default = loop {
                    let head = self.next()?;
                    if head == "default:" {
                        break self.region()?;
                    }
                    let Some(values) = head.strip_prefix("case ").and_then(|x| x.strip_suffix(':')) else {
                        return self.fail(format!("not a case: {head}"));
                    };
                    let values = values
                        .split(", ")
                        .map(|v| self.number(v))
                        .collect::<Result<Vec<u32>>>()?;
                    cases.push(Case {
                        values,
                        body: self.region()?,
                    });
                };
                let results = self.close()?;
                r.nodes.push(Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                });
            } else if let Some(n) = self.plain(line)? {
                r.nodes.push(n);
            } else {
                return self.fail(format!("not a node: {line}"));
            }
        }
    }

    fn target(&self, s: &str) -> Result<Target> {
        let Some((b, args)) = s
            .trim()
            .strip_prefix('b')
            .and_then(|x| x.strip_suffix(')'))
            .and_then(|x| x.split_once('('))
        else {
            return self.fail(format!("not a target: {s}"));
        };
        let block = b
            .parse::<u32>()
            .map_or_else(|_| self.fail(format!("not a block: {b}")), Ok)?;
        Ok(Target {
            block,
            args: self.list(args)?,
        })
    }

    fn graph(&mut self) -> Result<Vec<Block>> {
        let mut blocks = Vec::new();
        while let Some(header) = self.peek() {
            if header == "}" {
                break;
            }
            self.next()?;
            let want = format!("b{}(", blocks.len());
            let Some(ps) = header.strip_prefix(&want).and_then(|x| x.strip_suffix("):")) else {
                return self.fail(format!("not block {}'s header: {header}", blocks.len()));
            };
            let params = self.params(ps)?;
            let mut nodes = Vec::new();
            let term = loop {
                let line = self.next()?;
                if let Some(t) = line.strip_prefix("jump ") {
                    break Terminator::Jump(self.target(t)?);
                }
                if let Some(rest) = line.strip_prefix("branch ") {
                    let Some((c, rest)) = rest.split_once(", ") else {
                        return self.fail("a branch without targets");
                    };
                    let Some((a, b)) = rest.split_once("), ") else {
                        return self.fail("a branch without two targets");
                    };
                    break Terminator::Branch {
                        cond: self.value_id(c)?,
                        accept: self.target(&format!("{a})"))?,
                        reject: self.target(b)?,
                    };
                }
                if let Some(rest) = line.strip_prefix("switch ") {
                    let Some((sel, rest)) = rest.split_once(", ") else {
                        return self.fail("a switch without targets");
                    };
                    let mut cases = Vec::new();
                    let mut default = None;
                    for part in rest.split("), ") {
                        let part = if part.ends_with(')') {
                            part.to_string()
                        } else {
                            format!("{part})")
                        };
                        let Some((v, t)) = part.split_once(": ") else {
                            return self.fail(format!("not a switch target: {part}"));
                        };
                        if v == "default" {
                            default = Some(self.target(t)?);
                        } else {
                            cases.push((self.number(v)?, self.target(t)?));
                        }
                    }
                    let Some(default) = default else {
                        return self.fail("a switch without its default");
                    };
                    break Terminator::Switch {
                        selector: self.value_id(sel)?,
                        cases,
                        default,
                    };
                }
                match line {
                    "return" => break Terminator::Return,
                    "kill" => break Terminator::Kill,
                    "ignorehit" => break Terminator::IgnoreHit,
                    "accepthit" => break Terminator::AcceptHitAndEndSearch,
                    "unreachable" => break Terminator::Unreachable,
                    _ => {}
                }
                match self.plain(line)? {
                    Some(n) => nodes.push(n),
                    None => return self.fail(format!("not a node: {line}")),
                }
            };
            blocks.push(Block { params, nodes, term });
        }
        Ok(blocks)
    }
}

/// The function a text form states. Refused, with its line: a text without the schema line this version writes,
/// or with anything [`print`](super::print) does not write.
pub fn parse(text: &str) -> core::result::Result<Function, TextError> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i.saturating_add(1), l.trim()))
        .filter(|(_, l)| !l.is_empty())
        .collect();
    let mut p = Parser {
        lines,
        at: 0,
        values: Vec::new(),
        vars: Vec::new(),
        var_names: BTreeMap::new(),
    };
    if p.next()? != SCHEMA {
        return p.fail(format!("the first line is not {SCHEMA:?}"));
    }
    let header = p.next()?;
    let Some(name) = header.strip_prefix("fn ").and_then(|x| x.strip_suffix(" {")) else {
        return p.fail("not a function header");
    };
    while let Some(rest) = p.peek().and_then(|l| l.strip_prefix("var ")) {
        p.next()?;
        let Some((n, t)) = rest.split_once(": ") else {
            return p.fail("a variable without a type");
        };
        let (t, init) = match t.split_once(" = {") {
            Some((t, init)) => (t, init.strip_suffix('}')),
            None => (t, None),
        };
        let suffixes = [
            (" workgroup", Space::Workgroup),
            (" payload", Space::Payload),
            (" raypayload", Space::RayPayload),
            (" attributes", Space::HitAttributes),
        ];
        let (t, space) = suffixes
            .into_iter()
            .find_map(|(suffix, space)| Some((t.strip_suffix(suffix)?, space)))
            .unwrap_or((t, Space::Private));
        let (ty, len) = match t.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            Some(inner) => {
                let Some((t, n)) = inner.split_once("; ") else {
                    return p.fail("an array without a length");
                };
                (p.ty(t)?, Some(p.number(n)?))
            }
            None => (p.ty(t)?, None),
        };
        let mut elements = Vec::new();
        if let Some(init) = init {
            // each element is a literal `T(..)`; the literals' own commas are inside their parentheses
            for e in init.split("), ") {
                let e = if e.ends_with(')') {
                    e.to_string()
                } else {
                    format!("{e})")
                };
                match p.literal(ty, &e)? {
                    Op::Literal(bits) => elements.push(bits),
                    _ => return p.fail("not a literal"),
                }
            }
        }
        let id = VarId(u32::try_from(p.vars.len()).map_or_else(|_| p.fail("more than 2^32 variables"), Ok)?);
        if p.var_names.insert(n.to_string(), id).is_some() {
            return p.fail(format!("variable {n} is declared twice"));
        }
        p.vars.push(Var {
            name: n.to_string(),
            ty,
            len,
            space,
            init: elements,
        });
    }
    let graph = p
        .peek()
        .is_some_and(|l| l.starts_with("b0(") && l.ends_with("):"));
    let body = if graph {
        Body::Cfg(p.graph()?)
    } else {
        Body::Regions(p.region()?)
    };
    if p.next()? != "}" {
        return p.fail("the function does not end after its body");
    }
    if let Some(extra) = p.peek() {
        p.at = p.at.saturating_add(1);
        return p.fail(format!("text after the function: {extra}"));
    }
    let undef = ValueDef {
        ty: Type::BOOL,
        op: Op::Undef,
        origin: None,
        precision: Precision::Full,
    };
    let values = p
        .values
        .into_iter()
        .map(|v| v.unwrap_or_else(|| undef.clone()))
        .collect();
    Ok(Function {
        name: name.to_string(),
        values,
        vars: p.vars,
        body,
    })
}
