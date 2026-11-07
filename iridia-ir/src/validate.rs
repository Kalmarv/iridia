//! The validator: types of every operation, scoping of values in regions and dominance in a graph, exits against
//! their constructs, interface and resource references against the interface, and stage rules.

use alloc::{borrow::ToOwned, format, string::String, vec, vec::Vec};
use core::fmt;

use crate::{
    AtomicOp, Barrier, Binary, Block, Body, Builtin, End, Function, GatherOffset, LabelId, Level, Memory,
    Node, Op, Program, Query, Region, Res, Scalar, Terminator, Ternary, Type, Unary, ValueId, VarId, Wave,
    WaveOp,
    interface::{
        BufferKind, Interface, Resource, ResourceKind, SampleType, ScalarType, Stage, StageFacts,
        StorageTarget, TextureDim, Value, Varying,
    },
};

/// One broken rule, with the value it is about where there is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The value the rule is about.
    pub value: Option<ValueId>,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            Some(v) => write!(f, "%{}: {}", v.0, self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// The IR type of an interface value.
pub fn value_type(v: Value) -> Type {
    let scalar = match v.scalar {
        ScalarType::Bool => Scalar::Bool,
        ScalarType::Sint => Scalar::I32,
        ScalarType::Uint => Scalar::U32,
        ScalarType::Float => Scalar::F32,
        ScalarType::Double => Scalar::F64,
    };
    Type::vector(scalar, v.width)
}

/// The type of the lanes `mask` selects of an interface value: its scalar, one lane per bit. An error says why the
/// mask selects none of it or lanes it does not have.
pub fn masked_type(v: Value, mask: u8) -> Result<Type, String> {
    let all = value_type(v);
    let lanes = u8::try_from(mask.count_ones()).unwrap_or(0);
    if mask == 0 || lanes > all.lanes || mask >> all.lanes != 0 {
        return Err(format!("lane mask {mask:#06b} of a value of {all}"));
    }
    Ok(Type::vector(all.scalar, lanes))
}

fn same(a: Type, b: Type, what: &str) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("{what}: {a} and {b} differ"))
    }
}

/// The type an operation gives from its operands' types (`ty` looks them up), or `None` where the operation's type is
/// stated rather than derived (literals, parameters, conversions, bitcasts, loads, samples). An error names the rule
/// the operands break.
pub fn derived_type(op: &Op, ty: impl Fn(ValueId) -> Option<Type>) -> Result<Option<Type>, String> {
    let get = |v: ValueId| ty(v).ok_or_else(|| format!("operand %{} is not a value of the function", v.0));
    let float = |t: Type, what: &str| {
        if t.scalar.is_float() {
            Ok(())
        } else {
            Err(format!("{what} of {t}, which is not a float"))
        }
    };
    Ok(Some(match op {
        Op::Literal(_)
        | Op::Undef
        | Op::Param
        | Op::Convert(..)
        | Op::Bitcast(_)
        | Op::LoadInput(..)
        | Op::LoadInputAt(..)
        | Op::LoadOutputAt(..)
        | Op::LoadPatch(..)
        | Op::Builtin(_)
        | Op::LoadVar(_)
        | Op::LoadElement(..)
        | Op::LoadState(_)
        | Op::LoadBuffer { .. }
        | Op::Sample { .. }
        | Op::Gather { .. }
        | Op::LoadTexel { .. }
        | Op::Query(..)
        | Op::Atomic { .. }
        | Op::Counter(..) => return Ok(None),
        Op::ReportHit { t, kind, .. } => {
            same(get(*t)?, Type::F32, "a reported hit's distance")?;
            same(get(*kind)?, Type::U32, "a reported hit's kind")?;
            Type::BOOL
        }
        Op::Wave(w) => {
            // a value read from another invocation, or gathered over them, is a scalar or vector of numbers
            let numbers = |t: Type, what: &str| {
                if t.scalar == Scalar::Bool {
                    Err(format!("{what} of {t}, which is not a number"))
                } else {
                    Ok(t)
                }
            };
            match w {
                Wave::Ballot(c) => {
                    same(get(*c)?, Type::BOOL, "a ballot's condition")?;
                    Type::U32X4
                }
                Wave::First(a) | Wave::QuadSwap(a, _) => numbers(get(*a)?, "a wave read")?,
                Wave::Lane(a, l) | Wave::QuadLane(a, l) => {
                    index_type(get(*l)?, "a lane number")?;
                    numbers(get(*a)?, "a wave read")?
                }
                Wave::Collective { op, value, .. } => {
                    let t = get(*value)?;
                    match op {
                        WaveOp::All | WaveOp::Any => {
                            same(t, Type::BOOL, "a wave's all or any")?;
                            t
                        }
                        WaveOp::And | WaveOp::Or | WaveOp::Xor => {
                            if !t.scalar.is_integer() {
                                return Err(format!("a wave bit operation of {t}"));
                            }
                            t
                        }
                        WaveOp::Add | WaveOp::Mul | WaveOp::Min | WaveOp::Max => {
                            numbers(t, "a wave operation")?
                        }
                    }
                }
            }
        }
        Op::Unary(u, a) => {
            let t = get(*a)?;
            match u {
                Unary::Not => {
                    if t.scalar != Scalar::Bool && !t.scalar.is_integer() {
                        return Err(format!("not of {t}"));
                    }
                    t
                }
                Unary::INeg => {
                    if !t.scalar.is_integer() {
                        return Err(format!("integer negation of {t}"));
                    }
                    t
                }
                Unary::Any | Unary::All => {
                    if t.scalar != Scalar::Bool {
                        return Err(format!("any/all of {t}"));
                    }
                    Type::BOOL
                }
                Unary::IsNan | Unary::IsInf => {
                    float(t, "a float test")?;
                    t.with(Scalar::Bool)
                }
                Unary::CountOnes | Unary::BitReverse => {
                    if !t.scalar.is_integer() {
                        return Err(format!("a bit operation of {t}"));
                    }
                    t
                }
                Unary::FirstBitHigh | Unary::FirstBitHighSigned | Unary::FirstBitLow => {
                    if !t.scalar.is_integer() || t.scalar.bits() != 32 {
                        return Err(format!(
                            "a first-bit operation of {t}, which is not a 32-bit integer"
                        ));
                    }
                    t
                }
                Unary::F32ToF16 => {
                    if t.scalar != Scalar::F32 {
                        return Err(format!("f32tof16 of {t}"));
                    }
                    t.with(Scalar::U32)
                }
                Unary::F16ToF32 => {
                    if t.scalar != Scalar::U32 {
                        return Err(format!("f16tof32 of {t}"));
                    }
                    t.with(Scalar::F32)
                }
                Unary::ClampRange { lo, hi, nan } => {
                    let (l, h) = (f32::from_bits(*lo), f32::from_bits(*hi));
                    if !(l.is_finite() && h.is_finite() && l <= h) {
                        return Err(format!("a clamp to {l:?}..{h:?}, which is not a finite range"));
                    }
                    if *nan == crate::SaturateNan::Zero && !(l <= 0.0 && 0.0 <= h) {
                        return Err(format!("a clamp to {l:?}..{h:?} that gives 0 for NaN"));
                    }
                    float(t, "a clamp")?;
                    if t.scalar != Scalar::F32 {
                        return Err(format!("a clamp of {t}, whose bounds are f32"));
                    }
                    t
                }
                Unary::Quantize { frac_bits, .. } => {
                    if *frac_bits > 23 {
                        return Err(format!("a quantize to {frac_bits} fractional bits, more than 23"));
                    }
                    float(t, "a quantize")?;
                    t
                }
                _ => {
                    float(t, "a float operation")?;
                    t
                }
            }
        }
        Op::Binary(Binary::Shl | Binary::ShrU | Binary::ShrS, x, y) => {
            let (a, c) = (get(*x)?, get(*y)?);
            if !a.scalar.is_integer() || !c.scalar.is_unsigned() || a.lanes != c.lanes {
                return Err(format!("a shift of {a} by {c}"));
            }
            a
        }
        Op::Binary(b, x, y) => {
            let (a, c) = (get(*x)?, get(*y)?);
            same(a, c, "binary operands")?;
            match b {
                Binary::FAdd
                | Binary::FSub
                | Binary::FMul(_)
                | Binary::FDiv
                | Binary::FRem
                | Binary::FMin(_)
                | Binary::FMax(_)
                | Binary::FPow => {
                    float(a, "a float operation")?;
                    a
                }
                Binary::FCmp(_) => {
                    float(a, "a float comparison")?;
                    a.with(Scalar::Bool)
                }
                Binary::SCmp(_) | Binary::UCmp(_) => {
                    if a.scalar.is_float() {
                        return Err(format!("an integer comparison of {a}"));
                    }
                    a.with(Scalar::Bool)
                }
                Binary::IAdd
                | Binary::ISub
                | Binary::IMul
                | Binary::UDiv(_)
                | Binary::URem(_)
                | Binary::SDiv(_)
                | Binary::SRem(_)
                | Binary::UMulHigh
                | Binary::SMulHigh
                | Binary::UAddCarry
                | Binary::USubBorrow
                | Binary::SMin
                | Binary::SMax
                | Binary::UMin
                | Binary::UMax
                | Binary::Xor => {
                    if !a.scalar.is_integer() {
                        return Err(format!("an integer operation of {a}"));
                    }
                    a
                }
                Binary::And | Binary::Or => {
                    if a.scalar.is_float() {
                        return Err(format!("a bitwise operation of {a}"));
                    }
                    a
                }
                Binary::Shl | Binary::ShrU | Binary::ShrS => unreachable!("matched above"),
            }
        }
        Op::Ternary(Ternary::Fma, x, y, z) => {
            let (a, b, c) = (get(*x)?, get(*y)?, get(*z)?);
            same(a, b, "fma operands")?;
            same(a, c, "fma operands")?;
            float(a, "a fused multiply-add")?;
            a
        }
        Op::Ternary(Ternary::BitExtract { .. }, x, offset, count) => {
            let a = get(*x)?;
            if !a.scalar.is_integer() || a.scalar.bits() != 32 {
                return Err(format!("a bit field of {a}, which is not a 32-bit integer"));
            }
            for c in [offset, count] {
                same(get(*c)?, a.with(Scalar::U32), "a bit field count")?;
            }
            a
        }
        Op::BitInsert {
            base,
            insert,
            offset,
            count,
        } => {
            let a = get(*base)?;
            same(get(*insert)?, a, "a bit field insert")?;
            if !a.scalar.is_integer() || a.scalar.bits() != 32 {
                return Err(format!("a bit field of {a}, which is not a 32-bit integer"));
            }
            for c in [offset, count] {
                same(get(*c)?, a.with(Scalar::U32), "a bit field count")?;
            }
            a
        }
        Op::Select { cond, accept, reject } => {
            let (c, a, r) = (get(*cond)?, get(*accept)?, get(*reject)?);
            same(a, r, "select arms")?;
            if c.scalar != Scalar::Bool || (c.lanes != 1 && c.lanes != a.lanes) {
                return Err(format!("select condition {c} for {a}"));
            }
            a
        }
        Op::Compose(parts) => {
            let mut scalar = None;
            let mut lanes: u32 = 0;
            for p in parts {
                let t = get(*p)?;
                if *scalar.get_or_insert(t.scalar) != t.scalar {
                    return Err("compose parts of different scalar types".into());
                }
                lanes = lanes.saturating_add(u32::from(t.lanes));
            }
            let lanes = u8::try_from(lanes).ok().filter(|n| (2..=4).contains(n));
            match (scalar, lanes) {
                (Some(s), Some(n)) => Type::vector(s, n),
                _ => return Err("compose of other than 2 to 4 lanes".into()),
            }
        }
        Op::Swizzle(a, lanes) => {
            let t = get(*a)?;
            if lanes.iter().any(|&l| l >= t.lanes) {
                return Err(format!("swizzle {lanes:?} of {t}"));
            }
            return Ok(None);
        }
        Op::Splat(a) => {
            let t = get(*a)?;
            if t.lanes != 1 {
                return Err(format!("splat of {t}"));
            }
            return Ok(None);
        }
        Op::Dot(x, y) => {
            let (a, b) = (get(*x)?, get(*y)?);
            same(a, b, "dot operands")?;
            float(a, "a dot product")?;
            if a.lanes < 2 {
                return Err("a dot product of scalars".into());
            }
            Type::scalar(a.scalar)
        }
    }))
}

/// The interface's resource of an id.
fn resource(interface: &Interface, r: Res) -> Result<&Resource, String> {
    interface
        .resources
        .iter()
        .find(|x| x.id == r.id)
        .ok_or_else(|| format!("resource {} is not in the interface", r.id.0))
}

/// The lanes of a texture coordinate without an array layer.
fn dims(dim: TextureDim) -> u8 {
    match dim {
        TextureDim::D1 => 1,
        TextureDim::D2 => 2,
        TextureDim::D3 | TextureDim::Cube => 3,
    }
}

/// The lanes of a texture's size: a cube's is its faces' width and height.
fn size_lanes(dim: TextureDim) -> u8 {
    match dim {
        TextureDim::D1 => 1,
        TextureDim::D2 | TextureDim::Cube => 2,
        TextureDim::D3 => 3,
    }
}

/// The IR scalar a texel of a sample type reads as.
fn sample_scalar(s: SampleType) -> Scalar {
    match s {
        SampleType::Float | SampleType::Depth => Scalar::F32,
        SampleType::Sint => Scalar::I32,
        SampleType::Uint => Scalar::U32,
    }
}

/// The type of a value the pipeline gives.
pub fn builtin_type(b: Builtin) -> Type {
    match b {
        Builtin::OutputControlPoint | Builtin::PrimitiveId | Builtin::GsInstance | Builtin::PhaseInstance => {
            Type::U32
        }
        Builtin::DomainLocation => Type::vector(Scalar::F32, 3),
    }
}

/// The patch constants of a hull or domain program.
pub fn patch_constants(interface: &Interface) -> &[Varying] {
    match &interface.facts {
        StageFacts::Hull(h) => &h.patch_constants,
        StageFacts::Domain(d) => &d.patch_constants,
        _ => &[],
    }
}

fn index_type(t: Type, what: &str) -> Result<(), String> {
    if t == Type::U32 || t == Type::I32 {
        Ok(())
    } else {
        Err(format!("{what} of {t}"))
    }
}

fn int_coord(t: Type, lanes: u8, what: &str) -> Result<(), String> {
    if (t.scalar == Scalar::U32 || t.scalar == Scalar::I32) && t.lanes == lanes {
        Ok(())
    } else {
        Err(format!("{what} of {t}, where {lanes} integer lanes are wanted"))
    }
}

/// Check an operation whose type is stated: the operands' types and the stated type agree with the rule.
fn stated(op: &Op, t: Type, f: &Function, interface: &Interface) -> Result<(), String> {
    let get = |v: ValueId| {
        f.ty(v)
            .ok_or_else(|| format!("operand %{} is not a value of the function", v.0))
    };
    let element = |r: Res| -> Result<(), String> {
        let res = resource(interface, r)?;
        match r.index {
            Some(i) => index_type(get(i)?, "a resource array index"),
            None if res.count == 0 => Err(format!("resource {} has no elements", res.provenance.name)),
            None => Ok(()),
        }
    };
    let var = |v: crate::VarId| {
        f.vars
            .get(v.0 as usize)
            .ok_or_else(|| format!("variable {} does not exist", v.0))
    };
    // a filtered read's texture, sampler and coordinate: the texture's dimensions (without a layer) and sample type
    let sampled = |texture: Res, sampler: Res, coord: ValueId, compare: Option<ValueId>| {
        element(texture)?;
        element(sampler)?;
        let ResourceKind::Texture {
            dim,
            arrayed,
            sample,
            multisampled,
        } = resource(interface, texture)?.kind
        else {
            return Err("sample of a resource that is not a texture".to_owned());
        };
        if multisampled {
            return Err("a sample of a multisampled texture".into());
        }
        if !matches!(resource(interface, sampler)?.kind, ResourceKind::Sampler { .. }) {
            return Err("sample through a resource that is not a sampler".into());
        }
        let n = dims(dim);
        same(
            get(coord)?,
            Type::vector(Scalar::F32, n.saturating_add(u8::from(arrayed))),
            "sample coordinate",
        )?;
        if let Some(c) = compare {
            same(get(c)?, Type::F32, "a comparison reference")?;
        }
        Ok((n, sample))
    };
    match op {
        Op::Literal(_) | Op::Undef | Op::Param => Ok(()),
        Op::Convert(a, _) => {
            let a = get(*a)?;
            if a.lanes != t.lanes || a.scalar == Scalar::Bool || t.scalar == Scalar::Bool {
                return Err(format!("conversion of {a} to {t}"));
            }
            Ok(())
        }
        Op::Bitcast(a) => {
            let a = get(*a)?;
            let bits = |t: Type| t.scalar.bits().saturating_mul(u32::from(t.lanes));
            if bits(a) != bits(t) || a.scalar == Scalar::Bool || t.scalar == Scalar::Bool {
                return Err(format!("bitcast of {a} to {t}"));
            }
            Ok(())
        }
        Op::Swizzle(a, _) => {
            let a = get(*a)?;
            if a.scalar == t.scalar {
                Ok(())
            } else {
                Err(format!("swizzle of {a} to {t}"))
            }
        }
        Op::Splat(a) => {
            if get(*a)? == Type::scalar(t.scalar) && t.lanes >= 2 {
                Ok(())
            } else {
                Err(format!("splat of {} to {t}", get(*a)?))
            }
        }
        Op::LoadInput(slot, mask) | Op::LoadInputAt(slot, mask, _) => {
            if let Op::LoadInputAt(_, _, at) = op {
                index_type(get(*at)?, "a vertex index")?;
            }
            let v = interface.inputs.get(*slot as usize).ok_or_else(|| {
                format!(
                    "input slot {slot}, where the interface has {}",
                    interface.inputs.len()
                )
            })?;
            same(masked_type(v.value, *mask)?, t, "input")
        }
        Op::LoadOutputAt(slot, mask, at) => {
            index_type(get(*at)?, "a control point index")?;
            let v = interface
                .outputs
                .get(*slot as usize)
                .ok_or_else(|| format!("output slot {slot} is not in the interface"))?;
            same(masked_type(v.value, *mask)?, t, "output control point")
        }
        Op::LoadPatch(slot, mask) => {
            let v = patch_constants(interface)
                .get(*slot as usize)
                .ok_or_else(|| format!("patch constant slot {slot} is not in the interface"))?;
            same(masked_type(v.value, *mask)?, t, "patch constant")
        }
        Op::Builtin(b) => same(builtin_type(*b), t, "a pipeline value"),
        Op::LoadVar(v) => {
            let d = var(*v)?;
            if d.len.is_some() {
                return Err(format!("a whole load of the array variable {}", d.name));
            }
            same(d.ty, t, "variable load")
        }
        Op::LoadElement(v, i, _) => {
            let d = var(*v)?;
            if d.len.is_none() {
                return Err(format!("an element load of {}, which is not an array", d.name));
            }
            index_type(get(*i)?, "an element index")?;
            same(d.ty, t, "element load")
        }
        Op::LoadState(resource) => {
            let r = interface
                .resources
                .iter()
                .find(|r| r.id == *resource)
                .ok_or_else(|| format!("resource {} is not in the interface", resource.0))?;
            match r.kind {
                ResourceKind::PipelineState(p) => same(value_type(p.value()), t, "state load"),
                _ => Err(format!("state load from {:?}", r.kind)),
            }
        }
        Op::LoadBuffer {
            resource: r, address, ..
        } => {
            element(*r)?;
            match &resource(interface, *r)?.kind {
                ResourceKind::ConstantBuffer { .. }
                | ResourceKind::Buffer(BufferKind::Raw | BufferKind::Structured { .. })
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(_),
                    ..
                } => {}
                k => return Err(format!("buffer load from {k:?}")),
            }
            if let Some(i) = address.index {
                index_type(get(i)?, "a buffer index")?;
            }
            if t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
                return Err(format!("buffer load of {t}"));
            }
            Ok(())
        }
        Op::Sample {
            texture,
            sampler,
            coord,
            level,
            compare,
            clamp,
            ..
        } => {
            let (n, sample) = sampled(*texture, *sampler, *coord, *compare)?;
            match level {
                Level::Bias(x) | Level::Exact(x) => same(get(*x)?, Type::F32, "sample level")?,
                Level::Grad(x, y) => {
                    let g = Type::vector(Scalar::F32, n);
                    same(get(*x)?, g, "sample gradient along x")?;
                    same(get(*y)?, g, "sample gradient along y")?;
                }
                Level::Auto | Level::Zero => {}
            }
            if let Some(c) = clamp {
                same(get(*c)?, Type::F32, "a level clamp")?;
            }
            let want = if compare.is_some() || sample == SampleType::Depth {
                Type::F32
            } else {
                Type::vector(sample_scalar(sample), 4)
            };
            same(t, want, "sample result")
        }
        Op::Gather {
            texture,
            sampler,
            coord,
            component,
            compare,
            offset,
        } => {
            let (n, sample) = sampled(*texture, *sampler, *coord, *compare)?;
            if *component > 3 {
                return Err(format!("a gather of lane {component}"));
            }
            if let GatherOffset::Dynamic(o) = offset {
                same(get(*o)?, Type::vector(Scalar::I32, n), "a gather offset")?;
            }
            let want = if compare.is_some() {
                Type::vector(Scalar::F32, 4)
            } else {
                Type::vector(sample_scalar(sample), 4)
            };
            same(t, want, "gather result")
        }
        Op::LoadTexel {
            texture,
            coord,
            level,
            sample,
            ..
        } => {
            element(*texture)?;
            let c = get(*coord)?;
            for x in level.iter().chain(sample.iter()) {
                index_type(get(*x)?, "a texel level or sample")?;
            }
            let want = match resource(interface, *texture)?.kind {
                ResourceKind::Texture {
                    dim,
                    arrayed,
                    multisampled,
                    sample: s,
                } => {
                    int_coord(
                        c,
                        dims(dim).saturating_add(u8::from(arrayed)),
                        "a texel coordinate",
                    )?;
                    if multisampled != sample.is_some() {
                        return Err("a texel load whose sample does not match the texture".into());
                    }
                    if s == SampleType::Depth {
                        Type::F32
                    } else {
                        Type::vector(sample_scalar(s), 4)
                    }
                }
                ResourceKind::Storage {
                    target:
                        StorageTarget::Image {
                            dim,
                            arrayed,
                            sample: s,
                        },
                    ..
                } => {
                    int_coord(
                        c,
                        dims(dim).saturating_add(u8::from(arrayed)),
                        "a texel coordinate",
                    )?;
                    Type::vector(sample_scalar(s), 4)
                }
                ResourceKind::Buffer(BufferKind::Typed(v))
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(v)),
                    ..
                } => {
                    int_coord(c, 1, "a buffer element")?;
                    Type::vector(value_type(v).scalar, 4)
                }
                ref k => return Err(format!("texel load from {k:?}")),
            };
            same(t, want, "texel load")
        }
        Op::Query(r, q) => {
            element(*r)?;
            let kind = &resource(interface, *r)?.kind;
            let image = match kind {
                ResourceKind::Texture {
                    dim,
                    arrayed,
                    multisampled,
                    ..
                } => Some((*dim, *arrayed, *multisampled, true)),
                ResourceKind::Storage {
                    target: StorageTarget::Image { dim, arrayed, .. },
                    ..
                } => Some((*dim, *arrayed, false, false)),
                _ => None,
            };
            let buffer = matches!(
                kind,
                ResourceKind::Buffer(_)
                    | ResourceKind::Storage {
                        target: StorageTarget::Buffer(_),
                        ..
                    }
            );
            match (q, image) {
                (Query::Size(level), Some((dim, _, ms, levels))) => {
                    if level.is_some() != (levels && !ms) {
                        return Err("a size query whose level does not match the resource".into());
                    }
                    if let Some(l) = level {
                        index_type(get(*l)?, "a size query's level")?;
                    }
                    same(t, Type::vector(Scalar::U32, size_lanes(dim)), "size query")
                }
                (Query::Layers, Some((_, true, _, _))) => same(t, Type::U32, "layer query"),
                (Query::Levels, Some((_, _, false, true))) => same(t, Type::U32, "level query"),
                (Query::Samples, Some((_, _, true, _))) => same(t, Type::U32, "sample query"),
                (Query::SamplePosition(i), Some((_, _, true, _))) => {
                    index_type(get(*i)?, "a sample index")?;
                    same(t, Type::vector(Scalar::F32, 2), "sample position")
                }
                (Query::Elements, None) if buffer => same(t, Type::U32, "element query"),
                (q, _) => Err(format!("{q:?} of {kind:?}")),
            }
        }
        Op::Atomic {
            memory,
            op: aop,
            value,
            compare,
        } => {
            let v = get(*value)?;
            if !v.scalar.is_integer() || v.lanes != 1 || !(v.scalar.bits() == 32 || v.scalar.bits() == 64) {
                return Err(format!("an atomic operation of {v}"));
            }
            same(t, v, "atomic result")?;
            if (*aop == AtomicOp::CompareExchange) != compare.is_some() {
                return Err("a comparand without a compare-exchange, or the reverse".into());
            }
            if let Some(c) = compare {
                same(get(*c)?, v, "a comparand")?;
            }
            match memory {
                Memory::Buffer(r, address) => {
                    element(*r)?;
                    if !matches!(
                        resource(interface, *r)?.kind,
                        ResourceKind::Storage {
                            target: StorageTarget::Buffer(_),
                            ..
                        }
                    ) {
                        return Err("an atomic operation on a resource that is not a storage buffer".into());
                    }
                    if let Some(i) = address.index {
                        index_type(get(i)?, "a buffer index")?;
                    }
                }
                Memory::Image(r, c) => {
                    element(*r)?;
                    let (lanes, scalar) = match resource(interface, *r)?.kind {
                        ResourceKind::Storage {
                            target: StorageTarget::Image { dim, arrayed, sample },
                            ..
                        } => (dims(dim).saturating_add(u8::from(arrayed)), sample_scalar(sample)),
                        ResourceKind::Storage {
                            target: StorageTarget::Buffer(BufferKind::Typed(e)),
                            ..
                        } => (1, value_type(e).scalar),
                        _ => {
                            return Err(
                                "an atomic operation on a resource that is not a storage image or typed buffer".into()
                            );
                        }
                    };
                    int_coord(get(*c)?, lanes, "a texel coordinate")?;
                    if scalar != v.scalar {
                        return Err(format!("an atomic operation of {v} on {} texels", scalar.name()));
                    }
                }
                Memory::Var(x, i) => {
                    let d = var(*x)?;
                    match d.space {
                        crate::Space::Payload => {
                            return Err(format!(
                                "an atomic operation on task payload {}, which is read alone",
                                d.name
                            ));
                        }
                        crate::Space::RayPayload | crate::Space::HitAttributes => {
                            return Err(format!(
                                "an atomic operation on {}, a ray tracing program's own memory",
                                d.name
                            ));
                        }
                        crate::Space::Private | crate::Space::Workgroup => {}
                    }
                    if d.len.is_some() != i.is_some() {
                        return Err(format!("an atomic operation on {} without its element", d.name));
                    }
                    if let Some(i) = i {
                        index_type(get(*i)?, "an element index")?;
                    }
                    same(d.ty, v, "an atomic variable")?;
                }
            }
            Ok(())
        }
        Op::Counter(r, _) => {
            element(*r)?;
            if !matches!(
                resource(interface, *r)?.kind,
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Structured { .. }),
                    ..
                }
            ) {
                return Err("a counter of a resource that is not a structured storage buffer".into());
            }
            same(t, Type::U32, "a counter")
        }
        _ => Ok(()),
    }
}

struct Checker<'a> {
    f: &'a Function,
    interface: &'a Interface,
    stage: Stage,
    visible: Vec<bool>,
    placed: Vec<bool>,
    labels: Vec<(LabelId, Vec<ValueId>, Option<Vec<ValueId>>)>,
    problems: Vec<Problem>,
    /// Checking a hull program's patch constant function.
    patch: bool,
}

impl Checker<'_> {
    fn problem(&mut self, value: Option<ValueId>, message: impl Into<String>) {
        self.problems.push(Problem {
            value,
            message: message.into(),
        });
    }
    fn ty(&self, v: ValueId) -> Option<Type> {
        self.f.ty(v)
    }
    /// A store into variable `var`: a task payload and a hit's attributes are read alone.
    fn writable(&mut self, var: VarId, value: ValueId) {
        let what = match self.f.vars.get(var.0 as usize) {
            Some(d) if d.space == crate::Space::Payload => format!("task payload {}", d.name),
            Some(d) if d.space == crate::Space::HitAttributes => format!("hit attributes {}", d.name),
            _ => return,
        };
        self.problem(Some(value), format!("a store to {what}, which is read alone"));
    }
    /// The memory a ray tracing operation hands over (`what`): a private or ray payload array of `u32` words.
    fn handed(&mut self, var: VarId, what: &str) {
        match self.f.vars.get(var.0 as usize) {
            None => self.problem(
                None,
                format!("{what} of variable {}, which does not exist", var.0),
            ),
            Some(d)
                if !matches!(d.space, crate::Space::Private | crate::Space::RayPayload)
                    || d.ty != Type::U32
                    || d.len.is_none() =>
            {
                let m = format!("{what} of {}, which is not a private array of u32 words", d.name);
                self.problem(None, m);
            }
            Some(_) => {}
        }
    }
    fn uses(&mut self, at: Option<ValueId>, v: ValueId) {
        if !self.visible.get(v.0 as usize).copied().unwrap_or(false) {
            self.problem(
                at,
                format!(
                    "uses %{}, which is not defined before it in an enclosing region",
                    v.0
                ),
            );
        }
    }
    fn define(&mut self, v: ValueId, param: bool) {
        let Some(def) = self.f.value(v) else {
            self.problem(Some(v), "is not a value of the function");
            return;
        };
        if (def.op == Op::Param) != param {
            self.problem(
                Some(v),
                if param {
                    "a construct value that is not a parameter"
                } else {
                    "a parameter placed as a node"
                },
            );
        }
        match self.placed.get_mut(v.0 as usize) {
            Some(p) if *p => self.problem(Some(v), "is defined twice"),
            Some(p) => *p = true,
            None => {}
        }
        if let Some(s) = self.visible.get_mut(v.0 as usize) {
            *s = true;
        }
    }
    fn hide(&mut self, from: &[bool]) {
        // what a region defined is not visible after it: restore the visibility before it, keeping the slots
        for (v, &before) in self.visible.iter_mut().zip(from) {
            *v = before;
        }
    }
    fn op(&mut self, v: ValueId) {
        let Some(def) = self.f.value(v) else { return };
        for o in def.op.operands() {
            self.uses(Some(v), o);
        }
        if !def.ty.is_valid() {
            self.problem(Some(v), format!("type {} has no lanes count of 1 to 4", def.ty));
        }
        match derived_type(&def.op, |o| self.ty(o)) {
            Ok(Some(t)) if t != def.ty => self.problem(
                Some(v),
                format!("states {} where its operation gives {t}", def.ty),
            ),
            Ok(_) => {}
            Err(e) => self.problem(Some(v), e),
        }
        if let Err(e) = stated(&def.op, def.ty, self.f, self.interface) {
            self.problem(Some(v), e);
        }
        if matches!(
            def.op,
            Op::Sample {
                level: Level::Auto | Level::Bias(_),
                ..
            }
        ) && self.stage != Stage::Pixel
        {
            self.problem(Some(v), "an implicit-level sample outside a pixel program");
        }
        if matches!(def.op, Op::Unary(Unary::Derivative(..), _)) && self.stage != Stage::Pixel {
            self.problem(Some(v), "a derivative outside a pixel program");
        }
        if let Op::ReportHit { attributes, .. } = def.op {
            if self.stage != Stage::Intersection {
                self.problem(Some(v), "a reported hit outside an intersection program");
            }
            self.handed(attributes, "a reported hit");
        }
        let (stage, patch) = (self.stage, self.patch);
        let wrong = match &def.op {
            Op::LoadInputAt(..) if !matches!(stage, Stage::Geometry | Stage::Hull | Stage::Domain) => {
                Some("a control point or vertex input outside a geometry, hull or domain program")
            }
            Op::LoadOutputAt(..) if !patch => {
                Some("an output control point read outside a patch constant function")
            }
            Op::LoadPatch(..) if !(stage == Stage::Domain || patch) => {
                Some("a patch constant read outside a domain program or a patch constant function")
            }
            Op::Builtin(Builtin::OutputControlPoint) if stage != Stage::Hull || patch => {
                Some("the output control point outside a hull program's main function")
            }
            Op::Builtin(Builtin::PhaseInstance) if !patch => {
                Some("a phase instance outside a patch constant function")
            }
            Op::Builtin(Builtin::PrimitiveId)
                if !matches!(stage, Stage::Geometry | Stage::Hull | Stage::Domain) =>
            {
                Some("the primitive id builtin outside a geometry, hull or domain program")
            }
            Op::Builtin(Builtin::GsInstance) if stage != Stage::Geometry => {
                Some("the geometry instance outside a geometry program")
            }
            Op::Builtin(Builtin::DomainLocation) if stage != Stage::Domain => {
                Some("the domain location outside a domain program")
            }
            Op::Wave(Wave::QuadLane(..) | Wave::QuadSwap(..))
                if !matches!(stage, Stage::Pixel | Stage::Compute) =>
            {
                Some("a quad read outside a pixel or compute program")
            }
            _ => None,
        };
        if let Some(w) = wrong {
            self.problem(Some(v), w);
        }
    }
    fn node(&mut self, n: &Node) {
        match n {
            Node::Value(v) => {
                self.op(*v);
                self.define(*v, false);
            }
            Node::Demote if self.stage != Stage::Pixel => {
                self.problem(None, "a demote outside a pixel program");
            }
            Node::Demote => {}
            Node::StoreVar(var, value) => {
                self.uses(None, *value);
                self.writable(*var, *value);
                match (self.f.vars.get(var.0 as usize), self.ty(*value)) {
                    (Some(d), Some(t)) if d.ty == t => {}
                    (Some(d), Some(t)) => {
                        let m = format!("stores {t} into variable {} of {}", d.name, d.ty);
                        self.problem(Some(*value), m);
                    }
                    _ => self.problem(
                        Some(*value),
                        format!("store to variable {}, which does not exist", var.0),
                    ),
                }
            }
            Node::StoreElement(var, i, value, _) => {
                self.uses(None, *i);
                self.uses(None, *value);
                self.writable(*var, *value);
                match self.f.vars.get(var.0 as usize) {
                    Some(d) if d.len.is_none() => {
                        let m = format!("an element store to {}, which is not an array", d.name);
                        self.problem(Some(*value), m);
                    }
                    Some(d) if Some(d.ty) != self.ty(*value) => {
                        let m = format!(
                            "stores {:?} into an element of {} of {}",
                            self.ty(*value),
                            d.name,
                            d.ty
                        );
                        self.problem(Some(*value), m);
                    }
                    Some(_) => {}
                    None => self.problem(
                        Some(*value),
                        format!("store to variable {}, which does not exist", var.0),
                    ),
                }
                if let Some(t) = self.ty(*i)
                    && let Err(e) = index_type(t, "an element index")
                {
                    self.problem(Some(*i), e);
                }
            }
            Node::StoreBuffer {
                resource: r,
                address,
                value,
            } => {
                for v in n.values() {
                    self.uses(None, v);
                }
                let t = self.ty(*value);
                let mut wrong = Vec::new();
                match resource(self.interface, *r) {
                    Ok(Resource {
                        kind:
                            ResourceKind::Storage {
                                target: StorageTarget::Buffer(_),
                                ..
                            },
                        ..
                    }) => {}
                    Ok(res) => wrong.push(format!("a buffer store to {:?}", res.kind)),
                    Err(e) => wrong.push(e),
                }
                if let Some(i) = address.index.or(r.index)
                    && let Some(it) = self.ty(i)
                    && let Err(e) = index_type(it, "a buffer index")
                {
                    wrong.push(e);
                }
                if t.is_none_or(|t| t.scalar.bits() != 32 || t.scalar == Scalar::Bool) {
                    wrong.push(format!("a buffer store of {t:?}"));
                }
                for w in wrong {
                    self.problem(Some(*value), w);
                }
            }
            Node::StoreTexel { image, coord, value } => {
                for v in n.values() {
                    self.uses(None, v);
                }
                let want = match resource(self.interface, *image).map(|r| &r.kind) {
                    Ok(ResourceKind::Storage {
                        target: StorageTarget::Image { dim, arrayed, sample },
                        ..
                    }) => Ok((
                        dims(*dim).saturating_add(u8::from(*arrayed)),
                        sample_scalar(*sample),
                    )),
                    Ok(ResourceKind::Storage {
                        target: StorageTarget::Buffer(BufferKind::Typed(v)),
                        ..
                    }) => Ok((1, value_type(*v).scalar)),
                    Ok(k) => Err(format!("a texel store to {k:?}")),
                    Err(e) => Err(e),
                };
                match want {
                    Ok((lanes, scalar)) => {
                        if let Some(c) = self.ty(*coord)
                            && let Err(e) = int_coord(c, lanes, "a texel coordinate")
                        {
                            self.problem(Some(*coord), e);
                        }
                        if self.ty(*value) != Some(Type::vector(scalar, 4)) {
                            let m = format!(
                                "a texel store of {:?} where vec4<{}> is wanted",
                                self.ty(*value),
                                scalar.name()
                            );
                            self.problem(Some(*value), m);
                        }
                    }
                    Err(e) => self.problem(Some(*value), e),
                }
            }
            Node::StorePatch(slot, mask, value) => {
                self.uses(None, *value);
                if !self.patch {
                    self.problem(
                        Some(*value),
                        "a patch constant store outside a patch constant function",
                    );
                }
                match patch_constants(self.interface)
                    .get(*slot as usize)
                    .map(|o| masked_type(o.value, *mask))
                {
                    Some(Ok(t)) if Some(t) == self.ty(*value) => {}
                    Some(Ok(t)) => {
                        let m = format!(
                            "patch constant {slot} lanes {mask:#06b} of {t} take {:?}",
                            self.ty(*value)
                        );
                        self.problem(Some(*value), m);
                    }
                    Some(Err(e)) => self.problem(Some(*value), format!("patch constant {slot}: {e}")),
                    None => self.problem(
                        Some(*value),
                        format!("patch constant slot {slot} is not in the interface"),
                    ),
                }
            }
            // a storage fence alone orders one invocation's memory operations, which any stage may
            Node::Barrier(b) if (b.execution || b.workgroup) && !self.stage.has_thread_groups() => {
                self.problem(
                    None,
                    "a workgroup barrier outside a compute, mesh or amplification program",
                );
            }
            Node::Barrier(b) if *b == Barrier::default() => self.problem(None, "a barrier of nothing"),
            Node::Barrier(_) => {}
            Node::Emit(_) | Node::Cut(_) if self.stage != Stage::Geometry => {
                self.problem(None, "a vertex emitted or a strip cut outside a geometry program");
            }
            Node::Emit(_) | Node::Cut(_) => {}
            Node::DispatchMesh { .. } if self.stage != Stage::Amplification => {
                self.problem(None, "a mesh launch outside an amplification program");
            }
            Node::DispatchMesh { groups, payload } => {
                self.uses(None, *groups);
                if self.ty(*groups) != Some(Type::vector(Scalar::U32, 3)) {
                    self.problem(
                        Some(*groups),
                        "a mesh launch of other thread groups than a vec3<u32>",
                    );
                }
                let bytes = match self.interface.facts {
                    StageFacts::Amplification(a) => a.payload,
                    _ => 0,
                };
                match self.f.vars.get(payload.0 as usize) {
                    None => self.problem(
                        None,
                        format!("a mesh launch of variable {}, which does not exist", payload.0),
                    ),
                    Some(d) if d.space == crate::Space::Payload => {
                        self.problem(
                            None,
                            format!("a mesh launch of {}, a mesh program's task payload", d.name),
                        );
                    }
                    Some(d) if d.ty != Type::U32 || d.len != Some(bytes.div_ceil(4)) => {
                        let m = format!(
                            "a mesh launch of {}, not the {} u32 words of the {bytes}-byte payload the facts state",
                            d.name,
                            bytes.div_ceil(4)
                        );
                        self.problem(None, m);
                    }
                    Some(_) => {}
                }
            }
            Node::TraceRay { .. }
                if !matches!(self.stage, Stage::RayGeneration | Stage::ClosestHit | Stage::Miss) =>
            {
                self.problem(
                    None,
                    "a traced ray outside a ray generation, closest-hit or miss program",
                );
            }
            Node::TraceRay { scene, ray, payload } => {
                for v in scene.index.into_iter().chain(ray.values()) {
                    self.uses(None, v);
                }
                match self.interface.resource(scene.id).map(|r| &r.kind) {
                    Some(ResourceKind::AccelerationStructure) => {}
                    _ => self.problem(
                        None,
                        format!(
                            "a ray traced into resource {}, not an acceleration structure",
                            scene.id.0
                        ),
                    ),
                }
                let vec3 = Type::vector(Scalar::F32, 3);
                let wanted = [
                    (ray.flags, Type::U32, "flags"),
                    (ray.mask, Type::U32, "instance mask"),
                    (ray.hit_group_offset, Type::U32, "hit group offset"),
                    (ray.hit_group_stride, Type::U32, "hit group stride"),
                    (ray.miss_index, Type::U32, "miss index"),
                    (ray.origin, vec3, "origin"),
                    (ray.t_min, Type::F32, "least distance"),
                    (ray.direction, vec3, "direction"),
                    (ray.t_max, Type::F32, "greatest distance"),
                ];
                for (v, t, what) in wanted {
                    if self.ty(v) != Some(t) {
                        self.problem(Some(v), format!("a ray's {what} that is not a {t}"));
                    }
                }
                self.handed(*payload, "a traced ray's payload");
            }
            Node::CallShader { .. }
                if !matches!(
                    self.stage,
                    Stage::RayGeneration | Stage::ClosestHit | Stage::Miss | Stage::Callable
                ) =>
            {
                self.problem(
                    None,
                    "a shader call outside a ray generation, closest-hit, miss or callable program",
                );
            }
            Node::CallShader { index, parameter } => {
                self.uses(None, *index);
                if self.ty(*index) != Some(Type::U32) {
                    self.problem(Some(*index), "a shader call of an index that is not a u32");
                }
                self.handed(*parameter, "a shader call's parameter");
            }
            Node::SetMeshCounts { .. } | Node::StoreVertex { .. } | Node::StoreIndices { .. }
                if self.stage != Stage::Mesh =>
            {
                self.problem(None, "a meshlet written outside a mesh program");
            }
            Node::SetMeshCounts { vertices, primitives } => {
                for (v, what) in [(vertices, "vertex"), (primitives, "primitive")] {
                    self.uses(None, *v);
                    if self.ty(*v) != Some(Type::U32) {
                        let m = format!("a mesh {what} count of {:?}", self.ty(*v));
                        self.problem(Some(*v), m);
                    }
                }
            }
            Node::StoreVertex {
                vertex,
                slot,
                mask,
                value,
            } => {
                self.uses(None, *vertex);
                self.uses(None, *value);
                if self.ty(*vertex) != Some(Type::U32) {
                    let m = format!("a mesh vertex numbered by {:?}", self.ty(*vertex));
                    self.problem(Some(*vertex), m);
                }
                match self
                    .interface
                    .outputs
                    .get(*slot as usize)
                    .map(|o| masked_type(o.value, *mask))
                {
                    Some(Ok(t)) if Some(t) == self.ty(*value) => {}
                    Some(Ok(t)) => {
                        let m = format!(
                            "vertex output {slot} lanes {mask:#06b} of {t} take {:?}",
                            self.ty(*value)
                        );
                        self.problem(Some(*value), m);
                    }
                    Some(Err(e)) => self.problem(Some(*value), format!("vertex output {slot}: {e}")),
                    None => self.problem(
                        Some(*value),
                        format!("vertex output slot {slot} is not in the interface"),
                    ),
                }
            }
            Node::StoreIndices { primitive, indices } => {
                self.uses(None, *primitive);
                self.uses(None, *indices);
                if self.ty(*primitive) != Some(Type::U32) {
                    let m = format!("a mesh primitive numbered by {:?}", self.ty(*primitive));
                    self.problem(Some(*primitive), m);
                }
                if let StageFacts::Mesh(m) = &self.interface.facts {
                    let lanes = u8::try_from(m.topology.indices()).unwrap_or(0);
                    if self.ty(*indices) != Some(Type::vector(Scalar::U32, lanes)) {
                        let m = format!(
                            "a {:?} primitive's {lanes} indices given as {:?}",
                            m.topology,
                            self.ty(*indices)
                        );
                        self.problem(Some(*indices), m);
                    }
                }
            }
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => {
                self.uses(None, *selector);
                if !matches!(self.ty(*selector), Some(Type::U32 | Type::I32)) {
                    self.problem(Some(*selector), "a switch selector that is not a 32-bit integer");
                }
                let mut seen: Vec<u32> = Vec::new();
                for c in cases {
                    if c.values.is_empty() {
                        self.problem(None, "a switch case with no values");
                    }
                    for v in &c.values {
                        if seen.contains(v) {
                            self.problem(None, format!("switch value {v} in two cases"));
                        }
                        seen.push(*v);
                    }
                }
                let before = self.visible.clone();
                self.labels.push((*label, results.clone(), None));
                for r in cases.iter().map(|c| &c.body).chain(core::iter::once(default)) {
                    self.region(r, Some(results));
                    self.hide(&before);
                }
                self.labels.pop();
                for r in results {
                    self.define(*r, true);
                }
            }
            Node::StoreOutput(slot, mask, value) => {
                self.uses(None, *value);
                match self
                    .interface
                    .outputs
                    .get(*slot as usize)
                    .map(|o| masked_type(o.value, *mask))
                {
                    Some(Ok(t)) if Some(t) == self.ty(*value) => {}
                    Some(Ok(t)) => {
                        let m = format!(
                            "output {slot} lanes {mask:#06b} of {t} take {:?}",
                            self.ty(*value)
                        );
                        self.problem(Some(*value), m);
                    }
                    Some(Err(e)) => self.problem(Some(*value), format!("output {slot}: {e}")),
                    None => self.problem(
                        Some(*value),
                        format!("output slot {slot} is not in the interface"),
                    ),
                }
            }
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                self.uses(None, *cond);
                if self.ty(*cond) != Some(Type::BOOL) {
                    self.problem(Some(*cond), "an if condition that is not a bool");
                }
                let before = self.visible.clone();
                self.region(accept, Some(results));
                self.hide(&before);
                self.region(reject, Some(results));
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
            Node::Block { label, body, results } => {
                let before = self.visible.clone();
                self.labels.push((*label, results.clone(), None));
                self.region(body, Some(results));
                self.labels.pop();
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                if params.len() != inits.len() {
                    self.problem(None, "a loop with as many initial values as parameters");
                }
                for (p, i) in params.iter().zip(inits) {
                    self.uses(Some(*p), *i);
                    if self.ty(*p) != self.ty(*i) {
                        self.problem(
                            Some(*p),
                            "a loop parameter of another type than its initial value",
                        );
                    }
                }
                let before = self.visible.clone();
                for p in params {
                    self.define(*p, true);
                }
                self.labels.push((*label, results.clone(), Some(params.clone())));
                self.region(body, Some(results));
                self.labels.pop();
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
        }
    }
    fn args(&mut self, what: &str, given: &[ValueId], wanted: &[ValueId]) {
        for g in given {
            self.uses(None, *g);
        }
        let types = |c: &Self, vs: &[ValueId]| vs.iter().map(|v| c.ty(*v)).collect::<Vec<_>>();
        if types(self, given) != types(self, wanted) {
            self.problem(
                None,
                format!(
                    "{what} gives {} values that do not match the {} it fills",
                    given.len(),
                    wanted.len()
                ),
            );
        }
    }
    fn region(&mut self, r: &Region, results: Option<&Vec<ValueId>>) {
        for n in &r.nodes {
            self.node(n);
        }
        match &r.end {
            None => self.problem(None, "a region with no end"),
            Some(End::Yield(vs)) => {
                let wanted = results.cloned().unwrap_or_default();
                self.args("a yield", vs, &wanted);
            }
            Some(End::Br(l, vs)) => match self.labels.iter().rev().find(|(x, ..)| x == l).cloned() {
                Some((_, results, _)) => self.args("a break", vs, &results),
                None => self.problem(
                    None,
                    format!("a break to label {}, which no enclosing construct has", l.0),
                ),
            },
            Some(End::Continue(l, vs)) => match self.labels.iter().rev().find(|(x, ..)| x == l).cloned() {
                Some((_, _, Some(params))) => self.args("a continue", vs, &params),
                Some(_) => self.problem(None, format!("a continue to label {}, which is not a loop", l.0)),
                None => self.problem(
                    None,
                    format!("a continue to label {}, which no enclosing construct has", l.0),
                ),
            },
            Some(End::Kill) if self.stage != Stage::Pixel => {
                self.problem(None, "a kill outside a pixel program")
            }
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) if self.stage != Stage::AnyHit => {
                self.problem(None, "a hit decided outside an any-hit program")
            }
            Some(
                End::Return | End::Kill | End::IgnoreHit | End::AcceptHitAndEndSearch | End::Unreachable,
            ) => {}
        }
    }
    fn cfg(&mut self, blocks: &[Block]) {
        let n = blocks.len();
        if n == 0 {
            self.problem(None, "a graph with no blocks");
            return;
        }
        let succ = |b: &Block| -> Vec<u32> { b.term.targets().iter().map(|t| t.block).collect() };
        let idom = dominators(blocks.iter().map(succ).collect());
        // a value is visible in a block when its own block dominates it (earlier in the same block)
        let mut home = vec![None; self.f.values.len()];
        for (i, b) in blocks.iter().enumerate() {
            for v in b.params.iter().chain(b.nodes.iter().filter_map(|n| match n {
                Node::Value(v) => Some(v),
                _ => None,
            })) {
                if let Some(h) = home.get_mut(v.0 as usize) {
                    *h = Some(i);
                }
            }
        }
        // every path from the entry to a block it does not reach passes every block (none does): each dominates it.
        // Its code is still checked, by every rule but this one
        let dominates = |a: usize, mut b: usize| {
            if b != 0 && idom.get(b).copied().flatten().is_none() {
                return true;
            }
            loop {
                if a == b {
                    return true;
                }
                match idom.get(b).copied().flatten() {
                    Some(p) if p != b => b = p,
                    _ => return false,
                }
            }
        };
        for (i, b) in blocks.iter().enumerate() {
            let mut local = Vec::new();
            let seen = |c: &mut Self, local: &Vec<ValueId>, at: Option<ValueId>, v: ValueId| {
                let ok = local.contains(&v)
                    || home
                        .get(v.0 as usize)
                        .copied()
                        .flatten()
                        .is_some_and(|h| h != i && dominates(h, i));
                if !ok {
                    c.problem(at, format!("uses %{}, which does not dominate it", v.0));
                }
            };
            for p in &b.params {
                self.define(*p, true);
                local.push(*p);
            }
            for node in &b.nodes {
                match node {
                    Node::Value(v) => {
                        if let Some(def) = self.f.value(*v) {
                            for o in def.op.operands() {
                                seen(self, &local, Some(*v), o);
                            }
                        }
                        // operand visibility was checked above; mark them so `op` checks only the rules
                        let saved = self.visible.clone();
                        self.visible.iter_mut().for_each(|s| *s = true);
                        self.op(*v);
                        self.visible = saved;
                        self.define(*v, false);
                        local.push(*v);
                    }
                    Node::If { .. } | Node::Block { .. } | Node::Loop { .. } | Node::Switch { .. } => {
                        self.problem(None, "a construct inside a graph block")
                    }
                    _ => {
                        for x in node.values() {
                            seen(self, &local, None, x);
                        }
                        let saved = self.visible.clone();
                        self.visible.iter_mut().for_each(|s| *s = true);
                        self.node(node);
                        self.visible = saved;
                    }
                }
            }
            match &b.term {
                Terminator::Branch { cond, .. } => {
                    seen(self, &local, None, *cond);
                    if self.ty(*cond) != Some(Type::BOOL) {
                        self.problem(Some(*cond), "a branch condition that is not a bool");
                    }
                }
                Terminator::Switch { selector, cases, .. } => {
                    seen(self, &local, None, *selector);
                    if !matches!(self.ty(*selector), Some(Type::U32 | Type::I32)) {
                        self.problem(Some(*selector), "a switch selector that is not a 32-bit integer");
                    }
                    let mut values: Vec<u32> = cases.iter().map(|(v, _)| *v).collect();
                    values.sort_unstable();
                    if values.windows(2).any(|w| w.first() == w.get(1)) {
                        self.problem(None, "a switch value with two targets");
                    }
                }
                Terminator::Kill if self.stage != Stage::Pixel => {
                    self.problem(None, "a kill outside a pixel program");
                }
                Terminator::IgnoreHit | Terminator::AcceptHitAndEndSearch if self.stage != Stage::AnyHit => {
                    self.problem(None, "a hit decided outside an any-hit program");
                }
                _ => {}
            }
            let targets = b.term.targets();
            for t in targets {
                match blocks.get(t.block as usize) {
                    None => self.problem(None, format!("a branch to block {}, of {n}", t.block)),
                    Some(to) => {
                        for a in &t.args {
                            seen(self, &local, None, *a);
                        }
                        let types = |vs: &[ValueId]| vs.iter().map(|v| self.ty(*v)).collect::<Vec<_>>();
                        if types(&t.args) != types(&to.params) {
                            self.problem(
                                None,
                                format!(
                                    "a branch to block {} whose arguments do not match its parameters",
                                    t.block
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Immediate dominators of a graph given by successor lists, block 0 the entry; `None` for unreachable blocks (and
/// the entry is its own).
pub fn dominators(succ: Vec<Vec<u32>>) -> Vec<Option<usize>> {
    let n = succ.len();
    // reverse post-order
    let mut order = Vec::new();
    let mut seen = vec![false; n];
    let mut stack = vec![(0usize, 0usize)];
    if let Some(s) = seen.get_mut(0) {
        *s = true;
    }
    while let Some((b, i)) = stack.pop() {
        let next = succ.get(b).and_then(|s| s.get(i)).map(|&x| x as usize);
        match next {
            Some(x) => {
                stack.push((b, i.saturating_add(1)));
                if x < n && !seen.get(x).copied().unwrap_or(true) {
                    if let Some(s) = seen.get_mut(x) {
                        *s = true;
                    }
                    stack.push((x, 0));
                }
            }
            None => order.push(b),
        }
    }
    order.reverse();
    let mut rpo = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        if let Some(r) = rpo.get_mut(b) {
            *r = i;
        }
    }
    let mut preds = vec![Vec::new(); n];
    for (b, s) in succ.iter().enumerate() {
        for &x in s {
            if let Some(p) = preds.get_mut(x as usize) {
                p.push(b);
            }
        }
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if let Some(e) = idom.get_mut(0) {
        *e = Some(0);
    }
    let intersect = |idom: &Vec<Option<usize>>, mut a: usize, mut b: usize| {
        let r = |x: usize| rpo.get(x).copied().unwrap_or(usize::MAX);
        while a != b {
            while r(a) > r(b) {
                a = idom.get(a).copied().flatten().unwrap_or(0);
            }
            while r(b) > r(a) {
                b = idom.get(b).copied().flatten().unwrap_or(0);
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut new = None;
            for &p in preds.get(b).map(Vec::as_slice).unwrap_or(&[]) {
                if idom.get(p).copied().flatten().is_some() {
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&idom, p, q),
                    });
                }
            }
            if new.is_some() && idom.get(b).copied().flatten() != new {
                if let Some(slot) = idom.get_mut(b) {
                    *slot = new;
                }
                changed = true;
            }
        }
    }
    idom
}

fn check_function(p: &Program, f: &Function, patch: bool, problems: &mut Vec<Problem>) {
    let mut c = Checker {
        f,
        interface: &p.interface,
        stage: p.stage,
        visible: vec![false; f.values.len()],
        placed: vec![false; f.values.len()],
        labels: Vec::new(),
        problems: Vec::new(),
        patch,
    };
    for v in &f.vars {
        if v.space == crate::Space::Workgroup && !p.stage.has_thread_groups() {
            c.problem(
                None,
                format!(
                    "workgroup variable {} outside a compute, mesh or amplification program",
                    v.name
                ),
            );
        }
        if v.space == crate::Space::Payload {
            let bytes = match p.interface.facts {
                StageFacts::Mesh(m) => Some(m.payload),
                _ => None,
            };
            match bytes {
                None => c.problem(None, format!("task payload {} outside a mesh program", v.name)),
                Some(b) if v.ty != Type::U32 || v.len != Some(b.div_ceil(4)) || b == 0 => c.problem(
                    None,
                    format!(
                        "task payload {} is not the {} u32 words of the {b}-byte payload the facts state",
                        v.name,
                        b.div_ceil(4)
                    ),
                ),
                Some(_) if !v.init.is_empty() => {
                    c.problem(None, format!("task payload {} has a start of its own", v.name));
                }
                Some(_) => {}
            }
        }
        if matches!(v.space, crate::Space::RayPayload | crate::Space::HitAttributes) {
            let (payload, attributes) = match &p.interface.facts {
                StageFacts::RayTracing(r) => (r.payload, r.attributes),
                _ => (0, 0),
            };
            let (what, bytes, stages): (&str, u32, &[Stage]) = if v.space == crate::Space::RayPayload {
                (
                    "ray payload",
                    payload,
                    &[Stage::AnyHit, Stage::ClosestHit, Stage::Miss, Stage::Callable],
                )
            } else {
                ("hit attributes", attributes, &[Stage::AnyHit, Stage::ClosestHit])
            };
            let others = f.vars.iter().filter(|w| w.space == v.space).count();
            if !stages.contains(&p.stage) {
                c.problem(None, format!("{what} {} in a {:?} program", v.name, p.stage));
            } else if others > 1 {
                c.problem(None, format!("{others} variables of {what}"));
            } else if v.ty != Type::U32 || v.len != Some(bytes.div_ceil(4)) || bytes == 0 {
                c.problem(
                    None,
                    format!(
                        "{what} {} is not the {} u32 words of the {bytes} bytes the facts state",
                        v.name,
                        bytes.div_ceil(4)
                    ),
                );
            } else if !v.init.is_empty() {
                c.problem(None, format!("{what} {} has a start of its own", v.name));
            }
        }
        if !v.init.is_empty() && v.init.len() != v.len.map_or(1, |n| n as usize) {
            c.problem(
                None,
                format!("variable {} starts with {} elements", v.name, v.init.len()),
            );
        }
        if v.len == Some(0) {
            c.problem(None, format!("array variable {} has no elements", v.name));
        }
    }
    match &f.body {
        Body::Regions(r) => {
            c.region(r, None);
            if !matches!(
                r.end,
                Some(
                    End::Return
                        | End::Kill
                        | End::IgnoreHit
                        | End::AcceptHitAndEndSearch
                        | End::Unreachable
                        | End::Yield(_)
                )
            ) {
                c.problem(None, "the body ends in an exit to a construct it is not in");
            }
        }
        Body::Cfg(blocks) => c.cfg(blocks),
    }
    problems.extend(c.problems);
}

/// Check a program: every problem found.
pub fn validate(p: &Program) -> Result<(), Vec<Problem>> {
    let mut problems = Vec::new();
    if p.stage != p.interface.stage {
        problems.push(Problem {
            value: None,
            message: "the program's stage is not its interface's".into(),
        });
    }
    check_function(p, &p.entry, false, &mut problems);
    match (&p.patch, p.stage) {
        (Some(f), Stage::Hull) => check_function(p, f, true, &mut problems),
        (Some(_), _) => problems.push(Problem {
            value: None,
            message: "a patch constant function in a program that is not a hull program".into(),
        }),
        (None, Stage::Hull) => problems.push(Problem {
            value: None,
            message: "a hull program without its patch constant function".into(),
        }),
        (None, _) => {}
    }
    match (p.workgroup, p.stage.has_thread_groups()) {
        (Some(size), true) if size.contains(&0) => problems.push(Problem {
            value: None,
            message: "a thread group with no invocations along an axis".into(),
        }),
        (None, true) => problems.push(Problem {
            value: None,
            message: "a compute, mesh or amplification program without its thread group".into(),
        }),
        (Some(_), false) => problems.push(Problem {
            value: None,
            message: "a thread group in a program that is not a compute, mesh or amplification program"
                .into(),
        }),
        _ => {}
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}
