//! The lowering: every naga node this crate builds is built here. Operations naga has whole map to one node; the
//! others expand here, each in one function (`is_nan`, `saturate`, `min_max`, `float_to_int`, `round_away`, ...).

use alloc::{format, string::String, vec, vec::Vec};

use iridia_ir::{
    Axis, Binary, Body, Cmp, End, Fineness, FloatToInt, Function, Level, MinMaxNan, MulZero, Node, OnZero,
    Op, OutOfBounds, Program, Region, Rounding, SaturateNan, Scalar, Ternary, Type, Unary, ValueId,
    interface::{ResourceId, ResourceKind, Stage, TextureDim, Value},
};
use naga::{
    AddressSpace, ArraySize, BinaryOperator as Bo, Block, DerivativeAxis, DerivativeControl, EntryPoint,
    Expression as E, GlobalVariable, Handle, ImageClass, ImageDimension, Literal, LocalVariable,
    MathFunction as Mf, RelationalFunction, ResourceBinding, SampleLevel, ScalarKind, ShaderStage, Span,
    Statement, StructMember, SwizzleComponent, TypeInner, UnaryOperator, VectorSize,
};

use crate::{BindingPlan, Error, io::entry_io};

/// The lanes a mask names, in ascending order.
fn masked(mask: u8) -> impl Iterator<Item = u8> {
    (0..4u8).filter(move |n| mask >> n & 1 != 0)
}

fn err(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn naga_scalar(s: Scalar) -> Result<naga::Scalar, Error> {
    Ok(match s {
        Scalar::Bool => naga::Scalar::BOOL,
        Scalar::I16 => naga::Scalar {
            kind: ScalarKind::Sint,
            width: 2,
        },
        Scalar::I32 => naga::Scalar::I32,
        Scalar::I64 => naga::Scalar::I64,
        Scalar::U16 => naga::Scalar {
            kind: ScalarKind::Uint,
            width: 2,
        },
        Scalar::U32 => naga::Scalar::U32,
        Scalar::U64 => naga::Scalar::U64,
        Scalar::F16 => naga::Scalar::F16,
        Scalar::F32 => naga::Scalar::F32,
        Scalar::F64 => naga::Scalar::F64,
        Scalar::I8 | Scalar::U8 => return Err(err(format!("{} has no naga type", s.name()))),
    })
}

fn size(lanes: u8) -> Result<VectorSize, Error> {
    Ok(match lanes {
        2 => VectorSize::Bi,
        3 => VectorSize::Tri,
        4 => VectorSize::Quad,
        n => return Err(err(format!("a vector of {n} lanes"))),
    })
}

fn component(lane: u8) -> Result<SwizzleComponent, Error> {
    Ok(match lane {
        0 => SwizzleComponent::X,
        1 => SwizzleComponent::Y,
        2 => SwizzleComponent::Z,
        3 => SwizzleComponent::W,
        n => return Err(err(format!("lane {n}"))),
    })
}

fn value_type(v: Value) -> Type {
    iridia_ir::value_type(v)
}

/// A construct being lowered: the label exits name, and where they put their values.
struct Construct {
    label: iridia_ir::LabelId,
    /// The locals a yield or a break stores the construct's results in.
    results: Vec<Handle<LocalVariable>>,
    /// The locals a continue stores the loop's parameters in; `None` for a construct that is not a loop.
    params: Option<Vec<Handle<LocalVariable>>>,
    /// What naga construct it is lowered as: a loop (`Break` and `Continue` reach it), a switch (`Break` reaches it), or
    /// a block (neither does).
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Block,
    Loop,
    Switch,
}

/// What a region's yield does.
#[derive(Clone, Copy)]
enum Yield<'a> {
    /// The function's body: a yield of nothing falls off its end.
    Top,
    /// An `If` arm or a `Block` body: store the results and fall through.
    Fall(&'a [Handle<LocalVariable>]),
    /// A loop body: store the results and leave the loop.
    Break(&'a [Handle<LocalVariable>]),
}

/// Where the outputs go.
enum Outputs {
    /// Stored at the top level only, before the one return at its end: the last value of each slot.
    Direct(Vec<Option<Handle<E>>>),
    /// A local per output, loaded at each return.
    Locals(Vec<Handle<LocalVariable>>),
}

struct Lower<'a> {
    program: &'a Program,
    f: &'a Function,
    module: naga::Module,
    func: naga::Function,
    /// The expression of each value, once it is computed.
    values: Vec<Option<Handle<E>>>,
    /// The global of each resource.
    globals: Vec<(ResourceId, Handle<GlobalVariable>)>,
    /// The argument of each input slot.
    arguments: Vec<Option<u32>>,
    vars: Vec<Handle<LocalVariable>>,
    /// The blocks being filled, innermost last, each with the first expression not yet emitted.
    blocks: Vec<(Block, Option<Handle<E>>)>,
    constructs: Vec<Construct>,
    outputs: Outputs,
    /// The output structure's members: the value each carries and the varying's own value.
    members: Vec<(Value, Value)>,
    result: Option<Handle<naga::Type>>,
}

impl Lower<'_> {
    fn ty(&mut self, inner: TypeInner) -> Handle<naga::Type> {
        self.module
            .types
            .insert(naga::Type { name: None, inner }, Span::UNDEFINED)
    }
    fn ty_of(&mut self, t: Type) -> Result<Handle<naga::Type>, Error> {
        let scalar = naga_scalar(t.scalar)?;
        let inner = if t.lanes == 1 {
            TypeInner::Scalar(scalar)
        } else {
            TypeInner::Vector {
                size: size(t.lanes)?,
                scalar,
            }
        };
        Ok(self.ty(inner))
    }
    /// Close the open emit range of the innermost block.
    fn flush(&mut self) {
        if let Some((block, start)) = self.blocks.last_mut()
            && let Some(first) = start.take()
            && let Some(last) = self.func.expressions.iter().last().map(|(h, _)| h)
        {
            block.push(
                Statement::Emit(naga::Range::new_from_bounds(first, last)),
                Span::UNDEFINED,
            );
        }
    }
    fn expr(&mut self, e: E) -> Handle<E> {
        let pre = e.needs_pre_emit();
        if pre {
            self.flush();
        }
        let h = self.func.expressions.append(e, Span::UNDEFINED);
        if !pre
            && let Some((_, start)) = self.blocks.last_mut()
            && start.is_none()
        {
            *start = Some(h);
        }
        h
    }
    fn statement(&mut self, s: Statement) -> Result<(), Error> {
        self.flush();
        self.blocks
            .last_mut()
            .ok_or_else(|| err("no open block"))?
            .0
            .push(s, Span::UNDEFINED);
        Ok(())
    }
    fn get(&self, v: ValueId) -> Result<Handle<E>, Error> {
        self.values
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| err(format!("%{} is used before it is computed", v.0)))
    }
    fn vty(&self, v: ValueId) -> Result<Type, Error> {
        self.f.ty(v).ok_or_else(|| err(format!("%{} has no type", v.0)))
    }
    fn set(&mut self, v: ValueId, h: Handle<E>) -> Result<(), Error> {
        *self
            .values
            .get_mut(v.0 as usize)
            .ok_or_else(|| err(format!("%{} is out of range", v.0)))? = Some(h);
        Ok(())
    }

    // literals and the small expansions

    /// A literal of `scalar` with these bits, `lanes` times (a splat for a vector).
    fn literal_bits(&mut self, scalar: Scalar, bits: u64) -> Result<Handle<E>, Error> {
        let low = |b: u64| u32::try_from(b & 0xffff_ffff).unwrap_or(0);
        let l = match scalar {
            Scalar::Bool => Literal::Bool(bits != 0),
            Scalar::F32 => Literal::F32(f32::from_bits(low(bits))),
            Scalar::F64 => Literal::F64(f64::from_bits(bits)),
            Scalar::U32 => Literal::U32(low(bits)),
            Scalar::I32 => Literal::I32(low(bits).cast_signed()),
            Scalar::U64 => Literal::U64(bits),
            Scalar::I64 => Literal::I64(bits.cast_signed()),
            Scalar::U16 => Literal::U16(u16::try_from(bits & 0xffff).unwrap_or(0)),
            Scalar::I16 => Literal::I16(u16::try_from(bits & 0xffff).unwrap_or(0).cast_signed()),
            Scalar::F16 => {
                return Err(err("f16 literals are not lowered yet"));
            }
            Scalar::I8 | Scalar::U8 => return Err(err(format!("{} has no naga type", scalar.name()))),
        };
        Ok(self.expr(E::Literal(l)))
    }
    fn splat_to(&mut self, value: Handle<E>, lanes: u8) -> Result<Handle<E>, Error> {
        if lanes == 1 {
            return Ok(value);
        }
        let size = size(lanes)?;
        Ok(self.expr(E::Splat { size, value }))
    }
    /// A literal of the type `like` has (shape and all), every lane the same.
    fn constant(&mut self, like: Type, scalar: Scalar, bits: u64) -> Result<Handle<E>, Error> {
        let l = self.literal_bits(scalar, bits)?;
        self.splat_to(l, like.lanes)
    }
    fn f32s(&mut self, like: Type, x: f32) -> Result<Handle<E>, Error> {
        self.constant(like, Scalar::F32, u64::from(x.to_bits()))
    }
    fn binary(&mut self, op: Bo, left: Handle<E>, right: Handle<E>) -> Handle<E> {
        self.expr(E::Binary { op, left, right })
    }
    fn math(&mut self, fun: Mf, arg: Handle<E>, rest: &[Handle<E>]) -> Handle<E> {
        self.expr(E::Math {
            fun,
            arg,
            arg1: rest.first().copied(),
            arg2: rest.get(1).copied(),
            arg3: rest.get(2).copied(),
        })
    }
    fn select(&mut self, condition: Handle<E>, accept: Handle<E>, reject: Handle<E>) -> Handle<E> {
        self.expr(E::Select {
            condition,
            accept,
            reject,
        })
    }
    fn bitcast(&mut self, x: Handle<E>, kind: ScalarKind) -> Handle<E> {
        self.expr(E::As {
            expr: x,
            kind,
            convert: None,
        })
    }
    /// Whether a float is NaN, by its bits (WGSL has no `isNan`, and a compiler may assume no NaNs in comparisons).
    fn is_nan(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        if t.scalar != Scalar::F32 {
            return Err(err(format!("a NaN test of {t} is not lowered yet")));
        }
        let bits = self.bitcast(x, ScalarKind::Uint);
        let u = t.with(Scalar::U32);
        let abs = self.constant(u, Scalar::U32, 0x7fff_ffff)?;
        let infinity = self.constant(u, Scalar::U32, 0x7f80_0000)?;
        let magnitude = self.binary(Bo::And, bits, abs);
        Ok(self.binary(Bo::Greater, magnitude, infinity))
    }
    fn saturate(&mut self, x: Handle<E>, t: Type, nan: SaturateNan) -> Result<Handle<E>, Error> {
        let clamped = self.math(Mf::Saturate, x, &[]);
        match nan {
            SaturateNan::Any => Ok(clamped),
            SaturateNan::Zero => {
                let is_nan = self.is_nan(x, t)?;
                let zero = self.f32s(t, 0.0)?;
                Ok(self.select(is_nan, zero, clamped))
            }
        }
    }
    fn clamp_range(
        &mut self,
        x: Handle<E>,
        t: Type,
        lo: u32,
        hi: u32,
        nan: SaturateNan,
    ) -> Result<Handle<E>, Error> {
        let low = self.f32s(t, f32::from_bits(lo))?;
        let high = self.f32s(t, f32::from_bits(hi))?;
        let clamped = self.math(Mf::Clamp, x, &[low, high]);
        match nan {
            SaturateNan::Any => Ok(clamped),
            SaturateNan::Zero => {
                let is_nan = self.is_nan(x, t)?;
                let zero = self.f32s(t, 0.0)?;
                Ok(self.select(is_nan, zero, clamped))
            }
        }
    }
    /// `round(x * 2^f) / 2^f`, and `x` itself from `2^(23 - f)` on, where every float is on the grid already (and the
    /// scaled value could overflow).
    fn quantize(
        &mut self,
        x: Handle<E>,
        t: Type,
        frac_bits: u8,
        rounding: Rounding,
    ) -> Result<Handle<E>, Error> {
        let bits = u32::from(frac_bits);
        let scale = self.f32s(t, f32::from_bits(127u32.saturating_add(bits) << 23))?;
        let grid = self.f32s(t, f32::from_bits(150u32.saturating_sub(bits) << 23))?;
        let scaled = self.binary(Bo::Multiply, x, scale);
        let rounded = match rounding {
            Rounding::Floor => self.math(Mf::Floor, scaled, &[]),
            Rounding::Ceil => self.math(Mf::Ceil, scaled, &[]),
            Rounding::Trunc => self.math(Mf::Trunc, scaled, &[]),
            Rounding::NearestEven => self.math(Mf::Round, scaled, &[]),
            Rounding::NearestAway => self.round_away(scaled, t)?,
        };
        let stepped = self.binary(Bo::Divide, rounded, scale);
        let magnitude = self.math(Mf::Abs, x, &[]);
        let on_grid = self.binary(Bo::GreaterEqual, magnitude, grid);
        Ok(self.select(on_grid, x, stepped))
    }
    fn min_max(
        &mut self,
        fun: Mf,
        a: Handle<E>,
        b: Handle<E>,
        t: Type,
        nan: MinMaxNan,
    ) -> Result<Handle<E>, Error> {
        let plain = self.math(fun, a, &[b]);
        match nan {
            MinMaxNan::Any => Ok(plain),
            MinMaxNan::MinNum => {
                let a_nan = self.is_nan(a, t)?;
                let b_nan = self.is_nan(b, t)?;
                let unless_b = self.select(b_nan, a, plain);
                Ok(self.select(a_nan, b, unless_b))
            }
            MinMaxNan::Propagate => {
                let a_nan = self.is_nan(a, t)?;
                let b_nan = self.is_nan(b, t)?;
                let either = self.binary(Bo::LogicalOr, a_nan, b_nan);
                let nan = self.binary(Bo::Add, a, b);
                Ok(self.select(either, nan, plain))
            }
        }
    }
    /// D3D's `ftoi` and `ftou`: truncate toward zero, saturate to the integer type's range, NaN gives 0.
    fn float_to_int(
        &mut self,
        x: Handle<E>,
        from: Type,
        to: Type,
        rule: FloatToInt,
    ) -> Result<Handle<E>, Error> {
        let kind = naga_scalar(to.scalar)?.kind;
        let convert = |l: &mut Self, v: Handle<E>| {
            l.expr(E::As {
                expr: v,
                kind,
                convert: Some(4),
            })
        };
        if rule == FloatToInt::Undef {
            return Ok(convert(self, x));
        }
        // the largest floats below 2^31 and 2^32 are 2147483520 and 4294967040; from those limits on the result is
        // the type's maximum
        let (low, high, limit, max) = match to.scalar {
            Scalar::I32 => (
                -2_147_483_648.0,
                2_147_483_520.0,
                2_147_483_648.0,
                u64::from(i32::MAX.cast_unsigned()),
            ),
            Scalar::U32 => (0.0, 4_294_967_040.0, 4_294_967_296.0, u64::from(u32::MAX)),
            other => {
                return Err(err(format!(
                    "a saturating conversion to {} is not lowered yet",
                    other.name()
                )));
            }
        };
        if from.scalar != Scalar::F32 {
            return Err(err(format!(
                "a saturating conversion from {from} is not lowered yet"
            )));
        }
        let low = self.f32s(from, low)?;
        let high = self.f32s(from, high)?;
        let clamped = self.math(Mf::Clamp, x, &[low, high]);
        let converted = convert(self, clamped);
        let limit = self.f32s(from, limit)?;
        let too_big = self.binary(Bo::GreaterEqual, x, limit);
        let max = self.constant(to, to.scalar, max)?;
        let saturated = self.select(too_big, max, converted);
        let nan = self.is_nan(x, from)?;
        let zero = self.constant(to, to.scalar, 0)?;
        Ok(self.select(nan, zero, saturated))
    }
    /// To the nearest integer, a half away from zero: the truncation, moved one away from zero where the part cut off
    /// is at least a half (exact: `x - trunc(x)` is representable).
    fn round_away(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let truncated = self.math(Mf::Trunc, x, &[]);
        let cut = self.binary(Bo::Subtract, x, truncated);
        let size = self.math(Mf::Abs, cut, &[]);
        let half = self.f32s(t, 0.5)?;
        let up = self.binary(Bo::GreaterEqual, size, half);
        let sign = self.math(Mf::Sign, x, &[]);
        let away = self.binary(Bo::Add, truncated, sign);
        Ok(self.select(up, away, truncated))
    }
    /// Lane `k` of a value of type `t` (the value itself for a scalar).
    fn lane(&mut self, x: Handle<E>, t: Type, k: u8) -> Result<Handle<E>, Error> {
        Ok(if t.lanes == 1 {
            x
        } else {
            self.expr(E::AccessIndex {
                base: x,
                index: u32::from(k),
            })
        })
    }
    /// A value of type `t` built a lane at a time: `f` gives lane `k` as a scalar.
    fn per_lane(
        &mut self,
        t: Type,
        mut f: impl FnMut(&mut Self, u8) -> Result<Handle<E>, Error>,
    ) -> Result<Handle<E>, Error> {
        let parts = (0..t.lanes).map(|k| f(self, k)).collect::<Result<Vec<_>, _>>()?;
        match parts.as_slice() {
            [one] => Ok(*one),
            _ => {
                let ty = self.ty_of(t)?;
                Ok(self.expr(E::Compose { ty, components: parts }))
            }
        }
    }
    /// A `u32` count's low five bits: the count modulo 32.
    fn low_five(&mut self, x: Handle<E>) -> Result<Handle<E>, Error> {
        let mask = self.literal_bits(Scalar::U32, 31)?;
        Ok(self.binary(Bo::And, x, mask))
    }
    /// 1 where a bool lane holds, else 0, in the integer type `t`.
    fn bool_to_int(&mut self, cond: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let one = self.constant(t, t.scalar, 1)?;
        let zero = self.constant(t, t.scalar, 0)?;
        Ok(self.select(cond, one, zero))
    }
    /// -1, 0 or 1 by the sign; a zero gives itself and NaN 0 (WGSL's `sign` leaves both open).
    fn sign(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let signed = self.math(Mf::Sign, x, &[]);
        let zero = self.f32s(t, 0.0)?;
        let is_zero = self.binary(Bo::Equal, x, zero);
        let kept = self.select(is_zero, x, signed);
        let nan = self.is_nan(x, t)?;
        Ok(self.select(nan, zero, kept))
    }
    /// Whether a float is an infinity, by its bits.
    fn is_inf(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        if t.scalar != Scalar::F32 {
            return Err(err(format!("an infinity test of {t} is not lowered yet")));
        }
        let bits = self.bitcast(x, ScalarKind::Uint);
        let u = t.with(Scalar::U32);
        let abs = self.constant(u, Scalar::U32, 0x7fff_ffff)?;
        let infinity = self.constant(u, Scalar::U32, 0x7f80_0000)?;
        let magnitude = self.binary(Bo::And, bits, abs);
        Ok(self.binary(Bo::Equal, magnitude, infinity))
    }
    /// D3D's `firstbit_hi` (of `u32` bits) or `firstbit_shi` (of `i32` bits) of an integer of type `t`: 31 less WGSL's
    /// `firstLeadingBit`, all ones where that finds none.
    fn first_bit_high(&mut self, x: Handle<E>, t: Type, signed: bool) -> Result<Handle<E>, Error> {
        let kind = if signed { ScalarKind::Sint } else { ScalarKind::Uint };
        let own = naga_scalar(t.scalar)?.kind;
        let x = if own == kind { x } else { self.bitcast(x, kind) };
        let index = self.math(Mf::FirstLeadingBit, x, &[]);
        let index = if signed { self.bitcast(index, ScalarKind::Uint) } else { index };
        let u = t.with(Scalar::U32);
        let none = self.constant(u, Scalar::U32, u64::from(u32::MAX))?;
        let top = self.constant(u, Scalar::U32, 31)?;
        let found = self.binary(Bo::Equal, index, none);
        let above = self.binary(Bo::Subtract, top, index);
        let r = self.select(found, none, above);
        Ok(if own == ScalarKind::Uint { r } else { self.bitcast(r, own) })
    }
    /// The nearest binary16 to each `f32` lane, in the low 16 bits: integer arithmetic, so every backend gives the same
    /// bits (`pack2x16float` needs naga's f16 capability and leaves the rounding open). Ties go to even; NaN keeps
    /// its payload's top bits and is quiet, a value beyond the range is an infinity, a binary16 denormal is kept.
    fn f32_to_f16(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let u = t.with(Scalar::U32);
        let k = |l: &mut Self, v: u32| l.constant(u, Scalar::U32, u64::from(v));
        let w = self.bitcast(x, ScalarKind::Uint);
        let (c16, c23, c13) = (k(self, 16)?, k(self, 23)?, k(self, 13)?);
        let high = self.binary(Bo::ShiftRight, w, c16);
        let m8000 = k(self, 0x8000)?;
        let sign = self.binary(Bo::And, high, m8000);
        let shifted = self.binary(Bo::ShiftRight, w, c23);
        let mff = k(self, 0xff)?;
        let exp = self.binary(Bo::And, shifted, mff);
        let fraction = k(self, 0x7f_ffff)?;
        let man = self.binary(Bo::And, w, fraction);
        let zero = k(self, 0)?;
        let one = k(self, 1)?;
        // NaN and infinity
        let infinity = k(self, 0x7c00)?;
        let signed_infinity = self.binary(Bo::InclusiveOr, sign, infinity);
        let payload = self.binary(Bo::ShiftRight, man, c13);
        let m200 = k(self, 0x200)?;
        let payload = self.binary(Bo::InclusiveOr, payload, m200);
        let man_zero = self.binary(Bo::Equal, man, zero);
        let payload = self.select(man_zero, zero, payload);
        let special = self.binary(Bo::InclusiveOr, signed_infinity, payload);
        // a normal half keeps the exponent less 112 and drops 13 bits; below one the implicit bit joins a significand
        // shifted further, a denormal
        let c112 = k(self, 112)?;
        let normal = self.binary(Bo::Greater, exp, c112);
        let implicit = k(self, 0x80_0000)?;
        let with_lead = self.binary(Bo::InclusiveOr, man, implicit);
        let sig = self.select(normal, man, with_lead);
        let c126 = k(self, 126)?;
        let below = self.binary(Bo::Subtract, c126, exp);
        let shift = self.select(normal, c13, below);
        let rebased = self.binary(Bo::Subtract, exp, c112);
        let c10 = k(self, 10)?;
        let rebased = self.binary(Bo::ShiftLeft, rebased, c10);
        let top = self.select(normal, rebased, zero);
        let half = self.binary(Bo::ShiftRight, sig, shift);
        let unit = self.binary(Bo::ShiftLeft, one, shift);
        let below_unit = self.binary(Bo::Subtract, unit, one);
        let rest = self.binary(Bo::And, sig, below_unit);
        let shift_less = self.binary(Bo::Subtract, shift, one);
        let mid = self.binary(Bo::ShiftLeft, one, shift_less);
        let above = self.binary(Bo::Greater, rest, mid);
        let tie = self.binary(Bo::Equal, rest, mid);
        let odd = self.binary(Bo::And, half, one);
        let odd = self.binary(Bo::Equal, odd, one);
        let tie_odd = self.binary(Bo::And, tie, odd);
        let up = self.binary(Bo::InclusiveOr, above, tie_odd);
        let up = self.select(up, one, zero);
        // a carry out of the significand moves into the exponent, to an infinity at the top
        let body = self.binary(Bo::InclusiveOr, top, half);
        let body = self.binary(Bo::Add, body, up);
        let rounded = self.binary(Bo::InclusiveOr, sign, body);
        let c102 = k(self, 102)?;
        let tiny = self.binary(Bo::Less, exp, c102);
        let r = self.select(tiny, sign, rounded);
        let c143 = k(self, 143)?;
        let big = self.binary(Bo::GreaterEqual, exp, c143);
        let r = self.select(big, signed_infinity, r);
        let all_ones = self.binary(Bo::Equal, exp, mff);
        Ok(self.select(all_ones, special, r))
    }
    /// The `f32` of the binary16 in each lane's low 16 bits, in integer arithmetic (`unpack2x16float` needs naga's
    /// f16 capability).
    fn f16_to_f32(&mut self, x: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let u = t.with(Scalar::U32);
        let k = |l: &mut Self, v: u32| l.constant(u, Scalar::U32, u64::from(v));
        let m8000 = k(self, 0x8000)?;
        let c16 = k(self, 16)?;
        let sign = self.binary(Bo::And, x, m8000);
        let sign = self.binary(Bo::ShiftLeft, sign, c16);
        let c10 = k(self, 10)?;
        let shifted = self.binary(Bo::ShiftRight, x, c10);
        let m1f = k(self, 0x1f)?;
        let exp = self.binary(Bo::And, shifted, m1f);
        let m3ff = k(self, 0x3ff)?;
        let man = self.binary(Bo::And, x, m3ff);
        let (c13, c23, zero) = (k(self, 13)?, k(self, 23)?, k(self, 0)?);
        let fraction = self.binary(Bo::ShiftLeft, man, c13);
        // a normal half: the exponent plus 112
        let c112 = k(self, 112)?;
        let e = self.binary(Bo::Add, exp, c112);
        let e = self.binary(Bo::ShiftLeft, e, c23);
        let normal = self.binary(Bo::InclusiveOr, e, fraction);
        // a denormal, normalised: its leading bit becomes the implicit one
        let leading = self.math(Mf::CountLeadingZeros, man, &[]);
        let c21 = k(self, 21)?;
        let lead = self.binary(Bo::Subtract, leading, c21);
        let c113 = k(self, 113)?;
        let de = self.binary(Bo::Subtract, c113, lead);
        let de = self.binary(Bo::ShiftLeft, de, c23);
        let dm = self.binary(Bo::ShiftLeft, man, lead);
        let dm = self.binary(Bo::And, dm, m3ff);
        let dm = self.binary(Bo::ShiftLeft, dm, c13);
        let denormal = self.binary(Bo::InclusiveOr, de, dm);
        let man_zero = self.binary(Bo::Equal, man, zero);
        let denormal = self.select(man_zero, zero, denormal);
        // infinity and NaN
        let infinity = k(self, 0x7f80_0000)?;
        let special = self.binary(Bo::InclusiveOr, infinity, fraction);
        let exp_zero = self.binary(Bo::Equal, exp, zero);
        let r = self.select(exp_zero, denormal, normal);
        let exp_top = self.binary(Bo::Equal, exp, m1f);
        let r = self.select(exp_top, special, r);
        let r = self.binary(Bo::InclusiveOr, sign, r);
        Ok(self.bitcast(r, ScalarKind::Float))
    }
    /// C's `fmod` of `f32` lanes, exact: WGSL's `%` is `x - y * trunc(x / y)` with each step rounded, so the remainder
    /// is a long division of the significands in integers, one bit of the exponents' difference a step, every lane
    /// in one loop that stops when no lane has a step left.
    fn remainder(&mut self, x: Handle<E>, y: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        if t.scalar != Scalar::F32 {
            return Err(err(format!("a remainder of {t} is not lowered yet")));
        }
        let u = t.with(Scalar::U32);
        let k = |l: &mut Self, v: u32| l.constant(u, Scalar::U32, u64::from(v));
        let (bx, by) = (self.bitcast(x, ScalarKind::Uint), self.bitcast(y, ScalarKind::Uint));
        let magnitude = k(self, 0x7fff_ffff)?;
        let (ax, ay) = (self.binary(Bo::And, bx, magnitude), self.binary(Bo::And, by, magnitude));
        let sign_bit = k(self, 0x8000_0000)?;
        let sign = self.binary(Bo::And, bx, sign_bit);
        let infinity = k(self, 0x7f80_0000)?;
        let zero = k(self, 0)?;
        let one = k(self, 1)?;
        let implicit = k(self, 0x80_0000)?;
        let fraction = k(self, 0x7f_ffff)?;
        let twenty_three = k(self, 23)?;
        // NaN: x infinite or NaN, y zero or NaN; x itself: |x| < |y| (a zero x too) or y infinite
        let x_special = self.binary(Bo::GreaterEqual, ax, infinity);
        let y_zero = self.binary(Bo::Equal, ay, zero);
        let y_nan = self.binary(Bo::Greater, ay, infinity);
        let nan = self.binary(Bo::InclusiveOr, x_special, y_zero);
        let nan = self.binary(Bo::InclusiveOr, nan, y_nan);
        let y_infinite = self.binary(Bo::Equal, ay, infinity);
        let small = self.binary(Bo::Less, ax, ay);
        let unchanged = self.binary(Bo::InclusiveOr, small, y_infinite);
        let skip = self.binary(Bo::InclusiveOr, nan, unchanged);
        // each significand normalised to its top bit at bit 23, its exponent (32 more than the biased one, so a
        // denormal's stays positive) lowered by the shift
        let eight = k(self, 8)?;
        let thirty_two = k(self, 32)?;
        let parts = |l: &mut Self, a: Handle<E>| -> Result<(Handle<E>, Handle<E>), Error> {
            let e = l.binary(Bo::ShiftRight, a, twenty_three);
            let denormal = l.binary(Bo::Equal, e, zero);
            let lead = l.select(denormal, zero, implicit);
            let f = l.binary(Bo::And, a, fraction);
            let m = l.binary(Bo::InclusiveOr, f, lead);
            let leading = l.math(Mf::CountLeadingZeros, m, &[]);
            let up = l.binary(Bo::Subtract, leading, eight);
            let m = l.binary(Bo::ShiftLeft, m, up);
            let e = l.math(Mf::Max, e, &[one]);
            let e = l.binary(Bo::Add, e, thirty_two);
            Ok((l.binary(Bo::Subtract, e, up), m))
        };
        let (ex, mx) = parts(self, ax)?;
        let (ey, my) = parts(self, ay)?;
        let difference = self.binary(Bo::Subtract, ex, ey);
        let steps = self.select(skip, zero, difference);
        let (m, d) = (self.local(u)?, self.local(u)?);
        self.store_local(m, mx)?;
        self.store_local(d, steps)?;
        // while a lane has a step: m = (m >= my ? m - my : m) << 1, one exponent fewer
        self.open();
        let pm = self.expr(E::LocalVariable(m));
        let pd = self.expr(E::LocalVariable(d));
        let mv = self.expr(E::Load { pointer: pm });
        let dv = self.expr(E::Load { pointer: pd });
        let active = self.binary(Bo::Greater, dv, zero);
        let any = if t.lanes == 1 {
            active
        } else {
            self.expr(E::Relational {
                fun: RelationalFunction::Any,
                argument: active,
            })
        };
        let none = self.expr(E::Unary {
            op: UnaryOperator::LogicalNot,
            expr: any,
        });
        self.open();
        self.statement(Statement::Break)?;
        let leave = self.close()?;
        self.statement(Statement::If {
            condition: none,
            accept: leave,
            reject: Block::new(),
        })?;
        let ge = self.binary(Bo::GreaterEqual, mv, my);
        let less = self.binary(Bo::Subtract, mv, my);
        let reduced = self.select(ge, less, mv);
        let doubled = self.binary(Bo::ShiftLeft, reduced, one);
        let next = self.select(active, doubled, mv);
        let fewer = self.binary(Bo::Subtract, dv, one);
        let left = self.select(active, fewer, dv);
        self.statement(Statement::Store { pointer: pm, value: next })?;
        self.statement(Statement::Store { pointer: pd, value: left })?;
        let body = self.close()?;
        self.statement(Statement::Loop {
            body,
            continuing: Block::new(),
            break_if: None,
        })?;
        let pm = self.expr(E::LocalVariable(m));
        let mv = self.expr(E::Load { pointer: pm });
        let ge = self.binary(Bo::GreaterEqual, mv, my);
        let less = self.binary(Bo::Subtract, mv, my);
        let r = self.select(ge, less, mv);
        // the remainder (below 2^24, at y's exponent) normalised; a biased exponent below 1 makes it a denormal,
        // shifted down (exact: a remainder is representable)
        let leading = self.math(Mf::CountLeadingZeros, r, &[]);
        let up = self.binary(Bo::Subtract, leading, eight);
        let rn = self.binary(Bo::ShiftLeft, r, up);
        let e = self.binary(Bo::Subtract, ey, up);
        let normal_floor = k(self, 33)?;
        let is_normal = self.binary(Bo::GreaterEqual, e, normal_floor);
        let biased = self.binary(Bo::Subtract, e, thirty_two);
        let high = self.binary(Bo::ShiftLeft, biased, twenty_three);
        let low = self.binary(Bo::And, rn, fraction);
        let normal = self.binary(Bo::InclusiveOr, high, low);
        let down = self.binary(Bo::Subtract, normal_floor, e);
        let thirty_one = k(self, 31)?;
        let down = self.math(Mf::Min, down, &[thirty_one]);
        let denormal = self.binary(Bo::ShiftRight, rn, down);
        let bits = self.select(is_normal, normal, denormal);
        let r_zero = self.binary(Bo::Equal, r, zero);
        let bits = self.select(r_zero, zero, bits);
        let bits = self.binary(Bo::InclusiveOr, bits, sign);
        let result = self.bitcast(bits, ScalarKind::Float);
        let kept = self.select(unchanged, x, result);
        // naga takes no NaN literal: the quiet NaN's bits
        let quiet = k(self, 0x7fc0_0000)?;
        let quiet = self.bitcast(quiet, ScalarKind::Float);
        Ok(self.select(nan, quiet, kept))
    }
    /// The high half of the 64-bit product of 32-bit integers, from 16-bit halves (WGSL has no 64-bit integers in
    /// browsers): unsigned, then for signed operands less each operand where the other is negative.
    fn mul_high(&mut self, a: Handle<E>, b: Handle<E>, t: Type, signed: bool) -> Result<Handle<E>, Error> {
        let u = t.with(Scalar::U32);
        let (a, b) = if signed {
            (self.bitcast(a, ScalarKind::Uint), self.bitcast(b, ScalarKind::Uint))
        } else {
            (a, b)
        };
        let sixteen = self.constant(u, Scalar::U32, 16)?;
        let low = self.constant(u, Scalar::U32, 0xffff)?;
        let al = self.binary(Bo::And, a, low);
        let ah = self.binary(Bo::ShiftRight, a, sixteen);
        let bl = self.binary(Bo::And, b, low);
        let bh = self.binary(Bo::ShiftRight, b, sixteen);
        let ll = self.binary(Bo::Multiply, al, bl);
        let lh = self.binary(Bo::Multiply, al, bh);
        let hl = self.binary(Bo::Multiply, ah, bl);
        let hh = self.binary(Bo::Multiply, ah, bh);
        // the middle column: the carry into the high half, each part below 2^16
        let ll_high = self.binary(Bo::ShiftRight, ll, sixteen);
        let lh_low = self.binary(Bo::And, lh, low);
        let hl_low = self.binary(Bo::And, hl, low);
        let mid = self.binary(Bo::Add, ll_high, lh_low);
        let mid = self.binary(Bo::Add, mid, hl_low);
        let carry = self.binary(Bo::ShiftRight, mid, sixteen);
        let lh_high = self.binary(Bo::ShiftRight, lh, sixteen);
        let hl_high = self.binary(Bo::ShiftRight, hl, sixteen);
        let high = self.binary(Bo::Add, hh, lh_high);
        let high = self.binary(Bo::Add, high, hl_high);
        let high = self.binary(Bo::Add, high, carry);
        if !signed {
            return Ok(high);
        }
        let zero = self.constant(u, Scalar::U32, 0)?;
        let sign = self.constant(u, Scalar::U32, 0x8000_0000)?;
        let a_sign = self.binary(Bo::And, a, sign);
        let a_negative = self.binary(Bo::NotEqual, a_sign, zero);
        let b_sign = self.binary(Bo::And, b, sign);
        let b_negative = self.binary(Bo::NotEqual, b_sign, zero);
        let less_b = self.select(a_negative, b, zero);
        let less_a = self.select(b_negative, a, zero);
        let high = self.binary(Bo::Subtract, high, less_b);
        let high = self.binary(Bo::Subtract, high, less_a);
        Ok(self.bitcast(high, ScalarKind::Sint))
    }
    /// An integer division or remainder (`op`) with the zero rule; WGSL's own is `ByOne`, and the most negative value
    /// by -1 is the IR's (itself, and 0).
    fn divide(
        &mut self,
        op: Bo,
        a: Handle<E>,
        b: Handle<E>,
        t: Type,
        zero: OnZero,
    ) -> Result<Handle<E>, Error> {
        match zero {
            OnZero::ByOne | OnZero::Undef => Ok(self.binary(op, a, b)),
            OnZero::AllOnes => {
                let z = self.constant(t, t.scalar, 0)?;
                let one = self.constant(t, t.scalar, 1)?;
                let by_zero = self.binary(Bo::Equal, b, z);
                let divisor = self.select(by_zero, one, b);
                let result = self.binary(op, a, divisor);
                let ones = self.constant(t, t.scalar, u64::MAX)?;
                Ok(self.select(by_zero, ones, result))
            }
        }
    }

    fn global(&self, id: ResourceId) -> Result<Handle<GlobalVariable>, Error> {
        self.globals
            .iter()
            .find(|(r, _)| *r == id)
            .map(|(_, g)| *g)
            .ok_or_else(|| err(format!("resource {} has no binding", id.0)))
    }
    fn constant_rows(&self, id: ResourceId) -> Result<u32, Error> {
        match self
            .program
            .interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .map(|r| &r.kind)
        {
            Some(ResourceKind::ConstantBuffer { layout }) => Ok(layout.size.div_ceil(16)),
            _ => Err(err(format!("resource {} is not a constant buffer", id.0))),
        }
    }
    /// A constant buffer read: rows of `vec4<u32>`, the result's bits from the row (and lane) the address names. A read
    /// at a fixed address may name any 32-bit word (a bank of 32-bit booleans, four to a row); an indexed one whole rows.
    fn load_buffer(
        &mut self,
        resource: ResourceId,
        index: Option<ValueId>,
        stride: u32,
        offset: u32,
        oob: OutOfBounds,
        t: Type,
    ) -> Result<Handle<E>, Error> {
        let rows_apart = stride.is_multiple_of(16) || index.is_none();
        if t.scalar.bits() != 32 || !offset.is_multiple_of(4) || !rows_apart {
            return Err(err(format!(
                "a buffer read of {t} at stride {stride}, offset {offset} is not lowered yet"
            )));
        }
        let lane = (offset % 16) / 4;
        if lane.saturating_add(u32::from(t.lanes)) > 4 {
            return Err(err(format!(
                "a buffer read of {t} across a row at offset {offset}"
            )));
        }
        let rows = self.constant_rows(resource)?;
        let g = self.global(resource)?;
        let bank = self.expr(E::GlobalVariable(g));
        let first = offset / 16;
        let (row, inside) = match index {
            None => {
                if first >= rows {
                    return Err(err(format!("a read of row {first} of a buffer of {rows}")));
                }
                let p = self.expr(E::AccessIndex {
                    base: bank,
                    index: first,
                });
                (p, None)
            }
            Some(i) => {
                let it = self.vty(i)?;
                if it != Type::I32 {
                    return Err(err(format!("a buffer index of {it} is not lowered yet")));
                }
                let mut index = self.get(i)?;
                let per = stride / 16;
                if per != 1 {
                    let per = self.literal_bits(Scalar::I32, u64::from(per))?;
                    index = self.binary(Bo::Multiply, index, per);
                }
                if first != 0 {
                    let first = self.literal_bits(Scalar::I32, u64::from(first))?;
                    index = self.binary(Bo::Add, index, first);
                }
                let last = i32::try_from(rows.saturating_sub(1))
                    .map_err(|_| err("a buffer of more than 2^31 rows"))?;
                let zero = self.literal_bits(Scalar::I32, 0)?;
                let last = self.literal_bits(Scalar::I32, u64::from(last.cast_unsigned()))?;
                let above = self.binary(Bo::GreaterEqual, index, zero);
                let below = self.binary(Bo::LessEqual, index, last);
                let inside = self.binary(Bo::LogicalAnd, above, below);
                let raised = self.math(Mf::Max, index, &[zero]);
                let clamped = self.math(Mf::Min, raised, &[last]);
                let p = self.expr(E::Access {
                    base: bank,
                    index: clamped,
                });
                (p, Some(inside))
            }
        };
        let raw = self.expr(E::Load { pointer: row });
        let mut bits = raw;
        if t.lanes == 1 {
            bits = self.expr(E::AccessIndex {
                base: raw,
                index: lane,
            });
        } else if !(lane == 0 && t.lanes == 4) {
            let mut pattern = [SwizzleComponent::X; 4];
            for (n, p) in (0u8..t.lanes).zip(pattern.iter_mut()) {
                *p = component(u8::try_from(lane).unwrap_or(0).saturating_add(n))?;
            }
            bits = self.expr(E::Swizzle {
                size: size(t.lanes)?,
                vector: raw,
                pattern,
            });
        }
        let kind = naga_scalar(t.scalar)?.kind;
        let v = if kind == ScalarKind::Uint {
            bits
        } else {
            self.bitcast(bits, kind)
        };
        Ok(match (inside, oob) {
            (Some(inside), OutOfBounds::Zero) => {
                let zero = self.constant(t, t.scalar, 0)?;
                self.select(inside, v, zero)
            }
            _ => v,
        })
    }

    fn op(&mut self, v: ValueId) -> Result<Handle<E>, Error> {
        let def = self
            .f
            .value(v)
            .ok_or_else(|| err(format!("%{} is not a value", v.0)))?;
        let t = def.ty;
        let op = def.op.clone();
        Ok(match op {
            Op::Literal(bits) => {
                let lanes: Vec<Handle<E>> = bits
                    .iter()
                    .take(usize::from(t.lanes))
                    .map(|b| self.literal_bits(t.scalar, *b))
                    .collect::<Result<_, _>>()?;
                match lanes.as_slice() {
                    [one] => *one,
                    _ => {
                        let ty = self.ty_of(t)?;
                        self.expr(E::Compose {
                            ty,
                            components: lanes,
                        })
                    }
                }
            }
            Op::Undef => {
                let ty = self.ty_of(t)?;
                self.expr(E::ZeroValue(ty))
            }
            Op::Param => return Err(err(format!("parameter %{} has no value here", v.0))),
            Op::Unary(u, a) => {
                let at = self.vty(a)?;
                let x = self.get(a)?;
                match u {
                    Unary::FNeg | Unary::INeg => self.expr(E::Unary {
                        op: UnaryOperator::Negate,
                        expr: x,
                    }),
                    Unary::FAbs => self.math(Mf::Abs, x, &[]),
                    Unary::FFract => self.math(Mf::Fract, x, &[]),
                    Unary::FRound(Rounding::Floor) => self.math(Mf::Floor, x, &[]),
                    Unary::FRound(Rounding::Ceil) => self.math(Mf::Ceil, x, &[]),
                    Unary::FRound(Rounding::Trunc) => self.math(Mf::Trunc, x, &[]),
                    Unary::FRound(Rounding::NearestEven) => self.math(Mf::Round, x, &[]),
                    Unary::FRound(Rounding::NearestAway) => self.round_away(x, at)?,
                    Unary::FSqrt => self.math(Mf::Sqrt, x, &[]),
                    Unary::FInverseSqrt => self.math(Mf::InverseSqrt, x, &[]),
                    Unary::FExp2 => self.math(Mf::Exp2, x, &[]),
                    Unary::FLog2 => self.math(Mf::Log2, x, &[]),
                    Unary::FSin => self.math(Mf::Sin, x, &[]),
                    Unary::FCos => self.math(Mf::Cos, x, &[]),
                    Unary::FTan => self.math(Mf::Tan, x, &[]),
                    Unary::FAsin => self.math(Mf::Asin, x, &[]),
                    Unary::FAcos => self.math(Mf::Acos, x, &[]),
                    Unary::FAtan => self.math(Mf::Atan, x, &[]),
                    Unary::FSinh => self.math(Mf::Sinh, x, &[]),
                    Unary::FCosh => self.math(Mf::Cosh, x, &[]),
                    Unary::FTanh => self.math(Mf::Tanh, x, &[]),
                    Unary::FSign => self.sign(x, at)?,
                    Unary::Saturate(nan) => self.saturate(x, at, nan)?,
                    Unary::ClampRange { lo, hi, nan } => self.clamp_range(x, at, lo, hi, nan)?,
                    Unary::Quantize { frac_bits, rounding } => self.quantize(x, at, frac_bits, rounding)?,
                    Unary::Derivative(axis, fineness) => self.expr(E::Derivative {
                        axis: match axis {
                            Axis::X => DerivativeAxis::X,
                            Axis::Y => DerivativeAxis::Y,
                        },
                        ctrl: match fineness {
                            Fineness::Coarse => DerivativeControl::Coarse,
                            Fineness::Fine => DerivativeControl::Fine,
                            Fineness::Any => DerivativeControl::None,
                        },
                        expr: x,
                    }),
                    Unary::Not if at.scalar == Scalar::Bool => self.expr(E::Unary {
                        op: UnaryOperator::LogicalNot,
                        expr: x,
                    }),
                    Unary::Not => self.expr(E::Unary {
                        op: UnaryOperator::BitwiseNot,
                        expr: x,
                    }),
                    Unary::Any => self.expr(E::Relational {
                        fun: RelationalFunction::Any,
                        argument: x,
                    }),
                    Unary::All => self.expr(E::Relational {
                        fun: RelationalFunction::All,
                        argument: x,
                    }),
                    Unary::IsNan => self.is_nan(x, at)?,
                    Unary::IsInf => self.is_inf(x, at)?,
                    Unary::CountOnes => self.math(Mf::CountOneBits, x, &[]),
                    Unary::BitReverse => self.math(Mf::ReverseBits, x, &[]),
                    Unary::FirstBitHigh => self.first_bit_high(x, at, false)?,
                    Unary::FirstBitHighSigned => self.first_bit_high(x, at, true)?,
                    // WGSL's `firstTrailingBit` gives all ones where no bit is set, as D3D's `firstbit_lo` does
                    Unary::FirstBitLow => self.math(Mf::FirstTrailingBit, x, &[]),
                    Unary::F32ToF16 => self.f32_to_f16(x, at)?,
                    Unary::F16ToF32 => self.f16_to_f32(x, at)?,
                }
            }
            Op::Binary(b, l, r) => {
                let at = self.vty(l)?;
                let (x, y) = (self.get(l)?, self.get(r)?);
                let cmp = |c: Cmp| match c {
                    Cmp::Eq => Bo::Equal,
                    Cmp::Ne => Bo::NotEqual,
                    Cmp::Lt => Bo::Less,
                    Cmp::Le => Bo::LessEqual,
                    Cmp::Gt => Bo::Greater,
                    Cmp::Ge => Bo::GreaterEqual,
                };
                match b {
                    Binary::FAdd | Binary::IAdd => self.binary(Bo::Add, x, y),
                    Binary::FSub | Binary::ISub => self.binary(Bo::Subtract, x, y),
                    Binary::FMul(MulZero::Ieee) | Binary::IMul => self.binary(Bo::Multiply, x, y),
                    Binary::FMul(MulZero::ZeroWins) => {
                        let product = self.binary(Bo::Multiply, x, y);
                        let zero = self.f32s(at, 0.0)?;
                        let x_zero = self.binary(Bo::Equal, x, zero);
                        let y_zero = self.binary(Bo::Equal, y, zero);
                        // `|`, which takes bool vectors as well as scalars (`||` takes scalars only)
                        let either = self.binary(Bo::InclusiveOr, x_zero, y_zero);
                        self.select(either, zero, product)
                    }
                    Binary::FDiv => self.binary(Bo::Divide, x, y),
                    // WGSL's `%` on floats is `x - y * trunc(x / y)`
                    Binary::FRem => self.remainder(x, y, at)?,
                    Binary::UMulHigh => self.mul_high(x, y, at, false)?,
                    Binary::SMulHigh => self.mul_high(x, y, at, true)?,
                    Binary::UAddCarry => {
                        let sum = self.binary(Bo::Add, x, y);
                        let wrapped = self.binary(Bo::Less, sum, x);
                        self.bool_to_int(wrapped, at)?
                    }
                    Binary::USubBorrow => {
                        let below = self.binary(Bo::Less, x, y);
                        self.bool_to_int(below, at)?
                    }
                    Binary::SDiv(z) => self.divide(Bo::Divide, x, y, at, z)?,
                    Binary::SRem(z) => self.divide(Bo::Modulo, x, y, at, z)?,
                    Binary::UMin => self.math(Mf::Min, x, &[y]),
                    Binary::UMax => self.math(Mf::Max, x, &[y]),
                    // WGSL takes a shift's count modulo the bit width, as the IR does
                    Binary::Shl => self.binary(Bo::ShiftLeft, x, y),
                    Binary::ShrU | Binary::ShrS => self.binary(Bo::ShiftRight, x, y),
                    Binary::FMin(nan) => self.min_max(Mf::Min, x, y, at, nan)?,
                    Binary::FMax(nan) => self.min_max(Mf::Max, x, y, at, nan)?,
                    Binary::FPow => self.math(Mf::Pow, x, &[y]),
                    Binary::FCmp(c) | Binary::SCmp(c) | Binary::UCmp(c) => self.binary(cmp(c), x, y),
                    Binary::UDiv(z) => self.divide(Bo::Divide, x, y, at, z)?,
                    Binary::URem(z) => self.divide(Bo::Modulo, x, y, at, z)?,
                    Binary::SMin => self.math(Mf::Min, x, &[y]),
                    Binary::SMax => self.math(Mf::Max, x, &[y]),
                    Binary::And if at.scalar == Scalar::Bool => self.binary(Bo::LogicalAnd, x, y),
                    Binary::Or if at.scalar == Scalar::Bool => self.binary(Bo::LogicalOr, x, y),
                    Binary::And => self.binary(Bo::And, x, y),
                    Binary::Or => self.binary(Bo::InclusiveOr, x, y),
                    Binary::Xor => self.binary(Bo::ExclusiveOr, x, y),
                }
            }
            Op::Ternary(op, a, b, c) => {
                let at = self.vty(a)?;
                let (x, y, z) = (self.get(a)?, self.get(b)?, self.get(c)?);
                match op {
                    Ternary::Fma => self.math(Mf::Fma, x, &[y, z]),
                    // WGSL's `extractBits` cuts the field at the top bit as D3D does; it does not take the counts
                    // modulo 32, so they are masked first; its counts are one scalar for every lane, so a vector's
                    // lanes are taken one at a time
                    Ternary::BitExtract { .. } => self.per_lane(at, |l, k| {
                        let xs = l.lane(x, at, k)?;
                        let os = l.lane(y, at, k)?;
                        let cs = l.lane(z, at, k)?;
                        let os = l.low_five(os)?;
                        let cs = l.low_five(cs)?;
                        Ok(l.math(Mf::ExtractBits, xs, &[os, cs]))
                    })?,
                }
            }
            Op::BitInsert {
                base,
                insert,
                offset,
                count,
            } => {
                let at = self.vty(base)?;
                let (b, i, o, c) = (self.get(base)?, self.get(insert)?, self.get(offset)?, self.get(count)?);
                self.per_lane(at, |l, k| {
                    let bs = l.lane(b, at, k)?;
                    let is = l.lane(i, at, k)?;
                    let os = l.lane(o, at, k)?;
                    let cs = l.lane(c, at, k)?;
                    let os = l.low_five(os)?;
                    let cs = l.low_five(cs)?;
                    Ok(l.math(Mf::InsertBits, bs, &[is, os, cs]))
                })?
            }
            Op::Select { cond, accept, reject } => {
                let (c, a, r) = (self.get(cond)?, self.get(accept)?, self.get(reject)?);
                self.select(c, a, r)
            }
            Op::Compose(parts) => {
                let components = parts
                    .iter()
                    .map(|p| self.get(*p))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = self.ty_of(t)?;
                self.expr(E::Compose { ty, components })
            }
            Op::Swizzle(a, lanes) => {
                let x = self.get(a)?;
                if t.lanes == 1 {
                    self.expr(E::AccessIndex {
                        base: x,
                        index: u32::from(lanes[0]),
                    })
                } else {
                    let mut pattern = [SwizzleComponent::X; 4];
                    for (p, &l) in pattern.iter_mut().zip(lanes.iter()).take(usize::from(t.lanes)) {
                        *p = component(l)?;
                    }
                    self.expr(E::Swizzle {
                        size: size(t.lanes)?,
                        vector: x,
                        pattern,
                    })
                }
            }
            Op::Splat(a) => {
                let x = self.get(a)?;
                self.splat_to(x, t.lanes)?
            }
            Op::Dot(a, b) => {
                let (x, y) = (self.get(a)?, self.get(b)?);
                self.math(Mf::Dot, x, &[y])
            }
            Op::Convert(a, rule) => {
                let from = self.vty(a)?;
                let x = self.get(a)?;
                if from.scalar.is_float() && t.scalar.is_integer() {
                    self.float_to_int(x, from, t, rule)?
                } else {
                    let s = naga_scalar(t.scalar)?;
                    self.expr(E::As {
                        expr: x,
                        kind: s.kind,
                        convert: Some(s.width),
                    })
                }
            }
            Op::Bitcast(a) => {
                let x = self.get(a)?;
                self.bitcast(x, naga_scalar(t.scalar)?.kind)
            }
            Op::LoadInput(slot, mask) => {
                let arg = self
                    .arguments
                    .get(slot as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| err(format!("input {slot} is not an argument of the entry point")))?;
                let all = self.expr(E::FunctionArgument(arg));
                let width = self
                    .program
                    .interface
                    .inputs
                    .get(slot as usize)
                    .map_or(0, |v| v.value.width);
                let lanes: Vec<u8> = masked(mask).collect();
                match lanes.as_slice() {
                    _ if lanes.len() == usize::from(width) => all,
                    [one] => self.expr(E::AccessIndex {
                        base: all,
                        index: u32::from(*one),
                    }),
                    _ => {
                        let mut pattern = [SwizzleComponent::X; 4];
                        for (p, &l) in pattern.iter_mut().zip(&lanes) {
                            *p = component(l)?;
                        }
                        self.expr(E::Swizzle {
                            size: size(t.lanes)?,
                            vector: all,
                            pattern,
                        })
                    }
                }
            }
            Op::LoadState(r) => {
                return Err(err(format!(
                    "pipeline state r{} is read: specialise it first (a state block is not lowered yet)",
                    r.0
                )));
            }
            Op::LoadVar(var) => {
                let local = *self
                    .vars
                    .get(var.0 as usize)
                    .ok_or_else(|| err(format!("variable {} does not exist", var.0)))?;
                let p = self.expr(E::LocalVariable(local));
                self.expr(E::Load { pointer: p })
            }
            Op::LoadElement(var, index, oob) => {
                let (pointer, inside) = self.element(var, index, oob)?;
                let v = self.expr(E::Load { pointer });
                match inside {
                    None => v,
                    Some(inside) => {
                        let ty = self.ty_of(t)?;
                        let zero = self.expr(E::ZeroValue(ty));
                        self.select(inside, v, zero)
                    }
                }
            }
            Op::LoadInputAt(..) => return Err(err("a read of one vertex's input is not lowered yet")),
            Op::LoadOutputAt(..) => return Err(err("a read of one control point's output is not lowered yet")),
            Op::LoadPatch(..) => return Err(err("a patch constant read is not lowered yet")),
            Op::Builtin(b) => return Err(err(format!("builtin {b:?} is not lowered yet"))),
            Op::Gather { .. } => return Err(err("a gather is not lowered yet")),
            Op::LoadTexel { .. } => return Err(err("a texel load is not lowered yet")),
            Op::Query(..) => return Err(err("a resource query is not lowered yet")),
            Op::Atomic { .. } => return Err(err("an atomic is not lowered yet")),
            Op::Counter(..) => return Err(err("a buffer counter is not lowered yet")),
            Op::LoadBuffer {
                resource,
                address,
                oob,
            } => {
                if resource.index.is_some() {
                    return Err(err("a read of an element of an array of buffers is not lowered yet"));
                }
                self.load_buffer(resource.id, address.index, address.stride, address.offset, oob, t)?
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
                    return Err(err("a sample of an element of an array of resources is not lowered yet"));
                }
                if compare.is_some() || clamp.is_some() || offset != [0; 3] {
                    return Err(err("a sample with a comparison, offset or level clamp is not lowered yet"));
                }
                let (texture, sampler) = (texture.id, sampler.id);
                let image = self.global(texture)?;
                let image = self.expr(E::GlobalVariable(image));
                let s = self.global(sampler)?;
                let sampler = self.expr(E::GlobalVariable(s));
                let coordinate = self.get(coord)?;
                let level = match level {
                    Level::Auto => SampleLevel::Auto,
                    Level::Zero => SampleLevel::Zero,
                    Level::Bias(b) => SampleLevel::Bias(self.get(b)?),
                    Level::Exact(l) => SampleLevel::Exact(self.get(l)?),
                    Level::Grad(x, y) => SampleLevel::Gradient {
                        x: self.get(x)?,
                        y: self.get(y)?,
                    },
                };
                self.expr(E::ImageSample {
                    image,
                    sampler,
                    gather: None,
                    coordinate,
                    array_index: None,
                    offset: None,
                    level,
                    depth_ref: None,
                    clamp_to_edge: false,
                })
            }
        })
    }

    /// The pointer to element `index` of array variable `var`, and where an index past its end reads zero and writes
    /// nothing (`OutOfBounds::Zero`), whether the index is inside: the pointer is then to the index clamped, which WGSL
    /// keeps in bounds. A signed index is taken as unsigned, so a negative one is past the end.
    fn element(
        &mut self,
        var: iridia_ir::VarId,
        index: ValueId,
        oob: OutOfBounds,
    ) -> Result<(Handle<E>, Option<Handle<E>>), Error> {
        let local = *self
            .vars
            .get(var.0 as usize)
            .ok_or_else(|| err(format!("variable {} does not exist", var.0)))?;
        let len = self
            .f
            .vars
            .get(var.0 as usize)
            .and_then(|v| v.len)
            .ok_or_else(|| err(format!("variable {} is not an array", var.0)))?;
        let base = self.expr(E::LocalVariable(local));
        let i = self.get(index)?;
        let i = if self.vty(index)? == Type::I32 {
            self.bitcast(i, ScalarKind::Uint)
        } else {
            i
        };
        match oob {
            OutOfBounds::Undef => Ok((self.expr(E::Access { base, index: i }), None)),
            OutOfBounds::Zero => {
                let end = self.literal_bits(Scalar::U32, u64::from(len))?;
                let last = self.literal_bits(Scalar::U32, u64::from(len.saturating_sub(1)))?;
                let inside = self.binary(Bo::Less, i, end);
                let clamped = self.math(Mf::Min, i, &[last]);
                Ok((self.expr(E::Access { base, index: clamped }), Some(inside)))
            }
        }
    }
    fn local(&mut self, t: Type) -> Result<Handle<LocalVariable>, Error> {
        let ty = self.ty_of(t)?;
        Ok(self.func.local_variables.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::UNDEFINED,
        ))
    }
    fn store_local(&mut self, local: Handle<LocalVariable>, value: Handle<E>) -> Result<(), Error> {
        let pointer = self.expr(E::LocalVariable(local));
        self.statement(Statement::Store { pointer, value })
    }
    /// The construct values `params` take the values the locals hold.
    fn load_locals(&mut self, params: &[ValueId], locals: &[Handle<LocalVariable>]) -> Result<(), Error> {
        for (&p, &l) in params.iter().zip(locals) {
            let pointer = self.expr(E::LocalVariable(l));
            let v = self.expr(E::Load { pointer });
            self.set(p, v)?;
        }
        Ok(())
    }
    fn store_all(&mut self, locals: &[Handle<LocalVariable>], vs: &[ValueId]) -> Result<(), Error> {
        if locals.len() != vs.len() {
            return Err(err(
                "an exit with another number of values than its construct takes",
            ));
        }
        for (&l, &v) in locals.iter().zip(vs) {
            let value = self.get(v)?;
            self.store_local(l, value)?;
        }
        Ok(())
    }
    fn output_value(&mut self, slot: usize, value: Handle<E>) -> Result<Handle<E>, Error> {
        // a pixel target narrower than four lanes is carried as four: the lanes it lacks are zero
        let &(carried, own) = self
            .members
            .get(slot)
            .ok_or_else(|| err(format!("output {slot} is not carried")))?;
        if carried.width == own.width {
            return Ok(value);
        }
        let ct = value_type(carried);
        let ot = value_type(own);
        let mut parts = Vec::new();
        for n in 0..ct.lanes {
            if n < ot.lanes {
                parts.push(if ot.lanes == 1 {
                    value
                } else {
                    self.expr(E::AccessIndex {
                        base: value,
                        index: u32::from(n),
                    })
                });
            } else {
                parts.push(self.literal_bits(ot.scalar, 0)?);
            }
        }
        let ty = self.ty_of(ct)?;
        Ok(self.expr(E::Compose {
            ty,
            components: parts,
        }))
    }
    /// The whole value of output `slot` after lanes `mask` take `value`: the lanes it does not name are the slot's
    /// current ones (zero before its first store).
    fn merge_output(&mut self, slot: usize, mask: u8, value: Handle<E>) -> Result<Handle<E>, Error> {
        let &(_, own) = self
            .members
            .get(slot)
            .ok_or_else(|| err(format!("output {slot} is not carried")))?;
        let ot = value_type(own);
        let lanes: Vec<u8> = masked(mask).collect();
        if lanes.len() == usize::from(ot.lanes) {
            return Ok(value);
        }
        let current = match &self.outputs {
            Outputs::Direct(values) => values.get(slot).copied().flatten(),
            Outputs::Locals(locals) => match locals.get(slot).copied() {
                Some(l) => {
                    let pointer = self.expr(E::LocalVariable(l));
                    Some(self.expr(E::Load { pointer }))
                }
                None => None,
            },
        };
        let ty = self.ty_of(ot)?;
        let current = match current {
            Some(c) => c,
            None => self.expr(E::ZeroValue(ty)),
        };
        let mut parts = Vec::new();
        for n in 0..ot.lanes {
            let part = match lanes.iter().position(|&l| l == n) {
                Some(_) if lanes.len() == 1 => value,
                Some(k) => {
                    let index = u32::try_from(k).map_err(|_| err("a lane past w"))?;
                    self.expr(E::AccessIndex { base: value, index })
                }
                None => self.expr(E::AccessIndex {
                    base: current,
                    index: u32::from(n),
                }),
            };
            parts.push(part);
        }
        Ok(self.expr(E::Compose {
            ty,
            components: parts,
        }))
    }
    fn ret(&mut self) -> Result<(), Error> {
        let Some(result) = self.result else {
            return self.statement(Statement::Return { value: None });
        };
        let mut components = Vec::new();
        let slots = self.members.len();
        for slot in 0..slots {
            let value = match &self.outputs {
                Outputs::Direct(values) => values.get(slot).copied().flatten(),
                Outputs::Locals(locals) => match locals.get(slot).copied() {
                    Some(l) => {
                        let pointer = self.expr(E::LocalVariable(l));
                        Some(self.expr(E::Load { pointer }))
                    }
                    None => None,
                },
            };
            let value = match value {
                Some(v) => v,
                None => {
                    let &(_, own) = self
                        .members
                        .get(slot)
                        .ok_or_else(|| err("an output with no member"))?;
                    let ty = self.ty_of(value_type(own))?;
                    self.expr(E::ZeroValue(ty))
                }
            };
            components.push(self.output_value(slot, value)?);
        }
        let value = self.expr(E::Compose {
            ty: result,
            components,
        });
        self.statement(Statement::Return { value: Some(value) })
    }
    fn construct(&self, l: iridia_ir::LabelId) -> Result<(usize, &Construct), Error> {
        self.constructs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.label == l)
            .ok_or_else(|| {
                err(format!(
                    "an exit to label {}, which no enclosing construct has",
                    l.0
                ))
            })
    }
    /// A break to construct `at` is a naga `Break` only when it is the innermost loop or switch; a continue, when it
    /// is the innermost loop (a switch between does not stop it).
    fn innermost(&self, at: usize, continues: bool) -> Result<(), Error> {
        let last = self.constructs.iter().rposition(|c| {
            c.kind == Kind::Loop || (!continues && c.kind == Kind::Switch)
        });
        if last == Some(at) {
            Ok(())
        } else {
            Err(err(
                "an exit past an enclosing construct: run iridia_passes::break_flags first",
            ))
        }
    }
    fn region(&mut self, r: &Region, yields: Yield<'_>) -> Result<(), Error> {
        for n in &r.nodes {
            self.node(n)?;
        }
        match &r.end {
            None => return Err(err("a region with no end")),
            Some(End::Yield(vs)) => match yields {
                Yield::Top if vs.is_empty() => {}
                Yield::Top => return Err(err("a yield of values at the end of the function")),
                Yield::Fall(locals) => self.store_all(locals, vs)?,
                Yield::Break(locals) => {
                    self.store_all(locals, vs)?;
                    self.statement(Statement::Break)?;
                }
            },
            Some(End::Br(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let (locals, kind) = (c.results.clone(), c.kind);
                self.store_all(&locals, vs)?;
                if kind == Kind::Block {
                    return Err(err(
                        "a break out of a block: run iridia_passes::break_flags first",
                    ));
                }
                self.innermost(at, false)?;
                self.statement(Statement::Break)?;
            }
            Some(End::Continue(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let locals = c
                    .params
                    .clone()
                    .ok_or_else(|| err("a continue to a construct that is not a loop"))?;
                self.store_all(&locals, vs)?;
                self.innermost(at, true)?;
                self.statement(Statement::Continue)?;
            }
            Some(End::Return) => self.ret()?,
            // WGSL's `discard` demotes the invocation and goes on: the return ends it
            Some(End::Kill) => {
                self.statement(Statement::Kill)?;
                self.ret()?;
            }
            Some(End::Unreachable) => {}
        }
        Ok(())
    }
    fn open(&mut self) {
        self.flush();
        self.blocks.push((Block::new(), None));
    }
    fn close(&mut self) -> Result<Block, Error> {
        self.flush();
        Ok(self.blocks.pop().ok_or_else(|| err("no open block"))?.0)
    }
    fn locals_for(&mut self, vs: &[ValueId]) -> Result<Vec<Handle<LocalVariable>>, Error> {
        vs.iter()
            .map(|v| {
                let t = self.vty(*v)?;
                self.local(t)
            })
            .collect()
    }
    fn node(&mut self, n: &Node) -> Result<(), Error> {
        match n {
            Node::Value(v) => {
                let h = self.op(*v)?;
                self.set(*v, h)?;
            }
            // WGSL's `discard` is a demotion: the invocation goes on as a helper
            Node::Demote => self.statement(Statement::Kill)?,
            Node::StoreVar(var, x) => {
                let local = *self
                    .vars
                    .get(var.0 as usize)
                    .ok_or_else(|| err(format!("variable {} does not exist", var.0)))?;
                let value = self.get(*x)?;
                self.store_local(local, value)?;
            }
            Node::StoreOutput(slot, mask, x) => {
                let value = self.get(*x)?;
                let slot = *slot as usize;
                let value = self.merge_output(slot, *mask, value)?;
                match &mut self.outputs {
                    Outputs::Direct(values) => {
                        *values
                            .get_mut(slot)
                            .ok_or_else(|| err(format!("output {slot} is not carried")))? = Some(value);
                    }
                    Outputs::Locals(locals) => {
                        let l = *locals
                            .get(slot)
                            .ok_or_else(|| err(format!("output {slot} is not carried")))?;
                        self.store_local(l, value)?;
                    }
                }
            }
            Node::StoreElement(var, index, x, oob) => {
                let value = self.get(*x)?;
                let (pointer, inside) = self.element(*var, *index, *oob)?;
                match inside {
                    None => self.statement(Statement::Store { pointer, value })?,
                    Some(condition) => {
                        self.open();
                        self.statement(Statement::Store { pointer, value })?;
                        let accept = self.close()?;
                        self.statement(Statement::If {
                            condition,
                            accept,
                            reject: Block::new(),
                        })?;
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
                let sel = self.get(*selector)?;
                let signed = self.vty(*selector)? == Type::I32;
                let locals = self.locals_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: locals.clone(),
                    params: None,
                    kind: Kind::Switch,
                });
                let mut naga_cases = Vec::new();
                for c in cases {
                    self.open();
                    self.region(&c.body, Yield::Fall(&locals))?;
                    let body = self.close()?;
                    // a case of several values: each but the last falls through, empty, to the last
                    let last = c.values.len().saturating_sub(1);
                    for (k, v) in c.values.iter().enumerate() {
                        naga_cases.push(naga::SwitchCase {
                            value: if signed {
                                naga::SwitchValue::I32(v.cast_signed())
                            } else {
                                naga::SwitchValue::U32(*v)
                            },
                            body: if k == last { body.clone() } else { Block::new() },
                            fall_through: k != last,
                        });
                    }
                }
                self.open();
                self.region(default, Yield::Fall(&locals))?;
                let body = self.close()?;
                naga_cases.push(naga::SwitchCase {
                    value: naga::SwitchValue::Default,
                    body,
                    fall_through: false,
                });
                self.constructs.pop();
                self.statement(Statement::Switch {
                    selector: sel,
                    cases: naga_cases,
                })?;
                self.load_locals(results, &locals)?;
            }
            Node::StoreBuffer { .. } => return Err(err("a buffer store is not lowered yet")),
            Node::StoreTexel { .. } => return Err(err("an image store is not lowered yet")),
            Node::StorePatch(..) => return Err(err("a patch constant store is not lowered yet")),
            Node::Barrier(_) => return Err(err("a barrier is not lowered yet")),
            Node::Emit(_) | Node::Cut(_) => return Err(err("a geometry program's emit and cut are not lowered yet")),
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                let condition = self.get(*cond)?;
                let locals = self.locals_for(results)?;
                self.open();
                self.region(accept, Yield::Fall(&locals))?;
                let a = self.close()?;
                self.open();
                self.region(reject, Yield::Fall(&locals))?;
                let b = self.close()?;
                self.statement(Statement::If {
                    condition,
                    accept: a,
                    reject: b,
                })?;
                self.load_locals(results, &locals)?;
            }
            Node::Block { label, body, results } => {
                let locals = self.locals_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: locals.clone(),
                    params: None,
                    kind: Kind::Block,
                });
                self.open();
                self.region(body, Yield::Fall(&locals))?;
                let b = self.close()?;
                self.constructs.pop();
                self.statement(Statement::Block(b))?;
                self.load_locals(results, &locals)?;
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                let param_locals = self.locals_for(params)?;
                self.store_all(&param_locals, inits)?;
                let locals = self.locals_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: locals.clone(),
                    params: Some(param_locals.clone()),
                    kind: Kind::Loop,
                });
                self.open();
                self.load_locals(params, &param_locals)?;
                self.region(body, Yield::Break(&locals))?;
                let b = self.close()?;
                self.constructs.pop();
                self.statement(Statement::Loop {
                    body: b,
                    continuing: Block::new(),
                    break_if: None,
                })?;
                self.load_locals(results, &locals)?;
            }
        }
        Ok(())
    }
}

/// Whether every output store is at the top level of a body whose only return is its end.
fn direct(body: &Region) -> bool {
    fn nested(r: &Region) -> bool {
        r.nodes.iter().any(|n| match n {
            Node::StoreOutput(..) => true,
            _ => inner(n),
        })
    }
    fn returns(r: &Region) -> bool {
        matches!(r.end, Some(End::Return)) || r.nodes.iter().any(|n| n.regions().into_iter().any(returns))
    }
    // a construct's regions store an output or return
    fn inner(n: &Node) -> bool {
        n.regions().into_iter().any(|r| nested(r) || returns(r))
    }
    !body.nodes.iter().any(inner)
}

/// The module of a program: its resources as the plan binds them, its entry point `main`.
pub fn module(program: &Program, plan: &BindingPlan) -> Result<naga::Module, Error> {
    let f = &program.entry;
    let Body::Regions(body) = &f.body else {
        return Err(err("a graph body: structurize it first"));
    };
    let stage = match program.stage {
        Stage::Vertex => ShaderStage::Vertex,
        Stage::Pixel => ShaderStage::Fragment,
        other => return Err(err(format!("a {other:?} program is not lowered yet"))),
    };
    let io = entry_io(&program.interface).map_err(err)?;
    let mut l = Lower {
        program,
        f,
        module: naga::Module::default(),
        func: naga::Function::default(),
        values: vec![None; f.values.len()],
        globals: Vec::new(),
        arguments: vec![None; program.interface.inputs.len()],
        vars: Vec::new(),
        blocks: vec![(Block::new(), None)],
        constructs: Vec::new(),
        outputs: Outputs::Direct(Vec::new()),
        members: Vec::new(),
        result: None,
    };
    // the result structure first, then the resources, then the arguments: the order naga's writer names them in
    let mut members = Vec::new();
    let (mut offset, mut align) = (0u32, 1u32);
    for item in &io.results {
        let ty = l.ty_of(value_type(item.value))?;
        let own = program
            .interface
            .outputs
            .get(item.varying)
            .map(|v| v.value)
            .ok_or_else(|| err("an output with no varying"))?;
        // WGSL's layout: a vector of three lanes is aligned as one of four
        let bytes = item.value.scalar.bytes();
        let lanes = if item.value.width == 3 {
            4
        } else {
            u32::from(item.value.width)
        };
        let a = lanes.saturating_mul(bytes).max(1);
        offset = offset.div_ceil(a).saturating_mul(a);
        align = align.max(a);
        members.push(StructMember {
            name: Some(item.name.clone()),
            ty,
            binding: Some(item.binding.clone()),
            offset,
        });
        offset = offset.saturating_add(u32::from(item.value.width).saturating_mul(bytes));
        l.members.push((item.value, own));
    }
    if !members.is_empty() {
        let span = offset.div_ceil(align).saturating_mul(align);
        let out = naga::Type {
            name: Some("Out".into()),
            inner: TypeInner::Struct { members, span },
        };
        l.result = Some(l.module.types.insert(out, Span::UNDEFINED));
    }
    // pipeline state has no global: a read of it is refused where it is read
    for r in program
        .interface
        .resources
        .iter()
        .filter(|r| !matches!(r.kind, ResourceKind::PipelineState(_)))
    {
        let Some(bind) = plan.bindings.iter().find(|b| b.resource == r.id) else {
            return Err(err(format!(
                "resource {} ({}) has no binding in the plan",
                r.id.0, r.provenance.name
            )));
        };
        let (ty, space) = match &r.kind {
            ResourceKind::ConstantBuffer { layout } => {
                let rows = core::num::NonZeroU32::new(layout.size.div_ceil(16))
                    .ok_or_else(|| err("an empty constant buffer"))?;
                let row = l.ty(TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: naga::Scalar::U32,
                });
                (
                    l.ty(TypeInner::Array {
                        base: row,
                        size: ArraySize::Constant(rows),
                        stride: 16,
                    }),
                    AddressSpace::Uniform,
                )
            }
            ResourceKind::Texture {
                dim,
                arrayed,
                multisampled,
                sample,
            } => {
                let dim = match dim {
                    TextureDim::D1 => ImageDimension::D1,
                    TextureDim::D2 => ImageDimension::D2,
                    TextureDim::D3 => ImageDimension::D3,
                    TextureDim::Cube => ImageDimension::Cube,
                };
                let kind = match sample {
                    iridia_ir::interface::SampleType::Float => ScalarKind::Float,
                    iridia_ir::interface::SampleType::Sint => ScalarKind::Sint,
                    iridia_ir::interface::SampleType::Uint => ScalarKind::Uint,
                    iridia_ir::interface::SampleType::Depth => {
                        return Err(err("a depth texture is not lowered yet"));
                    }
                };
                let class = ImageClass::Sampled {
                    kind,
                    multi: *multisampled,
                };
                (
                    l.ty(TypeInner::Image {
                        dim,
                        arrayed: *arrayed,
                        class,
                    }),
                    AddressSpace::Handle,
                )
            }
            ResourceKind::Sampler { comparison } => (
                l.ty(TypeInner::Sampler {
                    comparison: *comparison,
                }),
                AddressSpace::Handle,
            ),
            other => return Err(err(format!("a {other:?} resource is not lowered yet"))),
        };
        let g = l.module.global_variables.append(
            GlobalVariable {
                name: Some(bind.name.clone()),
                space,
                binding: Some(ResourceBinding {
                    group: bind.group,
                    binding: bind.binding,
                }),
                ty,
                init: None,
                memory_decorations: Default::default(),
            },
            Span::UNDEFINED,
        );
        l.globals.push((r.id, g));
    }
    for (n, item) in io.arguments.iter().enumerate() {
        let ty = l.ty_of(value_type(item.value))?;
        l.func.arguments.push(naga::FunctionArgument {
            name: Some(item.name.clone()),
            ty,
            binding: Some(item.binding.clone()),
        });
        if let Some(a) = l.arguments.get_mut(item.varying) {
            *a = Some(u32::try_from(n).map_err(|_| err("more than 2^32 arguments"))?);
        }
    }
    for var in &f.vars {
        if var.space != iridia_ir::Space::Private {
            return Err(err(format!("workgroup variable {} is not lowered yet", var.name)));
        }
        let element = l.ty_of(var.ty)?;
        let ty = match var.len {
            None => element,
            Some(len) => {
                // WGSL's layout: a vector of three lanes takes the room of four
                let lanes = if var.ty.lanes == 3 { 4 } else { u32::from(var.ty.lanes) };
                let stride = lanes.saturating_mul(var.ty.scalar.bits() / 8);
                let size = core::num::NonZeroU32::new(len).ok_or_else(|| err("an array of no elements"))?;
                l.ty(TypeInner::Array {
                    base: element,
                    size: ArraySize::Constant(size),
                    stride,
                })
            }
        };
        let local = l.func.local_variables.append(
            LocalVariable {
                name: Some(var.name.clone()),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        l.vars.push(local);
        // an initial value (an immediate constant table) is stored before the body
        if !var.init.is_empty() {
            let mut elements = Vec::new();
            for bits in &var.init {
                let lanes = bits
                    .iter()
                    .take(usize::from(var.ty.lanes))
                    .map(|b| l.literal_bits(var.ty.scalar, *b))
                    .collect::<Result<Vec<_>, _>>()?;
                elements.push(match lanes.as_slice() {
                    [one] => *one,
                    _ => l.expr(E::Compose {
                        ty: element,
                        components: lanes,
                    }),
                });
            }
            let value = match (var.len, elements.as_slice()) {
                (None, [one]) => *one,
                _ => {
                    // elements past those given start as zero
                    let len = var.len.unwrap_or(1) as usize;
                    while elements.len() < len {
                        elements.push(l.expr(E::ZeroValue(element)));
                    }
                    l.expr(E::Compose { ty, components: elements })
                }
            };
            l.store_local(local, value)?;
        }
    }
    l.outputs = if direct(body) {
        Outputs::Direct(vec![None; l.members.len()])
    } else {
        let mut locals = Vec::new();
        for &(_, own) in &l.members.clone() {
            locals.push(l.local(value_type(own))?);
        }
        Outputs::Locals(locals)
    };
    l.region(body, Yield::Top)?;
    let block = l.close()?;
    l.func.body = block;
    l.func.result = l.result.map(|ty| naga::FunctionResult { ty, binding: None });
    l.func.name = Some("main".into());
    let mut function = core::mem::take(&mut l.func);
    name_shared(&l.module, stage, &mut function, &l.values)?;
    l.module.entry_points.push(EntryPoint {
        name: "main".into(),
        stage,
        early_depth_test: None,
        workgroup_size: [0; 3],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    Ok(l.module)
}

/// Whether naga's writers write `e` once, as a `let`, given how many times it is referenced: naga's
/// `back::Expression::bake_ref_count` (accesses never, samples, loads and derivatives at one reference, the rest at
/// two), or a name.
fn baked(f: &naga::Function, refs: &[usize], h: Handle<E>) -> bool {
    let at = refs.get(h.index()).copied().unwrap_or(0);
    let wanted = match f.expressions[h] {
        E::Access { .. } | E::AccessIndex { .. } => usize::MAX,
        E::ImageSample { .. } | E::ImageLoad { .. } | E::Derivative { .. } | E::Load { .. } => 1,
        _ => 2,
    };
    at >= wanted || f.named_expressions.contains_key(&h)
}

/// Name every access referenced more than once whose base the writer would write in place, so each IR value is
/// computed once in the text. naga's writers never give an access a `let` of its own (a pointer access must stay a
/// place), so a lane of an unnamed expression used twice would repeat the whole expression at each use; a value
/// here is never a pointer, so naming it is safe. Named for its IR value (`vN`) where it is one.
fn name_shared(
    module: &naga::Module,
    stage: ShaderStage,
    f: &mut naga::Function,
    values: &[Option<Handle<E>>],
) -> Result<(), Error> {
    // the reference counts are the validator's, of the function as the entry point it becomes
    let ep = naga::EntryPoint {
        name: "main".into(),
        stage,
        early_depth_test: None,
        workgroup_size: [0; 3],
        workgroup_size_overrides: None,
        function: f.clone(),
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    };
    let mut probe = module.clone();
    probe.entry_points.push(ep);
    // an analysis, not the target's check: with every capability lowering can need
    let info = crate::validate(&probe, &crate::Profile::NAGA)?;
    let fi = info.get_entry_point(0);
    let refs: Vec<usize> = f.expressions.iter().map(|(h, _)| fi[h].ref_count).collect();
    let mut value_of = alloc::collections::BTreeMap::new();
    for (n, h) in values.iter().enumerate() {
        if let Some(h) = h {
            value_of.entry(*h).or_insert(n);
        }
    }
    let leaf = |e: &E| {
        matches!(
            e,
            E::FunctionArgument(_)
                | E::GlobalVariable(_)
                | E::LocalVariable(_)
                | E::Literal(_)
                | E::Constant(_)
                | E::ZeroValue(_)
        )
    };
    let handles: Vec<Handle<E>> = f.expressions.iter().map(|(h, _)| h).collect();
    for h in handles {
        let (E::Access { base, .. } | E::AccessIndex { base, .. }) = f.expressions[h] else {
            continue;
        };
        if refs.get(h.index()).copied().unwrap_or(0) < 2
            || leaf(&f.expressions[base])
            || baked(f, &refs, base)
        {
            continue;
        }
        let name = value_of
            .get(&h)
            .map_or_else(|| format!("x{}", h.index()), |n| format!("v{n}"));
        f.named_expressions.insert(h, name);
    }
    Ok(())
}
