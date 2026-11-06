//! The lift: each register a variable, each instruction its operations in order, the outputs stored at the end.
//! Arithmetic is in `arith`, texture operations in `texture`, flow control and predication in `flow`.

mod arith;
mod flow;
mod texture;

use std::collections::BTreeMap;

use d3d9_reader::Modifier;
use iridia_ir::{
    Address, Binary, Builder, Cmp, Conventions, End, FloatEnv, FloatToInt, LabelId, MulZero, Op, OutOfBounds,
    Precision, Program, Provenance, Res, Rounding, SaturateNan, Scalar, Type, Unary, ValueId, VarId,
    interface::{ShaderFormat, TextureDim},
    source_hash,
};

use crate::{
    Destination, Index, Instruction, Register, Result, Shader, Signature, Source, error,
    signature::{outputs, read_input, signature},
};

/// The lifter's name in a program's provenance.
pub const LIFTER: &str = "iridia-d3d9-ir";

/// Increment when the same tokens lift differently.
pub const LIFTER_VERSION: u32 = 4;

/// The compatibility policies every program states, beside its multiply rule.
const POLICIES: [&str; 2] = ["d3d9.min_max_nan=min_num", "d3d9.saturate_nan=zero"];

/// How a pixel 1.x program's registers keep their range. Microsoft's "ps_1_X Registers" page gives `r#` a range of
/// ±`PixelShader1xMaxValue` (a device cap), `t#` used as a temporary ±`MaxTextureRepeat`, and says early hardware held
/// about eight fractional bits; the reads of `c#` (-1 to 1) and `v#` (0 to 1) are clamped under every policy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ps1x {
    /// The `r#` range `m`, each write clamped to `-m..=m`; `None` (the default, as modern drivers and DXVK, whose cap
    /// is `FLT_MAX`) for no clamp.
    pub register_max: Option<f32>,
    /// The range of `t#` written as a temporary (pixel 1.1 to 1.3), likewise.
    pub texture_max: Option<f32>,
    /// Fixed point after each write: the fractional bits kept and the rounding; `None` (the default) for none.
    pub quantize: Option<(u8, Rounding)>,
}

/// What the lift is told beyond the tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The rule of every multiplication an instruction makes (`mul`, `mad`, the dot products, the matrix operations,
    /// `lrp`, and the scaling of the modifiers and texture operations). The default, [`MulZero::ZeroWins`], is what
    /// Direct3D 9 drivers on Windows do (vkd3d's `float-comparison` test: `0 * rcp(0)` is 0 there), and what DXVK and
    /// wined3d emulate; [`MulZero::Ieee`] gives `0 * inf` as NaN.
    pub multiply: MulZero,
    /// The pixel 1.x register policy.
    pub ps1x: Ps1x,
    /// The texture dimension of pixel 1.x texture stages the host knows, by stage. A stage it does not name gets a
    /// texture of each dimension, picked by `StateParam::TextureDim`.
    pub texture_dims: BTreeMap<u32, TextureDim>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            multiply: MulZero::ZeroWins,
            ps1x: Ps1x::default(),
            texture_dims: BTreeMap::new(),
        }
    }
}

/// A write waiting for the end of its co-issued pair.
#[derive(Clone, Copy, Debug)]
struct Pending {
    register: Register,
    mask: u8,
    value: ValueId,
    /// The predicate of a predicated write, a bool of each lane.
    predicate: Option<ValueId>,
}

/// An open construct of the source's flow control.
#[derive(Clone, Debug)]
enum Frame {
    /// `if`, and whether `else` was met.
    If { otherwise: bool },
    /// `rep` or `loop`: the loop's label, and for `loop` its `aL` variable and step.
    Loop {
        label: LabelId,
        counter: Option<(VarId, ValueId)>,
    },
    /// An inlined subroutine: the block a `ret` leaves.
    Call { label: LabelId },
}

struct Lift<'a> {
    source: &'a Shader,
    options: &'a Options,
    signature: Signature,
    b: Builder,
    /// The variable of each register written so far, with the lanes written.
    registers: BTreeMap<Register, (VarId, u8)>,
    /// Each input register as four lanes, the undeclared ones zero.
    inputs: BTreeMap<Register, ValueId>,
    /// Each `def` constant's value.
    definitions: BTreeMap<u32, ValueId>,
    /// Each `defi` constant's value.
    integers: BTreeMap<u32, ValueId>,
    /// Each `defb` constant's value.
    booleans: BTreeMap<u32, ValueId>,
    /// Writes not yet committed: a pixel 1.x instruction's, until the one after it that is not co-issued.
    pending: Vec<Pending>,
    /// The open constructs, innermost last.
    flow: Vec<Frame>,
    /// The subroutines, by label: the instructions after their `label`.
    subroutines: BTreeMap<u32, std::ops::Range<usize>>,
    /// The labels of the subroutines being inlined, innermost last.
    calls: Vec<u32>,
    /// The dot products of `texm3x2pad` and `texm3x3pad` so far.
    rows: Vec<ValueId>,
    notes: Vec<String>,
    at: usize,
}

/// The mask of every lane of a varying `width` lanes wide.
fn every(width: impl TryInto<u32>) -> u8 {
    let width = width.try_into().unwrap_or(0).min(4);
    u8::try_from((1u32 << width).saturating_sub(1)).unwrap_or(0)
}

fn splat4(b: &mut Builder, x: f32) -> ValueId {
    b.f32x4([x; 4])
}

/// The type of a register's variable.
fn register_type(r: Register) -> Type {
    match r {
        Register::Address => Type::vector(Scalar::I32, 4),
        Register::Predicate => Type::vector(Scalar::Bool, 4),
        _ => Type::F32X4,
    }
}

/// The f32 bits of `x`, for a clamp's bounds.
fn bits(x: f32) -> u32 {
    x.to_bits()
}

impl Lift<'_> {
    fn op(&mut self, op: Op) -> Result<ValueId> {
        self.b.op(op).map_err(|e| error(self.at, e))
    }
    fn unary(&mut self, u: Unary, a: ValueId) -> Result<ValueId> {
        self.op(Op::Unary(u, a))
    }
    fn binary(&mut self, op: Binary, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.op(Op::Binary(op, a, b))
    }
    fn select(&mut self, cond: ValueId, accept: ValueId, reject: ValueId) -> Result<ValueId> {
        self.op(Op::Select { cond, accept, reject })
    }
    fn ty(&self, v: ValueId) -> Result<Type> {
        self.b
            .ty(v)
            .ok_or_else(|| error(self.at, format!("value %{} has no type", v.0)))
    }
    /// Lanes of a value, by index: a scalar for one.
    fn swizzle(&mut self, v: ValueId, lanes: &[u8]) -> Result<ValueId> {
        let ty = self.ty(v)?;
        let n = u8::try_from(lanes.len()).map_err(|_| error(self.at, "a swizzle of more than four lanes"))?;
        let mut all = [0u8; 4];
        for (to, from) in all.iter_mut().zip(lanes) {
            *to = *from;
        }
        Ok(self.b.value(Type::vector(ty.scalar, n), Op::Swizzle(v, all)))
    }
    fn lane(&mut self, v: ValueId, n: u8) -> Result<ValueId> {
        self.swizzle(v, &[n])
    }
    fn splat(&mut self, v: ValueId, lanes: u8) -> Result<ValueId> {
        let ty = self.ty(v)?;
        Ok(self.b.value(Type::vector(ty.scalar, lanes), Op::Splat(v)))
    }
    fn mul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.binary(Binary::FMul(self.options.multiply), a, b)
    }
    fn var(&mut self, r: Register) -> VarId {
        match self.registers.get(&r) {
            Some(&(v, _)) => v,
            None => {
                let v = self.b.var(&r.name(), register_type(r));
                self.registers.insert(r, (v, 0));
                v
            }
        }
    }
    /// Store `v`'s lanes of `mask` into a register's variable, its other lanes kept; where `predicate` is given, only
    /// the lanes it holds in.
    fn store(&mut self, r: Register, mask: u8, mut v: ValueId, predicate: Option<ValueId>) -> Result<()> {
        let var = self.var(r);
        let ty = self.ty(v)?;
        let written = self.registers.get(&r).map_or(0, |&(_, m)| m);
        if let Some(p) = predicate {
            let old = self.b.value(ty, Op::LoadVar(var));
            v = self.select(p, v, old)?;
        }
        let whole = if mask == 15 {
            v
        } else {
            let old = self.b.value(ty, Op::LoadVar(var));
            let mut parts = Vec::new();
            for n in 0u8..4 {
                let from = if mask & (1 << n) != 0 { v } else { old };
                parts.push(self.lane(from, n)?);
            }
            self.op(Op::Compose(parts))?
        };
        self.b.store_var(var, whole);
        self.registers.insert(r, (var, written | mask));
        Ok(())
    }
    /// Commit the writes waiting for their pair.
    fn flush(&mut self) -> Result<()> {
        for p in std::mem::take(&mut self.pending) {
            self.store(p.register, p.mask, p.value, p.predicate)?;
        }
        Ok(())
    }
    /// The four lanes of `vPos` or `vFace` from the pipeline input at `slot`, the lanes past `mask` zero; `None` for
    /// another register. `vPos` is the window position half a pixel back, at the pixel's top left corner where
    /// Direct3D 9 samples it; `vFace` is 1 for a front facing primitive and -1 for a back facing one in every lane.
    fn misc_input(&mut self, r: Register, slot: u32, mask: u8) -> Result<Option<ValueId>> {
        let v = match r {
            Register::PixelPosition => {
                let position = self.b.value(Type::F32X4, Op::LoadInput(slot, 0b1111));
                let half = splat4(&mut self.b, 0.5);
                let corner = self.binary(Binary::FSub, position, half)?;
                let zero = self.b.f32(0.0);
                let mut parts = [zero; 4];
                for (n, p) in (0u8..).zip(parts.iter_mut()) {
                    if mask & (1 << n) != 0 {
                        *p = self.lane(corner, n)?;
                    }
                }
                self.op(Op::Compose(parts.to_vec()))?
            }
            Register::Face => {
                let front = self.b.value(Type::scalar(Scalar::Bool), Op::LoadInput(slot, 0b1));
                let (one, minus) = (splat4(&mut self.b, 1.0), splat4(&mut self.b, -1.0));
                self.op(Op::Select {
                    cond: front,
                    accept: one,
                    reject: minus,
                })?
            }
            _ => return Ok(None),
        };
        Ok(Some(v))
    }
    fn prepare(&mut self) -> Result<()> {
        for i in &self.source.inputs {
            let &(slot, mask) = self
                .signature
                .inputs
                .get(&i.register)
                .ok_or_else(|| error(self.at, format!("input {} has no slot", i.register.name())))?;
            let width = self
                .signature
                .interface
                .inputs
                .get(slot as usize)
                .map_or(0, |v| v.value.width);
            if let Some(v) = self.misc_input(i.register, slot, mask)? {
                self.inputs.insert(i.register, v);
                continue;
            }
            let arg = self.b.value(
                Type::vector(Scalar::F32, width),
                Op::LoadInput(slot, every(width)),
            );
            let zero = self.b.f32(0.0);
            let mut parts = [zero; 4];
            let mut lane = 0;
            for (n, p) in (0u8..).zip(parts.iter_mut()) {
                if mask & (1 << n) != 0 {
                    *p = if width == 1 { arg } else { self.lane(arg, lane)? };
                    lane = lane.saturating_add(1);
                }
            }
            let mut expanded = self.op(Op::Compose(parts.to_vec()))?;
            // pixel 1.x colours read within 0 to 1
            if self.source.pixel_1x() && matches!(i.register, Register::Input(_)) {
                expanded = self.clamp(expanded, 0.0, 1.0)?;
            }
            self.inputs.insert(i.register, expanded);
        }
        let pixel_1x = self.source.pixel_1x();
        for (&n, words) in &self.source.definitions {
            // pixel 1.x constants read within -1 to 1, a `def` too, clamped here
            let words = if pixel_1x {
                words.map(|w| f32::from_bits(w).clamp(-1.0, 1.0).to_bits())
            } else {
                *words
            };
            let v = self.b.literal(Type::F32X4, words.map(u64::from));
            self.definitions.insert(n, v);
        }
        for (&n, values) in &self.source.integers {
            let v = self.b.literal(
                Type::vector(Scalar::I32, 4),
                values.map(|x| u64::from(x.cast_unsigned())),
            );
            self.integers.insert(n, v);
        }
        for (&n, &value) in &self.source.booleans {
            let v = self.b.bool(value);
            self.booleans.insert(n, v);
        }
        Ok(())
    }
    /// `ClampRange(lo, hi)` with NaN left to the target: a fixed-point register has none.
    fn clamp(&mut self, v: ValueId, lo: f32, hi: f32) -> Result<ValueId> {
        self.unary(
            Unary::ClampRange {
                lo: bits(lo),
                hi: bits(hi),
                nan: SaturateNan::Any,
            },
            v,
        )
    }
    fn bank(&self) -> Result<crate::Bank> {
        self.signature
            .bank
            .ok_or_else(|| error(self.at, "constant bank not allocated"))
    }
    /// The four lanes of a register, `required` of them checked as written or declared.
    fn register(&mut self, r: Register, required: u8) -> Result<ValueId> {
        match r {
            Register::Input(_) | Register::PixelPosition | Register::Face => self.input(r, required),
            // a pixel 1.4 or 2.x texture register is the texture coordinate input; before 1.4 a temporary
            Register::Texture(_) if !(self.source.pixel_1x() && self.source.version.minor < 4) => {
                self.input(r, required)
            }
            Register::Constant(n) => {
                if let Some(&v) = self.definitions.get(&n) {
                    return Ok(v);
                }
                let bank = self.bank()?;
                let load = Op::LoadBuffer {
                    resource: Res::of(bank.resource),
                    address: Address::fixed(n.saturating_mul(16)),
                    oob: OutOfBounds::Zero,
                };
                let v = self.b.value(Type::F32X4, load);
                if self.source.pixel_1x() {
                    return self.clamp(v, -1.0, 1.0);
                }
                Ok(v)
            }
            Register::Temporary(_) | Register::Texture(_) => {
                let &(var, mask) = self
                    .registers
                    .get(&r)
                    .ok_or_else(|| error(self.at, format!("uninitialized {}", r.name())))?;
                if required & !mask != 0 {
                    return Err(error(
                        self.at,
                        format!("{}: uninitialized lanes {:#x}", r.name(), required & !mask),
                    ));
                }
                Ok(self.b.value(Type::F32X4, Op::LoadVar(var)))
            }
            _ => Err(error(
                self.at,
                format!("not a readable float register: {}", r.name()),
            )),
        }
    }
    fn input(&mut self, r: Register, required: u8) -> Result<ValueId> {
        read_input(&mut self.signature, r, required, self.at)?;
        self.inputs
            .get(&r)
            .copied()
            .ok_or_else(|| error(self.at, format!("undeclared input {}", r.name())))
    }
    /// A source operand: the register (or relative constant), `_dz` and `_dw`, the swizzle, then the other modifier.
    fn source(&mut self, s: Source, needed: u8) -> Result<ValueId> {
        let required = s.read_mask(needed);
        let mut v = match (s.relative, s.register) {
            (Some(index), Register::Constant(base)) => self.relative_constant(base, index)?,
            (Some(index), Register::Input(base)) => self.relative_input(base, index, required)?,
            (Some(_), r) => {
                return Err(error(
                    self.at,
                    format!("relative addressing of {} is not lifted", r.name()),
                ));
            }
            _ => self.register(s.register, required)?,
        };
        if let Modifier::DivideZ | Modifier::DivideW = s.modifier {
            let by = if s.modifier == Modifier::DivideZ { 2 } else { 3 };
            let divisor = self.lane(v, by)?;
            let divisor = self.splat(divisor, 4)?;
            v = self.binary(Binary::FDiv, v, divisor)?;
        }
        if s.swizzle != [0, 1, 2, 3] {
            v = self.swizzle(v, &s.swizzle)?;
        }
        if s.absolute() {
            v = self.unary(Unary::FAbs, v)?;
        }
        match s.modifier {
            Modifier::Bias | Modifier::BiasNegate => {
                let h = splat4(&mut self.b, 0.5);
                v = self.binary(Binary::FSub, v, h)?;
            }
            Modifier::Sign | Modifier::SignNegate => {
                let h = splat4(&mut self.b, 0.5);
                let biased = self.binary(Binary::FSub, v, h)?;
                let two = splat4(&mut self.b, 2.0);
                v = self.mul(biased, two)?;
            }
            Modifier::Complement => {
                let one = splat4(&mut self.b, 1.0);
                v = self.binary(Binary::FSub, one, v)?;
            }
            Modifier::X2 | Modifier::X2Negate => {
                let two = splat4(&mut self.b, 2.0);
                v = self.mul(v, two)?;
            }
            _ => {}
        }
        if s.negate() {
            v = self.unary(Unary::FNeg, v)?;
        }
        Ok(v)
    }
    /// The `i32` index a relative operand adds to its register number.
    fn index(&mut self, index: Index) -> Result<ValueId> {
        match index {
            Index::Address(lane) => {
                let &(var, written) = self
                    .registers
                    .get(&Register::Address)
                    .ok_or_else(|| error(self.at, "a relative constant before any mova wrote a0"))?;
                if written & (1 << lane) == 0 {
                    return Err(error(
                        self.at,
                        format!("a0 lane {lane} is read before mova writes it"),
                    ));
                }
                let a0 = self.b.value(Type::vector(Scalar::I32, 4), Op::LoadVar(var));
                self.lane(a0, lane)
            }
            Index::Loop => {
                let var = self
                    .flow
                    .iter()
                    .rev()
                    .find_map(|f| match f {
                        Frame::Loop {
                            counter: Some((var, _)),
                            ..
                        } => Some(*var),
                        _ => None,
                    })
                    .ok_or_else(|| error(self.at, "aL read outside a loop"))?;
                Ok(self.b.value(Type::I32, Op::LoadVar(var)))
            }
        }
    }
    /// The input `v[index + base]`: the declared input register the index names whose lanes cover `required`, and zero
    /// for any other.
    fn relative_input(&mut self, base: u32, index: Index, required: u8) -> Result<ValueId> {
        let index = self.index(index)?;
        let mut declared: BTreeMap<u32, u8> = BTreeMap::new();
        for input in &self.source.inputs {
            if let Register::Input(n) = input.register
                && n >= base
            {
                *declared.entry(n).or_default() |= input.mask;
            }
        }
        let mut v = splat4(&mut self.b, 0.0);
        for (n, mask) in declared {
            if required & !mask != 0 {
                continue;
            }
            let value = self.input(Register::Input(n), required)?;
            let offset = self
                .b
                .i32(i32::try_from(n.saturating_sub(base)).map_err(|_| error(self.at, "input register"))?);
            let hit = self.binary(Binary::SCmp(Cmp::Eq), index, offset)?;
            v = self.select(hit, value, v)?;
        }
        Ok(v)
    }
    /// The float constant `c[index + base]`: the row of the bank (or the `def` that sits at that register) at the
    /// index plus `base`, and zero outside the bank.
    fn relative_constant(&mut self, base: u32, index: Index) -> Result<ValueId> {
        let a = self.index(index)?;
        let bank = self.bank()?;
        let offset = self
            .b
            .i32(i32::try_from(base).map_err(|_| error(self.at, "constant register"))?);
        let index = self.binary(Binary::IAdd, a, offset)?;
        let load = Op::LoadBuffer {
            resource: Res::of(bank.resource),
            address: Address {
                index: Some(index),
                stride: 16,
                offset: 0,
            },
            oob: OutOfBounds::Zero,
        };
        let mut v = self.b.value(Type::F32X4, load);
        // a `def` constant is not in the bank: the register it names reads as its value
        let definitions: Vec<_> = self.definitions.iter().map(|(&n, &value)| (n, value)).collect();
        for (n, value) in definitions {
            let n = self
                .b
                .i32(i32::try_from(n).map_err(|_| error(self.at, "constant register"))?);
            let hit = self.binary(Binary::SCmp(Cmp::Eq), index, n)?;
            v = self.select(hit, value, v)?;
        }
        Ok(v)
    }
    /// An integer constant's four lanes.
    fn integer(&mut self, n: u32) -> Result<ValueId> {
        if let Some(&v) = self.integers.get(&n) {
            return Ok(v);
        }
        let bank = self
            .signature
            .integers
            .ok_or_else(|| error(self.at, "integer constant bank not allocated"))?;
        let load = Op::LoadBuffer {
            resource: Res::of(bank.resource),
            address: Address::fixed(n.saturating_mul(16)),
            oob: OutOfBounds::Zero,
        };
        Ok(self.b.value(Type::vector(Scalar::I32, 4), load))
    }
    /// A boolean constant, as a `bool`: a stored word that is not zero is true.
    fn boolean(&mut self, n: u32) -> Result<ValueId> {
        if let Some(&v) = self.booleans.get(&n) {
            return Ok(v);
        }
        let bank = self
            .signature
            .booleans
            .ok_or_else(|| error(self.at, "boolean constant bank not allocated"))?;
        let load = Op::LoadBuffer {
            resource: Res::of(bank.resource),
            address: Address::fixed(n.saturating_mul(4)),
            oob: OutOfBounds::Zero,
        };
        let word = self.b.value(Type::U32, load);
        let zero = self.b.u32(0);
        self.binary(Binary::UCmp(Cmp::Ne), word, zero)
    }
    /// Write an instruction's result: the pixel 1.x shift, the register policy, `_sat`, the fixed-point policy, then
    /// the write itself, under the instruction's predicate; a pixel 1.x write waits for its co-issued pair.
    fn write(&mut self, i: &Instruction, d: Destination, mut v: ValueId) -> Result<()> {
        if d.shift != 0 {
            let factor = if d.shift > 0 {
                f32::from(1u8 << d.shift.unsigned_abs().min(3))
            } else {
                1.0 / f32::from(1u8 << d.shift.unsigned_abs().min(3))
            };
            let factor = splat4(&mut self.b, factor);
            v = self.mul(v, factor)?;
        }
        let ps1x = self.options.ps1x;
        if self.source.pixel_1x() {
            let range = match d.register {
                Register::Temporary(_) => ps1x.register_max,
                Register::Texture(_) => ps1x.texture_max,
                _ => None,
            };
            if let Some(m) = range {
                v = self.clamp(v, -m, m)?;
            }
        }
        if d.saturate {
            v = self.unary(Unary::Saturate(SaturateNan::Zero), v)?;
        }
        if self.source.pixel_1x()
            && let Some((frac_bits, rounding)) = ps1x.quantize
        {
            v = self.unary(Unary::Quantize { frac_bits, rounding }, v)?;
        }
        let predicate = match i.predicate {
            Some(p) => Some(self.predicate(p)?),
            None => None,
        };
        if let Some(index) = d.relative {
            return self.store_relative(d, index, v, predicate);
        }
        let pending = Pending {
            register: d.register,
            mask: d.mask,
            value: v,
            predicate,
        };
        if self.source.pixel_1x() {
            self.pending.push(pending);
            Ok(())
        } else {
            self.store(pending.register, pending.mask, pending.value, pending.predicate)
        }
    }
    /// Store `v` into `o[index + n]`: into each declared output register from `on` on, under the predicate that the
    /// index names it (and the instruction's own). A write the index takes past the declared registers is lost.
    fn store_relative(
        &mut self,
        d: Destination,
        index: Index,
        v: ValueId,
        predicate: Option<ValueId>,
    ) -> Result<()> {
        let Register::Output(base) = d.register else {
            return Err(error(self.at, format!("relative write of {}", d.register.name())));
        };
        let index = self.index(index)?;
        let mut candidates: Vec<u32> = self
            .source
            .outputs
            .iter()
            .filter_map(|o| match o.register {
                Register::Output(n) if n >= base => Some(n),
                _ => None,
            })
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        for n in candidates {
            let offset = self
                .b
                .i32(i32::try_from(n.saturating_sub(base)).map_err(|_| error(self.at, "output register"))?);
            let hit = self.binary(Binary::SCmp(Cmp::Eq), index, offset)?;
            let mut lanes = self.splat(hit, 4)?;
            if let Some(p) = predicate {
                lanes = self.binary(Binary::And, lanes, p)?;
            }
            self.store(Register::Output(n), d.mask, v, Some(lanes))?;
        }
        Ok(())
    }
    /// Round each lane of `v` and convert it to `i32` for the address register.
    fn address(&mut self, v: ValueId, rounding: Rounding) -> Result<ValueId> {
        let rounded = self.unary(Unary::FRound(rounding), v)?;
        let ty = self.ty(rounded)?;
        Ok(self
            .b
            .value(ty.with(Scalar::I32), Op::Convert(rounded, FloatToInt::Saturate)))
    }
    fn finish(mut self, bytes: &[u8]) -> Result<Program> {
        self.flush()?;
        if !self.flow.is_empty() {
            return Err(error(self.at, "flow control left open at the end of the program"));
        }
        self.b.precision(Precision::Full);
        self.b.at(None);
        let written: BTreeMap<Register, u8> = self.registers.iter().map(|(&r, &(_, m))| (r, m)).collect();
        let outs = outputs(self.source, &written, self.at)?;
        for (slot, (r, v)) in (0u32..).zip(outs) {
            let mask = v.provenance.d3d.as_ref().map_or(0, |d| d.mask);
            let &(var, _) = self
                .registers
                .get(&r)
                .ok_or_else(|| error(self.at, format!("uninitialized output {}", r.name())))?;
            self.signature.interface.outputs.push(v);
            let value = self.b.value(Type::F32X4, Op::LoadVar(var));
            let lanes: Vec<u8> = (0u8..4).filter(|n| mask & (1 << n) != 0).collect();
            let value = if lanes.len() == 4 {
                value
            } else {
                self.swizzle(value, &lanes)?
            };
            self.b.store_output(slot, every(lanes.len()), value);
        }
        let entry = self.b.finish(End::Return).map_err(|e| error(self.at, e))?;
        let mut notes = self.notes;
        notes.extend(self.signature.notes);
        notes.sort();
        notes.dedup();
        let source = self.source;
        let mut policies: Vec<String> = POLICIES.iter().map(|p| (*p).into()).collect();
        policies.insert(
            0,
            match self.options.multiply {
                MulZero::Ieee => "d3d9.mul=ieee",
                MulZero::ZeroWins => "d3d9.mul=zero_wins",
            }
            .into(),
        );
        if source.pixel_1x() {
            let ps1x = self.options.ps1x;
            let range = |m: Option<f32>| m.map_or_else(|| "none".into(), |m| format!("{m:?}"));
            policies.push(format!("d3d9.ps1x_max={}", range(ps1x.register_max)));
            policies.push(format!("d3d9.ps1x_texture_max={}", range(ps1x.texture_max)));
            policies.push(format!(
                "d3d9.ps1x_quantize={}",
                ps1x.quantize.map_or_else(
                    || "none".into(),
                    |(f, r)| format!("{f}.{}", format!("{r:?}").to_lowercase())
                )
            ));
        }
        let program = Program {
            stage: source.stage,
            entry,
            patch: None,
            workgroup: None,
            interface: self.signature.interface,
            conventions: Conventions::D3D,
            float: FloatEnv::default(),
            provenance: Provenance {
                format: ShaderFormat::D3d9,
                target: source.version.to_string(),
                lifter: LIFTER.into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(source_hash(bytes)),
                instruction_count: None,
                policies,
            },
            notes,
        };
        iridia_ir::validate(&program).map_err(|problems| {
            error(
                0,
                problems
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
        Ok(program)
    }
}

/// Lift one complete Direct3D 9 program of any version this crate reads, with the default [`Options`] (pixel 1.x
/// registers as modern drivers keep them); trailing bytes and unknown features are refused.
pub fn lift(bytes: &[u8]) -> Result<Program> {
    lift_with(bytes, &Options::default())
}

/// Lift one complete program with the stated [`Options`].
pub fn lift_with(bytes: &[u8], options: &Options) -> Result<Program> {
    let source = crate::read(bytes)?;
    let signature = signature(&source, &options.texture_dims)?;
    let mut l = Lift {
        source: &source,
        options,
        signature,
        b: Builder::new("main"),
        registers: BTreeMap::new(),
        inputs: BTreeMap::new(),
        definitions: BTreeMap::new(),
        integers: BTreeMap::new(),
        booleans: BTreeMap::new(),
        pending: Vec::new(),
        flow: Vec::new(),
        subroutines: BTreeMap::new(),
        calls: Vec::new(),
        rows: Vec::new(),
        notes: Vec::new(),
        at: 0,
    };
    l.prepare()?;
    let main = l.subroutines()?;
    l.run(main)?;
    l.finish(bytes)
}
