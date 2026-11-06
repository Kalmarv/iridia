//! Operands: a register's four raw lanes, the swizzle and modifier of a source, the write mask of a destination.

use iridia_ir::{
    Address, Binary, Cmp, Op, OutOfBounds, Res, Scalar, Type, Unary, ValueId, interface::SystemValue,
};

use super::{Lift, lanes_of};
use crate::Result;
use dxbc_reader::decode::{Index, Operand, Register, Selection};

/// A lane number as a `u8`: a lane is one of at most four.
pub(super) fn lane_u8(n: usize) -> u8 {
    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Component `lane` of four, where a swizzle's lane selectors are two bits wide.
pub(super) fn of_four<T: Copy + Default>(four: [T; 4], lane: u8) -> T {
    four.get(usize::from(lane)).copied().unwrap_or_default()
}

/// An instruction's operand by position: `dxbc_reader::decode` checks the count against the opcode's format.
pub(super) trait OperandAt {
    fn operand(&self, n: usize) -> Result<&Operand>;
}
impl OperandAt for dxbc_reader::decode::Instruction {
    fn operand(&self, n: usize) -> Result<&Operand> {
        self.operands.get(n).ok_or_else(|| {
            crate::error(
                self.word,
                format!(
                    "{}: operand {n}, where the instruction has {}",
                    self.name(),
                    self.operands.len()
                ),
            )
        })
    }
}

impl Lift<'_> {
    /// A swizzle of `v`'s lanes: a scalar for one.
    pub(super) fn swizzle(&mut self, v: ValueId, lanes: &[u8]) -> Result<ValueId> {
        let ty = self.ty(v)?;
        let n = lane_u8(lanes.len());
        if n == ty.lanes && lanes.iter().zip(0u8..).all(|(&l, k)| l == k) {
            return Ok(v);
        }
        let mut all = [0u8; 4];
        for (to, from) in all.iter_mut().zip(lanes) {
            *to = *from;
        }
        Ok(self.b.value(Type::vector(ty.scalar, n), Op::Swizzle(v, all)))
    }
    pub(super) fn lane(&mut self, v: ValueId, n: u8) -> Result<ValueId> {
        self.swizzle(v, &[n])
    }
    /// `v` as the bits of `scalar` (a value of that type is itself).
    pub(super) fn bitcast(&mut self, v: ValueId, scalar: Scalar) -> Result<ValueId> {
        let ty = self.ty(v)?;
        if ty.scalar == scalar {
            return Ok(v);
        }
        Ok(self.b.value(ty.with(scalar), Op::Bitcast(v)))
    }
    /// A literal of `lanes` lanes of `scalar`, each `bits`.
    pub(super) fn splat_bits(&mut self, scalar: Scalar, lanes: u8, bits: u32) -> ValueId {
        self.b.literal(Type::vector(scalar, lanes), [u64::from(bits); 4])
    }
    pub(super) fn unary(&mut self, u: Unary, a: ValueId) -> Result<ValueId> {
        self.op(Op::Unary(u, a))
    }
    pub(super) fn binary(&mut self, op: Binary, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.op(Op::Binary(op, a, b))
    }
    pub(super) fn select(&mut self, cond: ValueId, accept: ValueId, reject: ValueId) -> Result<ValueId> {
        self.op(Op::Select { cond, accept, reject })
    }
    /// Lanes of `scalar` `bits` as wide as `like`.
    pub(super) fn like(&mut self, like: ValueId, scalar: Scalar, bits: u32) -> Result<ValueId> {
        let n = self.ty(like)?.lanes;
        Ok(self.splat_bits(scalar, n, bits))
    }

    /// Index `n` of an operand, which the decoder checked the register has the dimensions for.
    pub(super) fn index_of<'o>(&self, op: &'o Operand, n: usize) -> Result<&'o Index> {
        op.indices.get(n).ok_or_else(|| {
            crate::error(
                self.word,
                format!(
                    "{:?} has {} indices, where it is read at index {n}",
                    op.register,
                    op.indices.len()
                ),
            )
        })
    }

    /// The `u32` index of a register operand: its base plus the relative register, if any.
    pub(super) fn index_value(&mut self, ix: &Index) -> Result<ValueId> {
        let base = self.b.u32(ix.base);
        match &ix.relative {
            None => Ok(base),
            Some(r) => {
                let relative = self.read(r, 1, Scalar::U32)?;
                self.binary(Binary::IAdd, base, relative)
            }
        }
    }

    /// The `dcl_index_range` of input or output registers that holds register `base`: its first register and count.
    fn index_range(&self, output: bool, base: u32) -> Option<(u32, u32)> {
        self.setup
            .index_ranges
            .iter()
            .find(|(o, first, count)| *o == output && (*first..first.saturating_add(*count)).contains(&base))
            .map(|(_, first, count)| (*first, *count))
    }

    /// A register addressed by a computed index: the register of its `dcl_index_range` the index names, and zero
    /// outside the range.
    fn relative_register(&mut self, op: &Operand, output: bool) -> Result<ValueId> {
        let Some((first, count)) = self.index_range(output, self.index_of(op, 0)?.base) else {
            return self.refuse("a register addressed by a computed index outside any dcl_index_range");
        };
        let index = self.index_value(&self.index_of(op, 0)?.clone())?;
        let mut v = self.splat_bits(Scalar::U32, 4, 0);
        for r in first..first.saturating_add(count) {
            let value = if output {
                match self.outputs.get(&r).copied() {
                    Some(o) => self.b.value(Type::U32X4, Op::LoadVar(o)),
                    None => continue,
                }
            } else {
                match self.inputs.get(&r).copied() {
                    Some(i) => i,
                    None => continue,
                }
            };
            let number = self.b.u32(r);
            let here = self.binary(Binary::UCmp(Cmp::Eq), index, number)?;
            v = self.select(here, value, v)?;
        }
        Ok(v)
    }

    /// The register's four raw lanes.
    fn reg_value(&mut self, op: &Operand) -> Result<ValueId> {
        match op.register {
            Register::Temp => {
                let Some(&var) = self.temps.get(self.index_of(op, 0)?.base as usize) else {
                    return self.refuse("a temporary beyond dcl_temps");
                };
                Ok(self.b.value(Type::U32X4, Op::LoadVar(var)))
            }
            Register::Input => {
                if self.index_of(op, 0)?.relative.is_some() {
                    return self.relative_register(op, false);
                }
                let register = self.index_of(op, 0)?.base;
                match self.inputs.get(&register) {
                    Some(&v) => Ok(v),
                    None => self.refuse(format!("input register v{register} is not in the signature")),
                }
            }
            Register::Output => {
                if self.index_of(op, 0)?.relative.is_some() {
                    return self.relative_register(op, true);
                }
                let register = self.index_of(op, 0)?.base;
                match self.outputs.get(&register) {
                    Some(&var) => Ok(self.b.value(Type::U32X4, Op::LoadVar(var))),
                    None => self.refuse(format!("output register o{register} is read but not declared")),
                }
            }
            Register::IndexableTemp => {
                let Some(&(var, _)) = self.indexable.get(&self.index_of(op, 0)?.base) else {
                    return self.refuse("an undeclared indexable temporary");
                };
                let index = self.index_value(&self.index_of(op, 1)?.clone())?;
                Ok(self
                    .b
                    .value(Type::U32X4, Op::LoadElement(var, index, OutOfBounds::Zero)))
            }
            Register::Constant => {
                let slot = self.index_of(op, 0)?.base;
                let Some(id) = self.resources.constant_buffers.get(&slot).copied() else {
                    return self.refuse(format!("cb{slot} is used but not declared"));
                };
                let row = self.index_of(op, 1)?.clone();
                let address = match row.relative {
                    None => Address::fixed(row.base.saturating_mul(16)),
                    Some(_) => Address {
                        index: Some(self.index_value(&row)?),
                        stride: 16,
                        offset: 0,
                    },
                };
                Ok(self.b.value(
                    Type::U32X4,
                    Op::LoadBuffer {
                        resource: Res::of(id),
                        address,
                        oob: OutOfBounds::Zero,
                    },
                ))
            }
            Register::ImmediateBuffer => {
                let Some((var, _)) = self.resources.immediate else {
                    return self.refuse("the immediate constant buffer is used but not declared");
                };
                let index = self.index_value(&self.index_of(op, 0)?.clone())?;
                Ok(self
                    .b
                    .value(Type::U32X4, Op::LoadElement(var, index, OutOfBounds::Zero)))
            }
            Register::PrimitiveId
            | Register::InputCoverage
            | Register::ThreadId
            | Register::ThreadGroupId
            | Register::ThreadIdInGroup
            | Register::ThreadIdInGroupFlattened => {
                let system = match op.register {
                    Register::PrimitiveId => SystemValue::PrimitiveIndex,
                    Register::InputCoverage => SystemValue::Coverage,
                    Register::ThreadId => SystemValue::DispatchThreadId,
                    Register::ThreadGroupId => SystemValue::GroupId,
                    Register::ThreadIdInGroup => SystemValue::GroupThreadId,
                    _ => SystemValue::GroupIndex,
                };
                match self.specials.get(&system) {
                    Some(&v) => Ok(v),
                    None => self.refuse(format!("{:?} is used but not declared", op.register)),
                }
            }
            other => self.refuse(format!("a source operand of register type {other:?}")),
        }
    }

    pub(super) fn dest_mask(&self, op: &Operand) -> u8 {
        match op.selection {
            Selection::Mask(m) => m,
            _ => 1,
        }
    }

    /// The source for the destination lanes of `mask`, as `scalar`.
    pub(super) fn read(&mut self, op: &Operand, mask: u8, scalar: Scalar) -> Result<ValueId> {
        self.read_lanes(op, &lanes_of(mask), scalar)
    }

    /// The source for the destination lanes `lanes` (the swizzle is indexed by destination lane), as `scalar`, its
    /// modifier applied.
    pub(super) fn read_lanes(&mut self, op: &Operand, lanes: &[u8], scalar: Scalar) -> Result<ValueId> {
        if lanes.is_empty() {
            return self.refuse("a read of no lanes: the destination's write mask is empty");
        }
        let components: Vec<u8> = lanes
            .iter()
            .map(|&l| match op.selection {
                Selection::Swizzle(s) => Ok(of_four(s, l)),
                Selection::Scalar(c) => Ok(c),
                Selection::Mask(_) if op.register == Register::Immediate => Ok(l),
                _ => Err(()),
            })
            .collect::<std::result::Result<_, _>>()
            .or_else(|()| self.refuse("a numeric source without a swizzle"))?;
        if op.precision != 0 {
            self.notes
                .insert("min-precision operands are computed at 32 bits".into());
        }
        let n = lane_u8(components.len());
        let value = if op.register == Register::Immediate {
            let mut bits = [0u64; 4];
            for (b, &c) in bits.iter_mut().zip(&components) {
                *b = u64::from(of_four(op.immediate, c));
            }
            self.b.literal(Type::vector(scalar, n), bits)
        } else {
            let base = self.reg_value(op)?;
            let picked = self.swizzle(base, &components)?;
            self.bitcast(picked, scalar)?
        };
        Ok(match (op.modifier, scalar) {
            (0, _) => value,
            (1, Scalar::F32) => self.unary(Unary::FNeg, value)?,
            (1 | 4, Scalar::I32) => self.unary(Unary::INeg, value)?,
            (1 | 4, Scalar::U32) => {
                let signed = self.bitcast(value, Scalar::I32)?;
                let negated = self.unary(Unary::INeg, signed)?;
                self.bitcast(negated, Scalar::U32)?
            }
            (2, Scalar::F32) => self.unary(Unary::FAbs, value)?,
            (2, Scalar::I32) => {
                // |x|: x where it is not negative, else its negation
                let zero = self.like(value, Scalar::I32, 0)?;
                let negative = self.binary(Binary::SCmp(Cmp::Lt), value, zero)?;
                let negated = self.unary(Unary::INeg, value)?;
                self.select(negative, negated, value)?
            }
            (3, Scalar::F32) => {
                let magnitude = self.unary(Unary::FAbs, value)?;
                self.unary(Unary::FNeg, magnitude)?
            }
            (m, s) => return self.refuse(format!("source modifier {m} on a {s:?} operand")),
        })
    }

    /// Write `v` (as wide as the destination's mask, or one lane for all of them) to the destination's masked lanes;
    /// `saturate` clamps a float result first.
    pub(super) fn write(&mut self, op: &Operand, v: ValueId, saturate: bool) -> Result<()> {
        if op.register == Register::Null {
            return Ok(());
        }
        let mask = self.dest_mask(op);
        let lanes = lanes_of(mask);
        if lanes.is_empty() {
            return self.refuse("a destination whose write mask is empty");
        }
        let mut v = v;
        let ty = self.ty(v)?;
        if ty.lanes == 1 && lanes.len() > 1 {
            v = self.b.value(Type::vector(ty.scalar, lane_u8(lanes.len())), Op::Splat(v));
        }
        if usize::from(self.ty(v)?.lanes) != lanes.len() {
            return self.refuse("a result not as wide as its destination's mask (a lifter bug)");
        }
        if saturate {
            if self.ty(v)?.scalar != Scalar::F32 {
                return self.refuse("saturation of a result that is not a float");
            }
            v = self.unary(Unary::Saturate(iridia_ir::SaturateNan::Zero), v)?;
        }
        let bits = self.bitcast(v, Scalar::U32)?;
        match op.register {
            Register::Temp => {
                let Some(&var) = self.temps.get(self.index_of(op, 0)?.base as usize) else {
                    return self.refuse("a temporary beyond dcl_temps");
                };
                self.store_lanes(var, mask, bits)
            }
            Register::Output => {
                if self.index_of(op, 0)?.relative.is_some() {
                    let Some((first, count)) = self.index_range(true, self.index_of(op, 0)?.base) else {
                        return self.refuse("an output addressed by a computed index outside any dcl_index_range");
                    };
                    let index = self.index_value(&self.index_of(op, 0)?.clone())?;
                    for r in first..first.saturating_add(count) {
                        let Some(&var) = self.outputs.get(&r) else {
                            continue;
                        };
                        let number = self.b.u32(r);
                        let here = self.binary(Binary::UCmp(Cmp::Eq), index, number)?;
                        self.b.open_if(here);
                        self.store_lanes(var, mask, bits)?;
                        self.b
                            .close(iridia_ir::End::Yield(vec![]), &[])
                            .map_err(|e| crate::error(self.word, e))?;
                    }
                    return Ok(());
                }
                let register = self.index_of(op, 0)?.base;
                let Some(&var) = self.outputs.get(&register) else {
                    return self.refuse(format!("a write to o{register}, which is not declared"));
                };
                self.store_lanes(var, mask, bits)
            }
            Register::IndexableTemp => {
                let Some(&(var, _)) = self.indexable.get(&self.index_of(op, 0)?.base) else {
                    return self.refuse("an undeclared indexable temporary");
                };
                let index = self.index_value(&self.index_of(op, 1)?.clone())?;
                let whole = if mask == 15 {
                    bits
                } else {
                    let old = self
                        .b
                        .value(Type::U32X4, Op::LoadElement(var, index, OutOfBounds::Zero));
                    self.merge(old, mask, bits)?
                };
                self.b.store_element(var, index, whole, OutOfBounds::Zero);
                Ok(())
            }
            Register::OutputDepth | Register::OutputDepthGreaterEqual | Register::OutputDepthLessEqual => {
                let Some(var) = self.depth else {
                    return self.refuse("the depth output is written but not declared");
                };
                self.store_lanes(var, mask, bits)
            }
            Register::OutputCoverage => {
                let Some(var) = self.coverage else {
                    return self.refuse("the coverage output is written but not declared");
                };
                self.store_lanes(var, mask, bits)
            }
            other => self.refuse(format!("a destination operand of register type {other:?}")),
        }
    }

    /// `old`'s four lanes with those of `mask` taken from `bits` in order.
    fn merge(&mut self, old: ValueId, mask: u8, bits: ValueId) -> Result<ValueId> {
        let mut parts = Vec::with_capacity(4);
        let mut k = 0u8;
        for n in 0u8..4 {
            parts.push(if mask & (1 << n) != 0 {
                let p = self.lane(bits, k)?;
                k = k.saturating_add(1);
                p
            } else {
                self.lane(old, n)?
            });
        }
        self.op(Op::Compose(parts))
    }

    /// Store `bits` (`u32`, one lane per lane of `mask`) into the lanes of `mask` of a register variable.
    pub(super) fn store_lanes(&mut self, var: iridia_ir::VarId, mask: u8, bits: ValueId) -> Result<()> {
        let whole = if mask == 15 {
            bits
        } else {
            let old = self.b.value(Type::U32X4, Op::LoadVar(var));
            self.merge(old, mask, bits)?
        };
        self.b.store_var(var, whole);
        Ok(())
    }
}
