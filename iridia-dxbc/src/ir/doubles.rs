//! Shader Model 5's double-precision instructions (the Direct3D 11.3 specification, 22.14 to 22.17). A register holds
//! two doubles, the first in `.xy` and the second in `.zw`, the low word first: the IR's 64-bit lanes, so a register's
//! raw lanes are its doubles by a bitcast. An instruction's doubles are the pairs its destination's mask names
//! (`.xy`, `.zw` or both) and the same pairs of each source after its swizzle; one that gives a 32-bit result per
//! double writes the first to the first lane of its mask and the second to the next. A source modifier is a double's.

use iridia_ir::{Binary, Cmp, FloatToInt, MinMaxNan, MulZero, Op, Scalar, Ternary, Type, Unary, ValueId};

use super::Lift;
use super::operands::{OperandAt, lane_u8, of_four};
use crate::Result;
use crate::operand::same_source;
use dxbc_reader::decode::{Instruction, Operand, Register};

/// The 32-bit lanes of doubles `which`: double k is lanes 2k and 2k + 1, the low word first.
fn words_of(which: &[u8]) -> Vec<u8> {
    which
        .iter()
        .flat_map(|&k| [k.saturating_mul(2), k.saturating_mul(2).saturating_add(1)])
        .collect()
}

impl Lift<'_> {
    /// The doubles a double destination's mask names: `.xy` the first, `.zw` the second.
    fn doubles_written(&self, op: &Operand) -> Result<Vec<u8>> {
        match self.dest_mask(op) {
            0b0011 => Ok(vec![0]),
            0b1100 => Ok(vec![1]),
            0b1111 => Ok(vec![0, 1]),
            m => self.refuse(format!(
                "a double destination of mask {m:#06b}, which is not .xy, .zw or .xyzw"
            )),
        }
    }

    /// The doubles of a 32-bit destination that takes one result per double: one for each lane of its mask (one or
    /// two lanes).
    fn doubles_into_lanes(&self, op: &Operand) -> Result<Vec<u8>> {
        match self.dest_mask(op).count_ones() {
            1 => Ok(vec![0]),
            2 => Ok(vec![0, 1]),
            n => self.refuse(format!(
                "{n} lanes written with the results of at most two doubles"
            )),
        }
    }

    /// Doubles `which` of a source (double k its lanes 2k and 2k + 1 after the swizzle), an `f64` or a `vec2<f64>`,
    /// its modifier applied as a double's.
    fn read_doubles(&mut self, op: &Operand, which: &[u8]) -> Result<ValueId> {
        let lanes = words_of(which);
        let words = if op.register == Register::Immediate64 {
            // a 64-bit immediate's words, the low then the high of each of its two doubles
            let mut bits = [0u64; 4];
            for (b, &l) in bits.iter_mut().zip(&lanes) {
                *b = u64::from(of_four(op.immediate, l));
            }
            self.b
                .literal(Type::vector(Scalar::U32, lane_u8(lanes.len())), bits)
        } else {
            let mut raw = op.clone();
            raw.modifier = 0;
            self.read_lanes(&raw, &lanes, Scalar::U32)?
        };
        let v = self.b.value(
            Type::vector(Scalar::F64, lane_u8(which.len())),
            Op::Bitcast(words),
        );
        match op.modifier {
            0 => Ok(v),
            1 => self.unary(Unary::FNeg, v),
            2 => self.unary(Unary::FAbs, v),
            3 => {
                let magnitude = self.unary(Unary::FAbs, v)?;
                self.unary(Unary::FNeg, magnitude)
            }
            m => self.refuse(format!("source modifier {m} on a double operand")),
        }
    }

    /// Write doubles `v` (as many as the destination's mask names) to a double destination, saturated first where
    /// `saturate`.
    fn write_doubles(&mut self, op: &Operand, v: ValueId, saturate: bool) -> Result<()> {
        let v = if saturate {
            self.unary(Unary::Saturate(iridia_ir::SaturateNan::Zero), v)?
        } else {
            v
        };
        let n = self.ty(v)?.lanes;
        let words = self
            .b
            .value(Type::vector(Scalar::U32, n.saturating_mul(2)), Op::Bitcast(v));
        self.write(op, words, false)
    }

    /// A double literal of `n` lanes, each `x`.
    fn doubles_like(&mut self, n: u8, x: f64) -> ValueId {
        self.b.literal(Type::vector(Scalar::F64, n), [x.to_bits(); 4])
    }

    /// A double-precision instruction.
    pub(super) fn double(&mut self, ins: &Instruction, name: &str) -> Result<()> {
        let dest = ins.operand(0)?;
        match name {
            "dadd" | "dmul" | "ddiv" | "dmin" | "dmax" => {
                let which = self.doubles_written(dest)?;
                let a = self.read_doubles(ins.operand(1)?, &which)?;
                let b = self.read_doubles(ins.operand(2)?, &which)?;
                let op = match name {
                    "dadd" => Binary::FAdd,
                    "dmul" => Binary::FMul(MulZero::Ieee),
                    "ddiv" => Binary::FDiv,
                    "dmin" => Binary::FMin(MinMaxNan::MinNum),
                    _ => Binary::FMax(MinMaxNan::MinNum),
                };
                let r = self.binary(op, a, b)?;
                self.write_doubles(dest, r, ins.saturate)
            }
            "dfma" => {
                let which = self.doubles_written(dest)?;
                let a = self.read_doubles(ins.operand(1)?, &which)?;
                let b = self.read_doubles(ins.operand(2)?, &which)?;
                let c = self.read_doubles(ins.operand(3)?, &which)?;
                let r = self.op(Op::Ternary(Ternary::Fma, a, b, c))?;
                self.write_doubles(dest, r, ins.saturate)
            }
            "drcp" => {
                let which = self.doubles_written(dest)?;
                let a = self.read_doubles(ins.operand(1)?, &which)?;
                let one = self.doubles_like(lane_u8(which.len()), 1.0);
                let r = self.binary(Binary::FDiv, one, a)?;
                self.write_doubles(dest, r, ins.saturate)
            }
            "dmov" => {
                let src = ins.operand(1)?;
                if src.modifier == 0 && !ins.saturate {
                    // without modifiers the bits are moved as they are
                    let mask = self.dest_mask(dest);
                    self.doubles_written(dest)?;
                    let a = self.read(src, mask, Scalar::U32)?;
                    return self.write(dest, a, false);
                }
                let which = self.doubles_written(dest)?;
                let a = self.read_doubles(src, &which)?;
                self.write_doubles(dest, a, ins.saturate)
            }
            "dmovc" => {
                // the k-th 32-bit lane of the condition chooses the k-th double
                let which = self.doubles_written(dest)?;
                let c = self.read_lanes(ins.operand(1)?, &which, Scalar::U32)?;
                let zero = self.like(c, Scalar::U32, 0)?;
                let cond = self.binary(Binary::UCmp(Cmp::Ne), c, zero)?;
                let a = self.read_doubles(ins.operand(2)?, &which)?;
                let b = self.read_doubles(ins.operand(3)?, &which)?;
                let r = self.select(cond, a, b)?;
                self.write_doubles(dest, r, ins.saturate)
            }
            "deq" | "dge" | "dlt" | "dne" => {
                let which = self.doubles_into_lanes(dest)?;
                let (first, second) = (ins.operand(1)?, ins.operand(2)?);
                let a = self.read_doubles(first, &which)?;
                let cmp = match name {
                    "deq" => Cmp::Eq,
                    "dge" => Cmp::Ge,
                    "dlt" => Cmp::Lt,
                    _ => Cmp::Ne,
                };
                // a double compared with itself is the NaN test, as a float is (`compared`): one value where both
                // of its words are
                let itself: Vec<bool> = which
                    .iter()
                    .map(|&k| same_source(first, second, &words_of(&[k])))
                    .collect();
                let c = self.compared(Binary::FCmp(cmp), a, &itself, |lift| {
                    lift.read_doubles(second, &which)
                })?;
                let r = self.mask_of(c)?;
                self.write(dest, r, false)
            }
            "dtof" | "dtoi" | "dtou" => {
                // a float to an integer truncated toward zero, clamped to its range, NaN 0 (22.17.3, 22.17.4); to
                // a float to nearest even (22.17.1)
                let which = self.doubles_into_lanes(dest)?;
                let a = self.read_doubles(ins.operand(1)?, &which)?;
                let scalar = match name {
                    "dtof" => Scalar::F32,
                    "dtoi" => Scalar::I32,
                    _ => Scalar::U32,
                };
                let r = self.b.value(
                    Type::vector(scalar, lane_u8(which.len())),
                    Op::Convert(a, FloatToInt::Saturate),
                );
                self.write(dest, r, false)
            }
            "ftod" | "itod" | "utod" => {
                let which = self.doubles_written(dest)?;
                let scalar = match name {
                    "ftod" => Scalar::F32,
                    "itod" => Scalar::I32,
                    _ => Scalar::U32,
                };
                // the k-th lane of the source for the k-th double
                let a = self.read_lanes(ins.operand(1)?, &which, scalar)?;
                let r = self.b.value(
                    Type::vector(Scalar::F64, lane_u8(which.len())),
                    Op::Convert(a, FloatToInt::Saturate),
                );
                self.write_doubles(dest, r, false)
            }
            _ => self.refuse(format!("{name} is not a double-precision instruction")),
        }
    }
}
