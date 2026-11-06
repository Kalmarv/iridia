//! The instructions: each DXBC instruction as IR operations, and the flow of the token stream as the IR's constructs.

use iridia_ir::{
    Binary, Cmp, End, FloatToInt, MinMaxNan, MulZero, OnZero, Op, Rounding, Scalar, Ternary, Unary, ValueId,
};

use super::operands::OperandAt;
use super::{Frame, Lift, SwitchState};
use crate::Result;
use dxbc_reader::decode::{Instruction, Operand, Register};

/// The opcodes that open a construct, close one, or start another region of the open one.
fn opens(name: &str) -> bool {
    matches!(name, "if" | "loop" | "switch")
}
fn closes(name: &str) -> bool {
    matches!(name, "endif" | "endloop" | "endswitch")
}
fn divides(name: &str) -> bool {
    matches!(name, "else" | "case" | "default")
}

impl Lift<'_> {
    pub(super) fn body(&mut self) -> Result<()> {
        let decoded = self.decoded;
        for ins in &decoded.instructions {
            self.word = ins.word;
            let name = ins.name();
            // code after an unconditional exit is not reached: skipped to the end of its region
            if let Some((_, depth)) = &mut self.ended {
                if opens(name) {
                    *depth = depth.saturating_add(1);
                    continue;
                }
                if *depth > 0 {
                    if closes(name) {
                        *depth = depth.saturating_sub(1);
                    }
                    continue;
                }
                if !(closes(name) || divides(name)) {
                    continue;
                }
                if self.frames.is_empty() {
                    // past a `ret` at the top level: the rest of the program is not reached
                    break;
                }
            }
            self.b.at(Some(iridia_ir::Origin {
                unit: iridia_ir::Unit::Word,
                offset: u32::try_from(ins.word).unwrap_or(u32::MAX),
            }));
            self.instruction(ins, name).map_err(|mut e| {
                if !e.detail.starts_with(name) {
                    e.detail = format!("{name}: {}", e.detail);
                }
                e
            })?;
        }
        self.b.at(None);
        Ok(())
    }

    /// How the region being filled ends where it falls off its end (`yield`), or how an exit ended it.
    fn region_end(&mut self, fallthrough: End) -> End {
        match self.ended.take() {
            Some((end, _)) => end,
            None => fallthrough,
        }
    }

    /// End the region being filled with `end`; the code after it, to the end of the region, is not reached.
    fn exit(&mut self, end: End) {
        self.ended = Some((end, 0));
    }

    /// The innermost loop's label.
    fn innermost_loop(&self) -> Result<iridia_ir::LabelId> {
        self.frames
            .iter()
            .rev()
            .find_map(|f| match f {
                Frame::Loop(l) => Some(*l),
                _ => None,
            })
            .map_or_else(|| self.refuse("outside a loop"), Ok)
    }
    /// The innermost loop's or switch's label: what `break` leaves.
    fn innermost_breakable(&self) -> Result<iridia_ir::LabelId> {
        self.frames
            .iter()
            .rev()
            .find_map(|f| match f {
                Frame::Loop(l) => Some(*l),
                Frame::Switch(s) => Some(s.label),
                Frame::If { .. } => None,
            })
            .map_or_else(|| self.refuse("outside a loop or switch"), Ok)
    }

    fn close(&mut self, end: End) -> Result<()> {
        self.b
            .close(end, &[])
            .map(|_| ())
            .map_err(|e| crate::error(self.word, e))
    }

    /// Under `cond`, the region `end` ends; the code after goes on where it does not hold.
    fn exit_if(&mut self, cond: ValueId, end: End) -> Result<()> {
        self.b.open_if(cond);
        self.close(end)
    }

    /// Store the outputs and return, under `cond` where it is given.
    fn ret(&mut self, cond: Option<ValueId>) -> Result<()> {
        match cond {
            Some(c) => {
                self.b.open_if(c);
                self.epilogue()?;
                self.close(End::Return)
            }
            None => {
                self.epilogue()?;
                self.exit(End::Return);
                Ok(())
            }
        }
    }

    fn float2(&mut self, ins: &Instruction, op: Binary) -> Result<()> {
        let mask = self.dest_mask(ins.operand(0)?);
        let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
        let b = self.read(ins.operand(2)?, mask, Scalar::F32)?;
        let r = self.binary(op, a, b)?;
        self.write(ins.operand(0)?, r, ins.saturate)
    }
    fn float1(&mut self, ins: &Instruction, op: Unary) -> Result<()> {
        let mask = self.dest_mask(ins.operand(0)?);
        let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
        let r = self.unary(op, a)?;
        self.write(ins.operand(0)?, r, ins.saturate)
    }
    fn int2(&mut self, ins: &Instruction, scalar: Scalar, op: Binary) -> Result<()> {
        let mask = self.dest_mask(ins.operand(0)?);
        let a = self.read(ins.operand(1)?, mask, scalar)?;
        let b = self.read(ins.operand(2)?, mask, scalar)?;
        let r = self.binary(op, a, b)?;
        self.write(ins.operand(0)?, r, false)
    }
    fn int1(&mut self, ins: &Instruction, scalar: Scalar, op: Unary) -> Result<()> {
        let mask = self.dest_mask(ins.operand(0)?);
        let a = self.read(ins.operand(1)?, mask, scalar)?;
        let r = self.unary(op, a)?;
        self.write(ins.operand(0)?, r, false)
    }
    /// A comparison whose result is D3D's mask: all ones where it holds.
    fn compare(&mut self, ins: &Instruction, scalar: Scalar, op: Binary) -> Result<()> {
        let mask = self.dest_mask(ins.operand(0)?);
        let a = self.read(ins.operand(1)?, mask, scalar)?;
        let b = self.read(ins.operand(2)?, mask, scalar)?;
        let c = self.binary(op, a, b)?;
        let r = self.mask_of(c)?;
        self.write(ins.operand(0)?, r, false)
    }
    /// A bool's lanes as D3D's mask: all ones or zero.
    fn mask_of(&mut self, c: ValueId) -> Result<ValueId> {
        let ones = self.like(c, Scalar::U32, u32::MAX)?;
        let zero = self.like(c, Scalar::U32, 0)?;
        self.select(c, ones, zero)
    }
    /// A flow condition: the first operand's x lane tested as zero or nonzero.
    fn condition(&mut self, ins: &Instruction) -> Result<ValueId> {
        let v = self.read(ins.operand(0)?, 1, Scalar::U32)?;
        let zero = self.b.u32(0);
        self.binary(
            Binary::UCmp(if ins.nonzero { Cmp::Ne } else { Cmp::Eq }),
            v,
            zero,
        )
    }
    /// A source read once, as all four lanes, before an instruction with two destinations writes either (a
    /// destination may alias a source); `lanes` then takes a destination's own.
    fn read_whole(&mut self, op: &Operand, scalar: Scalar) -> Result<ValueId> {
        self.read_lanes(op, &[0, 1, 2, 3], scalar)
    }
    fn lanes(&mut self, whole: ValueId, destination: &Operand) -> Result<ValueId> {
        let mask = self.dest_mask(destination);
        self.swizzle(whole, &super::lanes_of(mask))
    }
    /// Write each of two results to its destination, where it has one.
    fn write_two(&mut self, ins: &Instruction, first: ValueId, second: ValueId, saturate: bool) -> Result<()> {
        for (n, whole) in [(0, first), (1, second)] {
            let destination = ins.operand(n)?;
            if destination.register != Register::Null {
                let v = self.lanes(whole, destination)?;
                self.write(destination, v, saturate)?;
            }
        }
        Ok(())
    }

    fn instruction(&mut self, ins: &Instruction, name: &str) -> Result<()> {
        // the first code after a switch's case labels opens their region
        if !matches!(name, "case" | "default" | "endswitch") {
            self.open_case()?;
        }
        let mul = Binary::FMul(MulZero::Ieee);
        let min_num = MinMaxNan::MinNum;
        match name {
            "nop" => {}
            "mov" => {
                let mask = self.dest_mask(ins.operand(0)?);
                // modifiers and saturation on a mov apply to float data
                let scalar = if ins.operand(1)?.modifier != 0 || ins.saturate {
                    Scalar::F32
                } else {
                    Scalar::U32
                };
                let a = self.read(ins.operand(1)?, mask, scalar)?;
                self.write(ins.operand(0)?, a, ins.saturate)?;
            }
            "movc" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let c = self.read(ins.operand(1)?, mask, Scalar::U32)?;
                // a modifier on a movc data operand is a float negate or abs (the data is moved, not interpreted)
                let kind = |op: &Operand| if op.modifier != 0 { Scalar::F32 } else { Scalar::U32 };
                let (ka, kb) = (kind(ins.operand(2)?), kind(ins.operand(3)?));
                let a = self.read(ins.operand(2)?, mask, ka)?;
                let b = self.read(ins.operand(3)?, mask, kb)?;
                let (a, b) = (self.bitcast(a, Scalar::U32)?, self.bitcast(b, Scalar::U32)?);
                let zero = self.like(c, Scalar::U32, 0)?;
                let cond = self.binary(Binary::UCmp(Cmp::Ne), c, zero)?;
                let r = self.select(cond, a, b)?;
                if ins.saturate {
                    // movc_sat: the moved data is saturated as a float
                    let f = self.bitcast(r, Scalar::F32)?;
                    self.write(ins.operand(0)?, f, true)?;
                } else {
                    self.write(ins.operand(0)?, r, false)?;
                }
            }
            "add" => self.float2(ins, Binary::FAdd)?,
            "mul" => self.float2(ins, mul)?,
            "div" => self.float2(ins, Binary::FDiv)?,
            "min" => self.float2(ins, Binary::FMin(min_num))?,
            "max" => self.float2(ins, Binary::FMax(min_num))?,
            "mad" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
                let b = self.read(ins.operand(2)?, mask, Scalar::F32)?;
                let c = self.read(ins.operand(3)?, mask, Scalar::F32)?;
                let p = self.binary(mul, a, b)?;
                let r = self.binary(Binary::FAdd, p, c)?;
                self.write(ins.operand(0)?, r, ins.saturate)?;
            }
            "dp2" | "dp3" | "dp4" => {
                let n = name
                    .as_bytes()
                    .get(2)
                    .map_or(4, |digit| digit.saturating_sub(b'0'));
                let lanes: Vec<u8> = (0..n).collect();
                let a = self.read_lanes(ins.operand(1)?, &lanes, Scalar::F32)?;
                let b = self.read_lanes(ins.operand(2)?, &lanes, Scalar::F32)?;
                let r = self.op(Op::Dot(a, b))?;
                self.write(ins.operand(0)?, r, ins.saturate)?;
            }
            "rcp" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
                let one = self.like(a, Scalar::F32, 1f32.to_bits())?;
                let r = self.binary(Binary::FDiv, one, a)?;
                self.write(ins.operand(0)?, r, ins.saturate)?;
            }
            "rsq" => self.float1(ins, Unary::FInverseSqrt)?,
            "sqrt" => self.float1(ins, Unary::FSqrt)?,
            "exp" => self.float1(ins, Unary::FExp2)?,
            "log" => self.float1(ins, Unary::FLog2)?,
            "frc" => self.float1(ins, Unary::FFract)?,
            "round_ne" => self.float1(ins, Unary::FRound(Rounding::NearestEven))?,
            "round_ni" => self.float1(ins, Unary::FRound(Rounding::Floor))?,
            "round_pi" => self.float1(ins, Unary::FRound(Rounding::Ceil))?,
            "round_z" => self.float1(ins, Unary::FRound(Rounding::Trunc))?,
            "deriv_rtx" | "deriv_rty" | "deriv_rtx_coarse" | "deriv_rty_coarse" | "deriv_rtx_fine"
            | "deriv_rty_fine" => {
                let axis = if name.contains("rtx") {
                    iridia_ir::Axis::X
                } else {
                    iridia_ir::Axis::Y
                };
                let fineness = if name.ends_with("coarse") {
                    iridia_ir::Fineness::Coarse
                } else if name.ends_with("fine") {
                    iridia_ir::Fineness::Fine
                } else {
                    iridia_ir::Fineness::Any
                };
                self.float1(ins, Unary::Derivative(axis, fineness))?;
            }
            "sincos" => {
                let source = self.read_whole(ins.operand(2)?, Scalar::F32)?;
                let sin = self.unary(Unary::FSin, source)?;
                let cos = self.unary(Unary::FCos, source)?;
                self.write_two(ins, sin, cos, ins.saturate)?;
            }
            // swapc d0, d1, c, a, b: where c holds, d0 = b and d1 = a, else d0 = a and d1 = b
            "swapc" => {
                let c = self.read_whole(ins.operand(2)?, Scalar::U32)?;
                let a = self.read_whole(ins.operand(3)?, Scalar::U32)?;
                let b = self.read_whole(ins.operand(4)?, Scalar::U32)?;
                let zero = self.like(c, Scalar::U32, 0)?;
                let cond = self.binary(Binary::UCmp(Cmp::Ne), c, zero)?;
                let first = self.select(cond, b, a)?;
                let second = self.select(cond, a, b)?;
                self.write_two(ins, first, second, false)?;
            }
            "eq" => self.compare(ins, Scalar::F32, Binary::FCmp(Cmp::Eq))?,
            "ne" => self.compare(ins, Scalar::F32, Binary::FCmp(Cmp::Ne))?,
            "lt" => self.compare(ins, Scalar::F32, Binary::FCmp(Cmp::Lt))?,
            "ge" => self.compare(ins, Scalar::F32, Binary::FCmp(Cmp::Ge))?,
            "ieq" => self.compare(ins, Scalar::U32, Binary::UCmp(Cmp::Eq))?,
            "ine" => self.compare(ins, Scalar::U32, Binary::UCmp(Cmp::Ne))?,
            "ilt" => self.compare(ins, Scalar::I32, Binary::SCmp(Cmp::Lt))?,
            "ige" => self.compare(ins, Scalar::I32, Binary::SCmp(Cmp::Ge))?,
            "ult" => self.compare(ins, Scalar::U32, Binary::UCmp(Cmp::Lt))?,
            "uge" => self.compare(ins, Scalar::U32, Binary::UCmp(Cmp::Ge))?,
            "and" => self.int2(ins, Scalar::U32, Binary::And)?,
            "or" => self.int2(ins, Scalar::U32, Binary::Or)?,
            "xor" => self.int2(ins, Scalar::U32, Binary::Xor)?,
            "not" => self.int1(ins, Scalar::U32, Unary::Not)?,
            "iadd" => self.int2(ins, Scalar::U32, Binary::IAdd)?,
            "ineg" => self.int1(ins, Scalar::I32, Unary::INeg)?,
            "imad" | "umad" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::U32)?;
                let b = self.read(ins.operand(2)?, mask, Scalar::U32)?;
                let c = self.read(ins.operand(3)?, mask, Scalar::U32)?;
                let p = self.binary(Binary::IMul, a, b)?;
                let r = self.binary(Binary::IAdd, p, c)?;
                self.write(ins.operand(0)?, r, false)?;
            }
            "imul" | "umul" => {
                let scalar = if name == "imul" { Scalar::I32 } else { Scalar::U32 };
                let a = self.read_whole(ins.operand(2)?, scalar)?;
                let b = self.read_whole(ins.operand(3)?, scalar)?;
                let high = self.binary(
                    if name == "imul" {
                        Binary::SMulHigh
                    } else {
                        Binary::UMulHigh
                    },
                    a,
                    b,
                )?;
                let low = self.binary(Binary::IMul, a, b)?;
                self.write_two(ins, high, low, false)?;
            }
            "imin" => self.int2(ins, Scalar::I32, Binary::SMin)?,
            "imax" => self.int2(ins, Scalar::I32, Binary::SMax)?,
            "umin" => self.int2(ins, Scalar::U32, Binary::UMin)?,
            "umax" => self.int2(ins, Scalar::U32, Binary::UMax)?,
            // the IR's shifts take the count's low five bits, as D3D's do
            "ishl" => self.int2(ins, Scalar::U32, Binary::Shl)?,
            "ushr" => self.int2(ins, Scalar::U32, Binary::ShrU)?,
            "ishr" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::I32)?;
                let b = self.read(ins.operand(2)?, mask, Scalar::U32)?;
                let r = self.binary(Binary::ShrS, a, b)?;
                self.write(ins.operand(0)?, r, false)?;
            }
            "udiv" => {
                let a = self.read_whole(ins.operand(2)?, Scalar::U32)?;
                let b = self.read_whole(ins.operand(3)?, Scalar::U32)?;
                let q = self.binary(Binary::UDiv(OnZero::AllOnes), a, b)?;
                let r = self.binary(Binary::URem(OnZero::AllOnes), a, b)?;
                self.write_two(ins, q, r, false)?;
            }
            "itof" | "utof" => {
                let scalar = if name == "itof" { Scalar::I32 } else { Scalar::U32 };
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, scalar)?;
                let ty = self.ty(a)?;
                let r = self
                    .b
                    .value(ty.with(Scalar::F32), Op::Convert(a, FloatToInt::Saturate));
                self.write(ins.operand(0)?, r, ins.saturate)?;
            }
            "ftoi" | "ftou" => {
                let scalar = if name == "ftoi" { Scalar::I32 } else { Scalar::U32 };
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
                let ty = self.ty(a)?;
                let r = self
                    .b
                    .value(ty.with(scalar), Op::Convert(a, FloatToInt::Saturate));
                self.write(ins.operand(0)?, r, false)?;
            }
            "countbits" => self.int1(ins, Scalar::U32, Unary::CountOnes)?,
            "bfrev" => self.int1(ins, Scalar::U32, Unary::BitReverse)?,
            "firstbit_hi" => self.int1(ins, Scalar::U32, Unary::FirstBitHigh)?,
            "firstbit_shi" => self.int1(ins, Scalar::U32, Unary::FirstBitHighSigned)?,
            "firstbit_lo" => self.int1(ins, Scalar::U32, Unary::FirstBitLow)?,
            "ubfe" | "ibfe" => {
                let signed = name == "ibfe";
                let mask = self.dest_mask(ins.operand(0)?);
                let width = self.read(ins.operand(1)?, mask, Scalar::U32)?;
                let offset = self.read(ins.operand(2)?, mask, Scalar::U32)?;
                let source = self.read(
                    ins.operand(3)?,
                    mask,
                    if signed { Scalar::I32 } else { Scalar::U32 },
                )?;
                let r = self.op(Op::Ternary(Ternary::BitExtract { signed }, source, offset, width))?;
                self.write(ins.operand(0)?, r, false)?;
            }
            "bfi" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let count = self.read(ins.operand(1)?, mask, Scalar::U32)?;
                let offset = self.read(ins.operand(2)?, mask, Scalar::U32)?;
                let insert = self.read(ins.operand(3)?, mask, Scalar::U32)?;
                let base = self.read(ins.operand(4)?, mask, Scalar::U32)?;
                let r = self.op(Op::BitInsert {
                    base,
                    insert,
                    offset,
                    count,
                })?;
                self.write(ins.operand(0)?, r, false)?;
            }
            "f32tof16" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::F32)?;
                let ty = self.ty(a)?;
                let r = self.b.value(ty.with(Scalar::U32), Op::Unary(Unary::F32ToF16, a));
                self.write(ins.operand(0)?, r, false)?;
            }
            "f16tof32" => {
                let mask = self.dest_mask(ins.operand(0)?);
                let a = self.read(ins.operand(1)?, mask, Scalar::U32)?;
                let ty = self.ty(a)?;
                let r = self.b.value(ty.with(Scalar::F32), Op::Unary(Unary::F16ToF32, a));
                self.write(ins.operand(0)?, r, false)?;
            }
            "uaddc" | "usubb" => {
                let a = self.read_whole(ins.operand(2)?, Scalar::U32)?;
                let b = self.read_whole(ins.operand(3)?, Scalar::U32)?;
                let (op, flag) = if name == "usubb" {
                    (Binary::ISub, Binary::USubBorrow)
                } else {
                    (Binary::IAdd, Binary::UAddCarry)
                };
                let result = self.binary(op, a, b)?;
                let carry = self.binary(flag, a, b)?;
                self.write_two(ins, result, carry, false)?;
            }
            // ---- flow
            "if" => {
                let c = self.condition(ins)?;
                self.b.open_if(c);
                self.frames.push(Frame::If { otherwise: false });
            }
            "else" => {
                let end = self.region_end(End::Yield(vec![]));
                match self.frames.last_mut() {
                    Some(Frame::If { otherwise: o @ false }) => *o = true,
                    _ => return self.refuse("else without its if"),
                }
                self.b.else_(end).map_err(|e| crate::error(self.word, e))?;
            }
            "endif" => {
                let end = self.region_end(End::Yield(vec![]));
                if !matches!(self.frames.pop(), Some(Frame::If { .. })) {
                    return self.refuse("endif without its if");
                }
                self.close(end)?;
            }
            "loop" => {
                let (label, _) = self.b.open_loop(&[]).map_err(|e| crate::error(self.word, e))?;
                self.frames.push(Frame::Loop(label));
            }
            "endloop" => {
                let Some(Frame::Loop(label)) = self.frames.pop() else {
                    return self.refuse("endloop without its loop");
                };
                let end = self.region_end(End::Continue(label, vec![]));
                self.close(end)?;
            }
            "break" => {
                let label = self.innermost_breakable()?;
                self.exit(End::Br(label, vec![]));
            }
            "continue" => {
                let label = self.innermost_loop()?;
                self.exit(End::Continue(label, vec![]));
            }
            "breakc" => {
                let c = self.condition(ins)?;
                let label = self.innermost_breakable()?;
                self.exit_if(c, End::Br(label, vec![]))?;
            }
            "continuec" => {
                let c = self.condition(ins)?;
                let label = self.innermost_loop()?;
                self.exit_if(c, End::Continue(label, vec![]))?;
            }
            "switch" => {
                let selector = self.read(ins.operand(0)?, 1, Scalar::U32)?;
                let label = self.b.open_switch(selector);
                self.frames.push(Frame::Switch(SwitchState {
                    label,
                    labels: None,
                    open: false,
                    before: None,
                }));
            }
            "case" | "default" => {
                let value = if name == "default" {
                    None
                } else {
                    Some(ins.operand(0)?.immediate[0])
                };
                let ended = self.ended.take();
                let Some(Frame::Switch(state)) = self.frames.last_mut() else {
                    return self.refuse("case outside a switch");
                };
                if state.open {
                    // the region before ends here: by its exit, or it falls into this one
                    let Some((end, _)) = ended else {
                        return Err(crate::error(
                            self.word,
                            "a case that falls into the next one after code of its own is not lifted to the IR yet",
                        ));
                    };
                    state.open = false;
                    state.before = Some(end);
                }
                // labels with no code between them share the region that follows; a case that falls into the
                // default is the default's already
                let labels = state.labels.get_or_insert_with(|| (Vec::new(), false));
                match value {
                    Some(v) => labels.0.push(v),
                    None => labels.1 = true,
                }
            }
            "endswitch" => {
                self.open_case()?;
                let Some(Frame::Switch(state)) = self.frames.pop() else {
                    return self.refuse("endswitch without its switch");
                };
                // the last region falls off the end of the switch
                let end = self.region_end(End::Br(state.label, vec![]));
                self.close(end)?;
            }
            "ret" => self.ret(None)?,
            "retc" => {
                let c = self.condition(ins)?;
                self.ret(Some(c))?;
            }
            "discard" => {
                let c = self.condition(ins)?;
                self.b.open_if(c);
                self.b.demote();
                self.close(End::Yield(vec![]))?;
            }
            // ---- resources
            "sample" | "sample_l" | "sample_b" | "sample_d" | "sample_c" | "sample_c_lz" | "gather4"
            | "gather4_c" => self.sample(ins)?,
            "ld" | "ld_ms" => self.load(ins)?,
            "resinfo" => self.resource_info(ins)?,
            "sample_info" => self.sample_info(ins)?,
            "ld_structured" => self.load_structured(ins)?,
            "ld_raw" => self.load_raw(ins)?,
            _ => return self.refuse(format!("opcode {} is not lifted to the IR yet", ins.opcode)),
        }
        Ok(())
    }
}

impl Lift<'_> {
    /// Open the region of the case labels the innermost switch is waiting with, ending the region before as it ended.
    fn open_case(&mut self) -> Result<()> {
        let word = self.word;
        let Some(Frame::Switch(state)) = self.frames.last_mut() else {
            return Ok(());
        };
        let Some((values, default)) = state.labels.take() else {
            return Ok(());
        };
        // the end of the region before the first case is not one
        let end = state.before.take().unwrap_or(End::Unreachable);
        state.open = true;
        self.b
            .case(end, if default { None } else { Some(values) })
            .map_err(|e| crate::error(word, e))
    }
}
