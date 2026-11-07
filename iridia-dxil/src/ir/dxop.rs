//! The `dx.op` intrinsics that are not resource operations: signature reads and writes, math, bit operations, the
//! invocation ids, derivatives, discard and barriers. Each takes Direct3D's result as the naga-path lift does. The
//! stage operations of hull, domain and geometry programs are in [`super::stages`].

use iridia_ir::{
    Axis, Barrier, Binary, Block, Builtin, Cmp, Fineness, FloatToInt, Node, Op, Rounding, SaturateNan,
    Scalar, Scope, Target, Terminator, Ternary, Type, Unary, ValueId as IrValue, interface::SystemValue,
};

use super::{Lift, Lv};
use crate::Result;
use crate::lift::setup::const_int;
use dxil_reader::module::{Instruction, ValueId};

impl Lift<'_> {
    /// Argument `n` of a call.
    pub(crate) fn arg(&self, args: &[ValueId], n: usize) -> Result<ValueId> {
        args.get(n).copied().map_or_else(
            || {
                self.refuse(format!(
                    "an operation reads argument {n}, where the call has {}",
                    args.len()
                ))
            },
            Ok,
        )
    }

    /// The constant integer argument `n`.
    pub(crate) fn const_arg(&self, args: &[ValueId], n: usize) -> Option<i64> {
        args.get(n).and_then(|&a| const_int(self.m, self.f, a))
    }

    /// Define a call's result: a half rounded where it is made.
    pub(crate) fn define_result(&mut self, inst: &Instruction, lv: Lv) -> Result<()> {
        match inst.result {
            Some((v, ty)) => {
                let lv = self.quantized(lv, ty)?;
                self.define(v, lv)
            }
            None => Ok(()),
        }
    }

    pub(crate) fn call(&mut self, inst: &Instruction, function: ValueId, args: &[ValueId]) -> Result<()> {
        let Some(code) = self.opcode(function, args) else {
            let name = self.m.name(function).unwrap_or("a function");
            // the compiler's lifetime markers of locals say nothing the program does
            if name.starts_with("llvm.lifetime.") {
                return Ok(());
            }
            return self.refuse(format!("a call of {name}"));
        };
        if let Some(n) = crate::lift::dxop::arguments(code)
            && args.len() != n
        {
            return self.refuse(format!(
                "dx.op {code} called with {} arguments, where it takes {n}",
                args.len()
            ));
        }
        let f = Type::F32;
        let u = Type::U32;
        let value = |l: &mut Self, x: IrValue| l.define_result(inst, Lv::Value(x));
        match code {
            4 => self.load_input(inst, args),
            5 => self.store_output(args),
            // the one-operand float functions
            6 | 12..=29 => {
                let x = self.float(self.arg(args, 1)?)?;
                let op = match code {
                    6 => Unary::FAbs,
                    12 => Unary::FCos,
                    13 => Unary::FSin,
                    14 => Unary::FTan,
                    15 => Unary::FAcos,
                    16 => Unary::FAsin,
                    17 => Unary::FAtan,
                    18 => Unary::FCosh,
                    19 => Unary::FSinh,
                    20 => Unary::FTanh,
                    21 => Unary::FExp2,
                    22 => Unary::FFract,
                    23 => Unary::FLog2,
                    24 => Unary::FSqrt,
                    25 => Unary::FInverseSqrt,
                    26 => Unary::FRound(Rounding::NearestEven),
                    27 => Unary::FRound(Rounding::Floor),
                    28 => Unary::FRound(Rounding::Ceil),
                    _ => Unary::FRound(Rounding::Trunc),
                };
                let r = self.unary(f, op, x);
                value(self, r)
            }
            7 => {
                let x = self.float(self.arg(args, 1)?)?;
                let r = self.unary(f, Unary::Saturate(SaturateNan::Zero), x);
                value(self, r)
            }
            8 => {
                let x = self.float(self.arg(args, 1)?)?;
                let r = self.unary(Type::BOOL, Unary::IsNan, x);
                value(self, r)
            }
            9..=11 => {
                let x = self.float(self.arg(args, 1)?)?;
                let bits = self.bitcast(u, x);
                let mask = self.u32_lit(0x7fff_ffff);
                let m = self.binary(u, Binary::And, bits, mask);
                let infinity = self.u32_lit(0x7f80_0000);
                let r = match code {
                    9 => self.binary(Type::BOOL, Binary::UCmp(Cmp::Eq), m, infinity),
                    10 => self.binary(Type::BOOL, Binary::UCmp(Cmp::Lt), m, infinity),
                    _ => {
                        let smallest = self.u32_lit(0x0080_0000);
                        let big = self.binary(Type::BOOL, Binary::UCmp(Cmp::Ge), m, smallest);
                        let finite = self.binary(Type::BOOL, Binary::UCmp(Cmp::Lt), m, infinity);
                        self.binary(Type::BOOL, Binary::And, big, finite)
                    }
                };
                value(self, r)
            }
            31..=33
                if self
                    .value(self.arg(args, 1)?)
                    .is_ok_and(|x| self.g.ty(x) == Some(Type::scalar(Scalar::U64))) =>
            {
                let r = self.bits_wide(code, self.arg(args, 1)?)?;
                value(self, r)
            }
            30 | 31 => {
                let x = self.uint(self.arg(args, 1)?)?;
                let op = if code == 30 {
                    Unary::BitReverse
                } else {
                    Unary::CountOnes
                };
                let r = self.unary(u, op, x);
                value(self, r)
            }
            32..=34 => {
                let x = self.uint(self.arg(args, 1)?)?;
                let op = match code {
                    32 => Unary::FirstBitLow,
                    33 => Unary::FirstBitHigh,
                    _ => Unary::FirstBitHighSigned,
                };
                let r = self.unary(u, op, x);
                value(self, r)
            }
            35 | 36 => {
                let (a, b) = (self.float(self.arg(args, 1)?)?, self.float(self.arg(args, 2)?)?);
                let r = self.d3d_min_max(code == 35, a, b);
                value(self, r)
            }
            37..=40 => {
                let (a, b) = (self.uint(self.arg(args, 1)?)?, self.uint(self.arg(args, 2)?)?);
                let r = match code {
                    37 | 38 => {
                        let (sa, sb) = (self.bitcast(Type::I32, a), self.bitcast(Type::I32, b));
                        let op = if code == 37 { Binary::SMax } else { Binary::SMin };
                        let r = self.binary(Type::I32, op, sa, sb);
                        self.bitcast(u, r)
                    }
                    39 => self.binary(u, Binary::UMax, a, b),
                    _ => self.binary(u, Binary::UMin, a, b),
                };
                value(self, r)
            }
            // {high, low} of the 64-bit product
            41 | 42 => {
                let (a, b) = (self.uint(self.arg(args, 1)?)?, self.uint(self.arg(args, 2)?)?);
                let low = self.binary(u, Binary::IMul, a, b);
                let high = if code == 41 {
                    let (sa, sb) = (self.bitcast(Type::I32, a), self.bitcast(Type::I32, b));
                    let h = self.binary(Type::I32, Binary::SMulHigh, sa, sb);
                    self.bitcast(u, h)
                } else {
                    self.binary(u, Binary::UMulHigh, a, b)
                };
                self.define_result(inst, Lv::Agg(vec![Lv::Value(high), Lv::Value(low)]))
            }
            43 => {
                let (a, b) = (self.uint(self.arg(args, 1)?)?, self.uint(self.arg(args, 2)?)?);
                let q = self.binary(u, Binary::UDiv(iridia_ir::OnZero::AllOnes), a, b);
                let r = self.binary(u, Binary::URem(iridia_ir::OnZero::AllOnes), a, b);
                self.define_result(inst, Lv::Agg(vec![Lv::Value(q), Lv::Value(r)]))
            }
            44 | 45 => {
                let (a, b) = (self.uint(self.arg(args, 1)?)?, self.uint(self.arg(args, 2)?)?);
                let (op, flag) = if code == 44 {
                    (Binary::IAdd, Binary::UAddCarry)
                } else {
                    (Binary::ISub, Binary::USubBorrow)
                };
                let v = self.binary(u, op, a, b);
                let c = self.binary(u, flag, a, b);
                self.define_result(inst, Lv::Agg(vec![Lv::Value(v), Lv::Value(c)]))
            }
            // a separately rounded multiply and add, as the DXBC lift's mad
            46 => {
                let a = self.float(self.arg(args, 1)?)?;
                let b = self.float(self.arg(args, 2)?)?;
                let c = self.float(self.arg(args, 3)?)?;
                let p = self.binary(f, Binary::FMul(iridia_ir::MulZero::Ieee), a, b);
                let r = self.binary(f, Binary::FAdd, p, c);
                value(self, r)
            }
            47 => {
                let a = self.float(self.arg(args, 1)?)?;
                let b = self.float(self.arg(args, 2)?)?;
                let c = self.float(self.arg(args, 3)?)?;
                let r = self.g.value(f, Op::Ternary(Ternary::Fma, a, b, c));
                value(self, r)
            }
            48 | 49 => {
                let a = self.uint(self.arg(args, 1)?)?;
                let b = self.uint(self.arg(args, 2)?)?;
                let c = self.uint(self.arg(args, 3)?)?;
                let p = self.binary(u, Binary::IMul, a, b);
                let r = self.binary(u, Binary::IAdd, p, c);
                value(self, r)
            }
            54..=56 => {
                let n = usize::try_from(code.saturating_sub(52)).unwrap_or(2);
                let mut a = Vec::new();
                let mut b = Vec::new();
                for k in 0..n {
                    a.push(self.float(self.arg(args, k.saturating_add(1))?)?);
                    b.push(self.float(self.arg(args, n.saturating_add(k).saturating_add(1))?)?);
                }
                let lanes = u8::try_from(n).unwrap_or(2);
                let va = self.g.value(Type::vector(Scalar::F32, lanes), Op::Compose(a));
                let vb = self.g.value(Type::vector(Scalar::F32, lanes), Op::Compose(b));
                let r = self.g.value(f, Op::Dot(va, vb));
                value(self, r)
            }
            57 | 160 | 216..=218 => self.create_handle(inst, code, args),
            141..=159 | 161 | 213 => self.ray_op(inst, code, args),
            59 => self.cbuffer_load(inst, args),
            60..=65 | 73 | 74 => self.sample(inst, code, args),
            66 => self.texture_load(inst, args),
            67 => self.texture_store(args),
            68 | 139 => self.buffer_load(inst, args),
            69 | 140 => self.buffer_store(args),
            70 => self.buffer_counter(inst, args),
            72 => self.dimensions(inst, args),
            75 => self.sample_position(inst, args),
            // MakeDouble(lo, hi) and SplitDouble(d): a double and its two words, the low first
            101 => {
                let lo = self.uint(self.arg(args, 1)?)?;
                let hi = self.uint(self.arg(args, 2)?)?;
                let words = self
                    .g
                    .value(Type::vector(Scalar::U32, 2), Op::Compose(vec![lo, hi]));
                let r = self.bitcast(Type::scalar(Scalar::F64), words);
                value(self, r)
            }
            102 => {
                let d = self.value_as(self.arg(args, 1)?, Type::scalar(Scalar::F64))?;
                let words = self.bitcast(Type::vector(Scalar::U32, 2), d);
                let lo = self.g.value(Type::U32, Op::Swizzle(words, [0, 0, 0, 0]));
                let hi = self.g.value(Type::U32, Op::Swizzle(words, [1, 0, 0, 0]));
                self.define_result(inst, Lv::Agg(vec![Lv::Value(lo), Lv::Value(hi)]))
            }
            78 => self.buffer_atomic(inst, args),
            79 => self.buffer_compare_exchange(inst, args),
            80 => self.barrier(args),
            82 => self.discard(args),
            83..=86 => {
                if self.setup.interface.stage != iridia_ir::interface::Stage::Pixel {
                    // Shader Model 6.6 takes a compute program's derivatives over its quads of invocations
                    return self.refuse("a derivative outside a pixel program is not lifted onto the IR yet");
                }
                let x = self.float(self.arg(args, 1)?)?;
                let (axis, fineness) = match code {
                    83 => (Axis::X, Fineness::Coarse),
                    84 => (Axis::Y, Fineness::Coarse),
                    85 => (Axis::X, Fineness::Fine),
                    _ => (Axis::Y, Fineness::Fine),
                };
                let r = self.unary(f, Unary::Derivative(axis, fineness), x);
                value(self, r)
            }
            90 => {
                let r = self.system_input(SystemValue::SampleIndex, None)?;
                value(self, r)
            }
            // Coverage: the samples of the pixel the primitive covers, ANDed with the sample mask
            91 => {
                let r = self.system_input(SystemValue::Coverage, None)?;
                value(self, r)
            }
            93..=95 => {
                let system = match code {
                    93 => SystemValue::DispatchThreadId,
                    94 => SystemValue::GroupId,
                    _ => SystemValue::GroupThreadId,
                };
                let Some(component) = self
                    .const_arg(args, 1)
                    .and_then(|c| u8::try_from(c).ok())
                    .filter(|c| *c < 3)
                else {
                    return self.refuse("an invocation id read at a computed component");
                };
                let r = self.system_input(system, Some(component))?;
                value(self, r)
            }
            96 => {
                let r = self.system_input(SystemValue::GroupIndex, None)?;
                value(self, r)
            }
            97..=99 => self.stream_op(code, args),
            110..=123 | 135 | 136 => self.wave(inst, code, args),
            168..=173 => self.mesh_op(code, args, inst.result.map(|(v, _)| v)),
            100 => self.pipeline_value(inst, Builtin::GsInstance),
            103 => self.load_output_control_point(inst, args),
            104 => self.load_patch_constant(inst, args),
            105 => self.domain_location(inst, args),
            106 => self.store_patch_constant(args),
            107 => self.pipeline_value(inst, Builtin::OutputControlPoint),
            108 => self.pipeline_value(inst, Builtin::PrimitiveId),
            // LegacyF32ToF16 and LegacyF16ToF32: the half in the low 16 bits of an integer
            130 => {
                let x = self.float(self.arg(args, 1)?)?;
                let r = self.unary(u, Unary::F32ToF16, x);
                value(self, r)
            }
            131 => {
                let x = self.uint(self.arg(args, 1)?)?;
                let r = self.unary(f, Unary::F16ToF32, x);
                value(self, r)
            }
            other => {
                let name = self.m.name(function).unwrap_or("dx.op");
                self.refuse(format!("{name}: dx.op {other} is not lifted onto the IR yet"))
            }
        }
    }

    /// The input varying a system value is, and its lane `component` (all its lanes for `None`), as `u32`s.
    pub(crate) fn system_input(&mut self, system: SystemValue, component: Option<u8>) -> Result<IrValue> {
        let Some(slot) = self
            .setup
            .interface
            .inputs
            .iter()
            .position(|v| v.system == Some(system))
        else {
            return self.refuse(format!("{system:?} has no input"));
        };
        let varying = self
            .setup
            .interface
            .inputs
            .get(slot)
            .cloned()
            .ok_or_else(|| crate::error(self.at, "an input past the interface"))?;
        let t = iridia_ir::value_type(varying.value);
        let (mask, ty) = match component {
            Some(c) if c < t.lanes => (1u8 << c, Type::scalar(t.scalar)),
            Some(c) => return self.refuse(format!("lane {c} of {system:?}, which has {}", t.lanes)),
            None => (0xfu8 >> 4u8.saturating_sub(t.lanes), t),
        };
        let slot = u32::try_from(slot).map_err(|_| crate::error(self.at, "an input slot past 2^32"))?;
        let x = self.g.value(ty, Op::LoadInput(slot, mask));
        if ty.scalar == Scalar::U32 {
            Ok(x)
        } else {
            self.retype(x, ty.with(Scalar::U32))
        }
    }

    /// `LoadInput(element, row, column, vertex)`: one lane of an input element, as the call's type; of the point
    /// `vertex` names in a program whose inputs are arrays of points.
    fn load_input(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let Some(id) = self.const_arg(args, 1) else {
            return self.refuse("an input read at a computed element");
        };
        let Some(element) = self
            .setup
            .input_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("input element {id}"));
        };
        let (Some(row), Some(col)) = (self.const_arg(args, 2), self.const_arg(args, 3)) else {
            return self.refuse("an input read at a computed row or column");
        };
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        let Some((varying, position)) = self.setup.input_cell(register, lane) else {
            return self.refuse(format!(
                "input {} has no lane at register {register}",
                element.name
            ));
        };
        let v = self
            .setup
            .interface
            .inputs
            .get(varying)
            .cloned()
            .ok_or_else(|| crate::error(self.at, "an input past the interface"))?;
        let scalar = iridia_ir::value_type(v.value).scalar;
        let slot = u32::try_from(varying).map_err(|_| crate::error(self.at, "an input slot past 2^32"))?;
        let op = if self.reads_points() {
            let point = self.uint(self.arg(args, 4)?)?;
            Op::LoadInputAt(slot, 1 << position, point)
        } else {
            Op::LoadInput(slot, 1 << position)
        };
        let x = self.g.value(Type::scalar(scalar), op);
        self.varying_result(inst, x, scalar)
    }

    /// The register and lane an element's row and column name.
    pub(crate) fn cell_of(
        &self,
        first_row: u32,
        first_column: u8,
        row: i64,
        column: i64,
    ) -> Result<(u32, u8)> {
        let register = u32::try_from(row).ok().and_then(|row| first_row.checked_add(row));
        let lane = u8::try_from(column)
            .ok()
            .and_then(|c| first_column.checked_add(c))
            .filter(|l| *l < 4);
        match (register, lane) {
            (Some(register), Some(lane)) => Ok((register, lane)),
            _ => self.refuse(format!(
                "row {row}, column {column} of an element that starts at row {first_row}, column {first_column}"
            )),
        }
    }

    /// `StoreOutput(element, row, column, value)`: one lane of an output varying.
    fn store_output(&mut self, args: &[ValueId]) -> Result<()> {
        let (slot, position, x) = self.output_lane(args)?;
        self.g.node(Node::StoreOutput(slot, 1 << position, x));
        Ok(())
    }

    /// The output lane `(element, row, column, value)` of a store (arguments 1 to 4) names: the output's slot, the
    /// lane's position in it, and the value as the output's scalar.
    pub(crate) fn output_lane(&mut self, args: &[ValueId]) -> Result<(u32, u8, IrValue)> {
        let Some(id) = self.const_arg(args, 1) else {
            return self.refuse("an output written at a computed element");
        };
        let Some(element) = self
            .setup
            .output_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("output element {id}"));
        };
        let (slot, position) = if element.placed {
            let (Some(row), Some(col)) = (self.const_arg(args, 2), self.const_arg(args, 3)) else {
                return self.refuse("an output written at a computed row or column");
            };
            let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
            match self.setup.output_cell(register, lane) {
                Some(cell) => cell,
                None => {
                    return self.refuse(format!(
                        "output {} has no lane at register {register}",
                        element.name
                    ));
                }
            }
        } else {
            // a depth or coverage output, which the packer does not place
            let system = match element.system_value {
                17 => SystemValue::Depth,
                18 => SystemValue::DepthLessEqual,
                19 => SystemValue::DepthGreaterEqual,
                14 => SystemValue::Coverage,
                _ => return self.refuse(format!("the unplaced output {}", element.name)),
            };
            let Some(slot) = self.setup.interface.outputs.iter().position(|v| {
                v.system == Some(system)
                    || (matches!(
                        system,
                        SystemValue::DepthLessEqual | SystemValue::DepthGreaterEqual
                    ) && v.system == Some(SystemValue::Depth))
            }) else {
                return self.refuse(format!("the output {} has no varying", element.name));
            };
            (slot, 0)
        };
        let v = self
            .setup
            .interface
            .outputs
            .get(slot)
            .cloned()
            .ok_or_else(|| crate::error(self.at, "an output past the interface"))?;
        let t = Type::scalar(iridia_ir::value_type(v.value).scalar);
        let x = self.value(self.arg(args, 4)?)?;
        let x = self.flag_bits(x);
        let x = self.retype(x, t)?;
        let slot = u32::try_from(slot).map_err(|_| crate::error(self.at, "an output slot past 2^32"))?;
        Ok((slot, position, x))
    }

    /// `Barrier(mode)`: 1 synchronise the thread group, 2 fence buffer memory across groups, 4 fence it within the
    /// group, 8 fence thread-group memory. A synchronising barrier orders the memory its group shares too, as the
    /// naga-path lift's does.
    fn barrier(&mut self, args: &[ValueId]) -> Result<()> {
        let Some(mode) = self.const_arg(args, 1) else {
            return self.refuse("a barrier of a computed mode");
        };
        let execution = mode & 1 != 0;
        let b = Barrier {
            execution,
            workgroup: execution || mode & 0b1000 != 0,
            storage: if mode & 0b0010 != 0 {
                Some(Scope::Device)
            } else if execution || mode & 0b0100 != 0 {
                Some(Scope::Workgroup)
            } else {
                None
            },
        };
        self.g.node(Node::Barrier(b));
        Ok(())
    }

    /// `Discard(condition)`: the invocation is demoted where the condition holds, and goes on.
    fn discard(&mut self, args: &[ValueId]) -> Result<()> {
        let cond = self.value_as(self.arg(args, 1)?, Type::BOOL)?;
        // the block splits: where the condition holds, a block that demotes, then on to the rest
        let kill = self.g.blocks.len();
        let rest = kill.saturating_add(1);
        let to = |b: usize| -> Result<Target> {
            Ok(Target {
                block: u32::try_from(b).map_err(|_| crate::error(self.at, "a block number past 2^32"))?,
                args: Vec::new(),
            })
        };
        self.g.block().term = Terminator::Branch {
            cond,
            accept: to(kill)?,
            reject: to(rest)?,
        };
        self.g.blocks.push(Block {
            params: Vec::new(),
            nodes: vec![Node::Demote],
            term: Terminator::Jump(to(rest)?),
        });
        self.g.blocks.push(Block {
            params: Vec::new(),
            nodes: Vec::new(),
            term: Terminator::Unreachable,
        });
        self.g.current = rest;
        Ok(())
    }

    /// `CountBits` (31), `FirstbitLo` (32) and `FirstbitHi` (33) of a 64-bit integer, from its two words as the
    /// naga-path lift takes them: the counts of both words; the lowest set bit's index or the count of bits above the
    /// highest, from the word that has one (32 more for the second word looked at), all ones where neither does.
    fn bits_wide(&mut self, code: u64, a: ValueId) -> Result<IrValue> {
        let u = Type::U32;
        let x = self.value(a)?;
        let pair = self.bitcast(Type::vector(Scalar::U32, 2), x);
        let lo = self.g.value(u, Op::Swizzle(pair, [0, 0, 0, 0]));
        let hi = self.g.value(u, Op::Swizzle(pair, [1, 0, 0, 0]));
        if code == 31 {
            let (l, h) = (
                self.unary(u, Unary::CountOnes, lo),
                self.unary(u, Unary::CountOnes, hi),
            );
            return Ok(self.binary(u, Binary::IAdd, l, h));
        }
        let (first, second, op) = if code == 32 {
            (lo, hi, Unary::FirstBitLow)
        } else {
            (hi, lo, Unary::FirstBitHigh)
        };
        let in_first = self.unary(u, op, first);
        let in_second = self.unary(u, op, second);
        let thirty_two = self.u32_lit(32);
        let in_second = self.binary(u, Binary::IAdd, in_second, thirty_two);
        let (zero, none) = (self.u32_lit(0), self.u32_lit(u32::MAX));
        let first_empty = self.binary(Type::BOOL, Binary::UCmp(Cmp::Eq), first, zero);
        let second_empty = self.binary(Type::BOOL, Binary::UCmp(Cmp::Eq), second, zero);
        let rest = self.select(u, second_empty, none, in_second);
        Ok(self.select(u, first_empty, rest, in_first))
    }

    /// A float to integer conversion with Direct3D's rule (saturating, NaN to zero), as `u32` bits.
    #[allow(dead_code)]
    pub(crate) fn d3d_float_to_uint(&mut self, x: IrValue, signed: bool) -> IrValue {
        let t = if signed { Type::I32 } else { Type::U32 };
        let r = self.g.value(t, Op::Convert(x, FloatToInt::Saturate));
        if signed { self.bitcast(Type::U32, r) } else { r }
    }
}
