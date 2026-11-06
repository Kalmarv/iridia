//! Lifting a Shader Model 4 or 5 program to an [`iridia_ir::Program`].
//!
//! The model is the token stream's own: each register (`r#`, `o#`, an indexable `x#`) is a variable of four raw
//! 32-bit lanes (`vec4<u32>`), each instruction reads its operands' lanes as the type it implies, computes, and stores
//! its result's bits back into the lanes its write mask names. `iridia_passes::promote` turns the variables into
//! values afterwards. The interface is [`crate::lift`]'s (the same signature, resource and stage reading), and so is
//! Direct3D's behaviour where it differs from other APIs, here in the IR's own operations: `min` and `max` take the
//! number over a NaN, `saturate` gives 0 for NaN, `ftoi` and `ftou` saturate, `udiv` by zero gives all ones, shift
//! and bit-field counts take their low five bits, and a buffer read outside the buffer gives zero.

mod body;
mod operands;
mod resources;

use std::collections::{BTreeMap, BTreeSet};

use iridia_ir::{
    Builder, Conventions, Denormals, End, FloatEnv, LabelId, Op, Program, Provenance, Scalar, Type, ValueId, VarId,
    interface::{InstructionCount, InstructionUnit, ShaderFormat, Stage, SystemValue},
};

use crate::lift::setup::Setup;
use crate::{Result, error};
use dxbc_reader::decode::{Declared, Decoded, Register, decode};

/// The lifter's name in a program's provenance.
pub const LIFTER: &str = "iridia-dxbc-ir";

/// Increment when the same bytecode lifts differently.
pub const LIFTER_VERSION: u32 = 1;

/// Lift a DXBC container's program onto the IR. What the lift does not carry is an `Err` naming the word and why.
pub fn lift(bytes: &[u8]) -> Result<Program> {
    lift_with(bytes, &iridia::BoundSizes::new())
}

/// [`lift`] with the constant buffers `bound` names bound at the size it states (as [`crate::lift_with`]).
pub fn lift_with(bytes: &[u8], bound: &iridia::BoundSizes) -> Result<Program> {
    let directory = crate::parts(bytes)?;
    let code = directory
        .iter()
        .find(|c| matches!(&c.kind, b"SHDR" | b"SHEX"))
        .ok_or_else(|| error(0, "missing shader code chunk"))?;
    if code.data.len() % 4 != 0 {
        return Err(error(0, "unaligned code length"));
    }
    let words: Vec<u32> = code
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    if words.len() > 1_048_576 {
        return Err(error(1, "program exceeds word budget"));
    }
    let decoded = decode(&words).map_err(crate::read)?;
    let stage = crate::reflection::stage_of(decoded.kind);
    let mut setup = Setup::new(bytes, &decoded, stage)?;
    setup.bind_sizes(bound)?;
    if setup.sigs.is_some() {
        return Err(error(
            0,
            format!("a {stage:?} program made of functions is not lifted to the IR yet"),
        ));
    }
    if setup.sm51 {
        return Err(error(0, "shader model 5.1 resource ranges are not lifted to the IR yet"));
    }
    let mut lift = Lift::new(&decoded, stage, setup);
    lift.declare()?;
    lift.prologue()?;
    lift.body()?;
    lift.finish(decoded.target(), bytes)
}

/// An open construct of the token stream's flow.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Frame {
    /// `if`, and whether its `else` was met.
    If { otherwise: bool },
    Loop(LabelId),
    Switch(SwitchState),
}

/// Where an open `switch` is in its cases.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SwitchState {
    label: LabelId,
    /// The case labels met since the last region, waiting for the code they choose: the values, and whether the
    /// default is among them.
    labels: Option<(Vec<u32>, bool)>,
    /// A case region is being filled.
    open: bool,
    /// How the last region ended, for the builder when the next opens.
    before: Option<End>,
}

/// The state of one lift.
struct Lift<'a> {
    b: Builder,
    decoded: &'a Decoded,
    stage: Stage,
    setup: Setup,
    /// `r#`.
    temps: Vec<VarId>,
    /// `x#[n]`: the array and its length.
    indexable: BTreeMap<u32, (VarId, u32)>,
    /// `v#`, as four lanes composed from the inputs the register holds (zero where none).
    inputs: BTreeMap<u32, ValueId>,
    /// `o#`.
    outputs: BTreeMap<u32, VarId>,
    /// `vPrim`, `vCoverage`, the compute ids: four lanes, the value in the first (in the first three for an id).
    specials: BTreeMap<SystemValue, ValueId>,
    /// `oDepth` and `oMask`.
    depth: Option<VarId>,
    coverage: Option<VarId>,
    resources: resources::Resources,
    frames: Vec<Frame>,
    /// The region being filled ended (an unconditional `break`, `continue` or `ret`): how, and the depth of the
    /// constructs opened since in the code that is not reached.
    ended: Option<(End, u32)>,
    notes: BTreeSet<String>,
    /// The word of the instruction being lifted, for refusals.
    word: usize,
}

/// The mask of the lowest `width` lanes.
fn every(width: u8) -> u8 {
    u8::try_from((1u32 << width.min(4)).saturating_sub(1)).unwrap_or(15)
}

pub(super) fn lanes_of(mask: u8) -> Vec<u8> {
    (0..4).filter(|i| mask >> i & 1 != 0).collect()
}

/// The IR scalar of an interface scalar type.
fn scalar_of(s: iridia::ScalarType) -> Scalar {
    match s {
        iridia::ScalarType::Float => Scalar::F32,
        iridia::ScalarType::Uint => Scalar::U32,
        iridia::ScalarType::Sint => Scalar::I32,
        iridia::ScalarType::Bool => Scalar::Bool,
        iridia::ScalarType::Double => Scalar::F64,
    }
}

impl<'a> Lift<'a> {
    fn new(decoded: &'a Decoded, stage: Stage, setup: Setup) -> Self {
        Lift {
            b: Builder::new("main"),
            decoded,
            stage,
            notes: setup.notes.clone(),
            setup,
            temps: Vec::new(),
            indexable: BTreeMap::new(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            specials: BTreeMap::new(),
            depth: None,
            coverage: None,
            resources: resources::Resources::default(),
            frames: Vec::new(),
            ended: None,
            word: 0,
        }
    }

    fn refuse<T>(&self, detail: impl Into<String>) -> Result<T> {
        Err(error(self.word, detail))
    }

    fn op(&mut self, op: Op) -> Result<ValueId> {
        self.b.op(op).map_err(|e| error(self.word, e))
    }

    fn ty(&self, v: ValueId) -> Result<Type> {
        self.b
            .ty(v)
            .ok_or_else(|| error(self.word, format!("value %{} has no type", v.0)))
    }

    /// A register variable of four raw lanes, zero at the start (an unwritten register reads as zero).
    fn register(&mut self, name: &str) -> VarId {
        let v = self.b.var(name, Type::U32X4);
        let zero = self.b.literal(Type::U32X4, [0; 4]);
        self.b.store_var(v, zero);
        v
    }

    /// The variables of the registers the program declares, and the resources' declarations.
    fn declare(&mut self) -> Result<()> {
        for n in 0..self.setup.temps {
            let v = self.register(&format!("r{n}"));
            self.temps.push(v);
        }
        for (register, size) in self.setup.indexable.clone() {
            if size == 0 {
                continue;
            }
            let v = self.b.array(
                &format!("x{register}"),
                Type::U32X4,
                size,
                iridia_ir::Space::Private,
                vec![[0; 4]; size as usize],
            );
            self.indexable.insert(register, (v, size));
        }
        let decoded = self.decoded;
        for d in &decoded.declarations {
            if let Declared::Output { operand, .. } = &d.kind {
                match operand.register {
                    Register::Output => {
                        let register = operand.indices.first().map_or(0, |i| i.base);
                        if !self.outputs.contains_key(&register) {
                            let v = self.register(&format!("o{register}"));
                            self.outputs.insert(register, v);
                        }
                    }
                    Register::OutputDepth
                    | Register::OutputDepthGreaterEqual
                    | Register::OutputDepthLessEqual
                        if self.depth.is_none() =>
                    {
                        self.depth = Some(self.register("oDepth"));
                    }
                    Register::OutputCoverage if self.coverage.is_none() => {
                        self.coverage = Some(self.register("oMask"));
                    }
                    _ => {}
                }
            }
        }
        for (output, first, count) in self.setup.index_ranges.clone() {
            for r in first..first.saturating_add(count) {
                if output && !self.outputs.contains_key(&r) {
                    let v = self.register(&format!("o{r}"));
                    self.outputs.insert(r, v);
                }
            }
        }
        let placed: Vec<u32> = self
            .setup
            .interface
            .outputs
            .iter()
            .filter_map(|v| v.provenance.d3d.map(|d| d.register))
            .filter(|&r| r != u32::MAX)
            .collect();
        for r in placed {
            if let std::collections::btree_map::Entry::Vacant(e) = self.outputs.entry(r) {
                let v = self.b.var(&format!("o{r}"), Type::U32X4);
                let zero = self.b.literal(Type::U32X4, [0; 4]);
                self.b.store_var(v, zero);
                e.insert(v);
            }
        }
        self.declare_resources()
    }

    /// The input registers, each composed from the input varyings it holds, and the special inputs.
    fn prologue(&mut self) -> Result<()> {
        let zero = self.b.u32(0);
        let mut lanes: BTreeMap<u32, [ValueId; 4]> = BTreeMap::new();
        let inputs = self.setup.interface.inputs.clone();
        for (slot, v) in (0u32..).zip(&inputs) {
            let width = v.value.width;
            let carried = matches!(
                v.system,
                Some(
                    SystemValue::PrimitiveIndex
                        | SystemValue::Coverage
                        | SystemValue::DispatchThreadId
                        | SystemValue::GroupId
                        | SystemValue::GroupThreadId
                        | SystemValue::GroupIndex
                )
            );
            let declared = v.provenance.d3d.and_then(|d3d| {
                let declared = self.setup.declared_inputs.get(&d3d.register).copied().unwrap_or(0);
                (declared & d3d.mask != 0).then_some(d3d)
            });
            if declared.is_none() && !carried {
                continue;
            }
            let ty = Type::vector(scalar_of(v.value.scalar), width);
            let value = self.b.value(ty, Op::LoadInput(slot, every(width)));
            // raw bits: a bool is D3D's all ones or zero
            let bits = if ty.scalar == Scalar::Bool {
                let ones = self.b.literal(Type::vector(Scalar::U32, width), [u64::from(u32::MAX); 4]);
                let none = self.b.literal(Type::vector(Scalar::U32, width), [0; 4]);
                self.op(Op::Select {
                    cond: value,
                    accept: ones,
                    reject: none,
                })?
            } else {
                self.b
                    .value(Type::vector(Scalar::U32, width), Op::Bitcast(value))
            };
            let lane = |l: &mut Self, k: u8| -> Result<ValueId> {
                if width == 1 {
                    Ok(bits)
                } else {
                    Ok(l.b.value(Type::U32, Op::Swizzle(bits, [k, 0, 0, 0])))
                }
            };
            if let Some(d3d) = declared {
                let row = lanes.entry(d3d.register).or_insert([zero; 4]);
                let mut row_now = *row;
                for (k, l) in (0u8..).zip(lanes_of(d3d.mask)) {
                    if k >= width {
                        break;
                    }
                    if let Some(at) = row_now.get_mut(usize::from(l)) {
                        *at = lane(self, k)?;
                    }
                }
                lanes.insert(d3d.register, row_now);
            }
            if let Some(system) = v.system.filter(|_| carried) {
                let mut parts = [zero; 4];
                for (k, p) in (0u8..).zip(parts.iter_mut()).take(usize::from(width)) {
                    *p = lane(self, k)?;
                }
                let four = self.op(Op::Compose(parts.to_vec()))?;
                self.specials.insert(system, four);
            }
        }
        for (register, row) in lanes {
            let four = self.op(Op::Compose(row.to_vec()))?;
            self.inputs.insert(register, four);
        }
        Ok(())
    }

    /// Store the output registers into the outputs, as the function returns.
    fn epilogue(&mut self) -> Result<()> {
        let outputs = self.setup.interface.outputs.clone();
        for (slot, v) in (0u32..).zip(&outputs) {
            let (var, lanes) = match v.system {
                Some(SystemValue::Depth | SystemValue::DepthGreaterEqual | SystemValue::DepthLessEqual) => {
                    match self.depth {
                        Some(d) => (d, vec![0]),
                        None => return self.refuse("the depth output is not declared"),
                    }
                }
                Some(SystemValue::Coverage) => match self.coverage {
                    Some(c) => (c, vec![0]),
                    None => return self.refuse("the coverage output is not declared"),
                },
                _ => {
                    let Some(d3d) = v.provenance.d3d else {
                        return self.refuse(format!("output {} has no register", v.label.name));
                    };
                    match self.outputs.get(&d3d.register) {
                        Some(&o) => (o, lanes_of(d3d.mask)),
                        None => return self.refuse(format!("output register o{} is not declared", d3d.register)),
                    }
                }
            };
            let width = v.value.width;
            let row = self.b.value(Type::U32X4, Op::LoadVar(var));
            let mut pattern = [0u8; 4];
            for (p, l) in pattern.iter_mut().zip(&lanes) {
                *p = *l;
            }
            let bits = self.b.value(Type::vector(Scalar::U32, width), Op::Swizzle(row, pattern));
            let scalar = scalar_of(v.value.scalar);
            let value = match scalar {
                Scalar::U32 => bits,
                Scalar::Bool => {
                    let zero = self.b.literal(Type::vector(Scalar::U32, width), [0; 4]);
                    self.op(Op::Binary(iridia_ir::Binary::UCmp(iridia_ir::Cmp::Ne), bits, zero))?
                }
                s => self.b.value(Type::vector(s, width), Op::Bitcast(bits)),
            };
            self.b.store_output(slot, every(width), value);
        }
        Ok(())
    }

    fn finish(mut self, target: String, bytes: &[u8]) -> Result<Program> {
        if let Some(frame) = self.frames.last() {
            return self.refuse(format!("{frame:?} is not closed at the end of the program"));
        }
        let end = match self.ended.take() {
            Some((end, _)) => end,
            None => {
                self.epilogue()?;
                End::Return
            }
        };
        let entry = self.b.finish(end).map_err(|e| error(self.word, e))?;
        let mut policies = vec!["dxbc.mul=ieee".to_owned()];
        policies.sort();
        let workgroup = match self.stage {
            Stage::Compute => Some(
                self.setup
                    .workgroup
                    .ok_or_else(|| error(0, "a compute program that declares no thread group size"))?,
            ),
            _ => None,
        };
        let program = Program {
            stage: self.stage,
            entry,
            patch: None,
            workgroup,
            interface: self.setup.interface,
            conventions: Conventions::D3D,
            // D3D10 and later: f32 denormals flushed (sign kept) on each operation's input and output, f64's kept
            float: FloatEnv {
                f16: Denormals::Any,
                f32: Denormals::Flush,
                f64: Denormals::Preserve,
            },
            provenance: Provenance {
                format: ShaderFormat::Dxbc,
                target,
                lifter: LIFTER.into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(iridia_ir::source_hash(bytes)),
                instruction_count: Some(InstructionCount {
                    count: self.decoded.instructions.len() as u64,
                    unit: InstructionUnit::DxbcInstructions,
                }),
                policies,
            },
            notes: self.notes.into_iter().collect(),
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
