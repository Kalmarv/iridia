//! Lifting a Shader Model 6 program (DXIL) to an [`iridia_ir::Program`].
//!
//! A DXIL function is LLVM's: values in static single assignment form over basic blocks. The IR takes it nearly as it
//! is: each basic block is a block of a `Cfg` body (`iridia_passes::structurize` makes regions of it), each phi a
//! parameter of its block that each predecessor's branch gives, and each instruction the IR's operations. LLVM's
//! integers are signless: every integer is carried as an unsigned IR value, made signed where an operation reads it so.
//! A half is carried as an `f32` rounded to a half where it is made, and a narrow integer (2 to 31 bits) as a `u32`
//! of its low bits, as the naga-path lift ([`crate::lift`]) carries them. Local arrays and thread-group memory are
//! variables of 32-bit words, addressed by the word a `getelementptr` reaches. `dx.op` calls are the intrinsics,
//! with Direct3D's behaviour stated in the IR's operations (`min` and `max` take the number over a NaN, `saturate`
//! gives 0 for NaN, a float-to-integer conversion saturates, a buffer read outside its buffer gives zero). The interface
//! is the naga-path lift's (the same signature, resource and stage reading).
//!
//! A hull program is two functions, as the IR states one: the entry, its control-point phase, and the patch-constant
//! function (`Program::patch`), each lifted from its own LLVM function. A domain or geometry program is its entry.

mod dxop;
mod mesh;
mod payload;
mod raytracing;
mod resources;
mod stages;
mod values;
mod waves;

use std::collections::{BTreeMap, BTreeSet};

use iridia_ir::{
    Block, Body, Conventions, Denormals, FloatEnv, Function, Node, Op, Origin, Precision, Program,
    Provenance, Res, Target, Terminator, Type, Unit, ValueDef, ValueId as IrValue, Var, VarId,
    interface::{InstructionCount, InstructionUnit, ShaderFormat, Stage},
};

use crate::lift::setup::{Setup, dx_opcode};
use crate::{Result, error, read};
use dxil_reader::module::{
    Function as LlvmFunction, Instruction, Module, Op as LlvmOp, Type as LlvmType, TypeId, ValueId,
};
use dxil_reader::reflection;

/// The lifter's name in a program's provenance.
pub const LIFTER: &str = "iridia-dxil-ir";

/// Increment when the same bytecode lifts differently.
pub const LIFTER_VERSION: u32 = 1;

/// Lift a DXIL container's entry function onto the IR: a vertex, pixel, compute, hull, domain, geometry, mesh,
/// amplification or ray tracing program (a library of one export). What the lift does not carry is an `Err` naming the
/// bit offset and why.
pub fn lift(container: &[u8]) -> Result<Program> {
    lift_with(container, &iridia::BoundSizes::new())
}

/// [`lift`] with the constant buffers `bound` names bound at the size it states (as [`crate::lift_with`]).
pub fn lift_with(container: &[u8], bound: &iridia::BoundSizes) -> Result<Program> {
    let chunks = dxil_reader::parts(container).map_err(read)?;
    let module = dxil_reader::module(&chunks, b"DXIL")
        .map_err(read)?
        .ok_or_else(|| error(0, "no DXIL part"))?;
    let part = dxil_reader::program(
        chunks
            .iter()
            .find(|c| &c.kind == b"DXIL")
            .ok_or_else(|| error(0, "no DXIL part"))?
            .data,
    )
    .map_err(read)?;
    let entry_value = reflection::entry(&module).map_err(read)?;
    let entry = module
        .functions
        .iter()
        .find(|f| f.value == entry_value)
        .ok_or_else(|| error(0, "the entry point has no body"))?;
    let patch_constant = crate::lift::patch_constant_function(&module, part.program_version)?;
    let mut functions = vec![entry];
    functions.extend(patch_constant);
    let mut setup = Setup::new(&chunks, &module, &functions)?;
    bound
        .apply(&mut setup.interface)
        .map_err(|why| error(0, format!("bound sizes: {why}")))?;
    let stage = setup.interface.stage;
    if !matches!(
        stage,
        Stage::Vertex
            | Stage::Pixel
            | Stage::Compute
            | Stage::Hull
            | Stage::Domain
            | Stage::Geometry
            | Stage::Mesh
            | Stage::Amplification
            | Stage::RayGeneration
            | Stage::Intersection
            | Stage::AnyHit
            | Stage::ClosestHit
            | Stage::Miss
            | Stage::Callable
    ) {
        return Err(error(
            0,
            format!("a {stage:?} program is not lifted onto the IR yet"),
        ));
    }
    let workgroup = if stage.has_thread_groups() {
        Some(
            reflection::num_threads(&module)
                .map_err(read)?
                .ok_or_else(|| error(0, format!("a {stage:?} program that states no thread group size")))?,
        )
    } else {
        None
    };
    let target = crate::lift::target_of(part.program_version);
    // the entry's 32-bit denormal mode (dxc's `-denorm`): kept where it says `preserve`; flushed, as Direct3D's
    // default, where it says `ftz` or `any` or states none
    let f32_denormals = if reflection::fp32_denorm_mode(&module).map_err(read)?.as_deref() == Some("preserve")
    {
        Denormals::Preserve
    } else {
        Denormals::Flush
    };
    let mut lift = Lift::new(&module, functions, setup)?;
    lift.declare_resources()?;
    let function = lift.function(entry, "main")?;
    let patch = patch_constant
        .map(|f| lift.function(f, "patch_constant"))
        .transpose()?;
    let mut notes: Vec<String> = lift.notes.into_iter().collect();
    notes.extend(lift.setup.notes.iter().cloned());
    notes.sort();
    notes.dedup();
    // the instructions of the program's functions
    let count = lift.functions.iter().map(|f| f.instructions.len() as u64).sum();
    let interface = lift.setup.interface;
    Ok(Program {
        stage,
        entry: function,
        patch,
        workgroup,
        interface,
        conventions: Conventions::D3D,
        // Direct3D's: a 32-bit float's denormals are flushed unless the entry keeps them (DXIL's `fp32-denorm-mode`
        // of `any` lets a driver flush them, and the naga-path lift does), a double's kept
        float: FloatEnv {
            f16: Denormals::Any,
            f32: f32_denormals,
            f64: Denormals::Preserve,
        },
        provenance: Provenance {
            format: ShaderFormat::Dxil,
            target,
            lifter: LIFTER.into(),
            lifter_version: LIFTER_VERSION,
            source_hash: Some(iridia::source_hash(container)),
            instruction_count: Some(InstructionCount {
                unit: InstructionUnit::DxilInstructions,
                count,
            }),
            policies: Vec::new(),
        },
        notes,
    })
}

/// A translated LLVM value.
#[derive(Clone, Debug)]
pub(crate) enum Lv {
    /// An IR value.
    Value(IrValue),
    /// A structure (a `dx.op` result such as `ResRet`, a `cmpxchg` pair): its fields.
    Agg(Vec<Lv>),
    /// A resource handle: the interface resource it names, and for an element of an array of resources the element
    /// (from the range's first register) the program picked.
    Handle(Res),
    /// A pointer into a variable: the variable and where in it the pointer reaches.
    Pointer(Pointer),
    /// A pointer into dxc's no-op marker globals (`dx.nothing`): a load of it is zero and a store does nothing.
    Nothing,
}

/// Where a pointer reaches in a variable: a fixed offset plus, where the pointer was computed, a `u32` value; in words,
/// or in bytes in a variable a task payload lives in (see [`payload`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pointer {
    pub var: VarId,
    pub at: u32,
    pub index: Option<IrValue>,
}

/// The IR function being built: its values, variables and `Cfg` blocks.
struct Graph {
    values: Vec<ValueDef>,
    vars: Vec<Var>,
    blocks: Vec<Block>,
    /// The block instructions are placed in.
    current: usize,
    /// The source offset values are made at.
    origin: Option<Origin>,
}

impl Graph {
    fn def(&mut self, ty: Type, op: Op) -> IrValue {
        let id = IrValue(u32::try_from(self.values.len()).unwrap_or(u32::MAX));
        self.values.push(ValueDef {
            ty,
            op,
            origin: self.origin,
            precision: Precision::Full,
        });
        id
    }
    fn block(&mut self) -> &mut Block {
        // the blocks are made before any instruction is placed, so `current` is one of them; were none made, a block
        // that cannot be reached takes what is placed
        if self.blocks.is_empty() {
            self.blocks.push(Block {
                params: Vec::new(),
                nodes: Vec::new(),
                term: Terminator::Unreachable,
            });
        }
        let at = self.current.min(self.blocks.len().saturating_sub(1));
        match self.blocks.get_mut(at) {
            Some(b) => b,
            None => unreachable!("the blocks are not empty and `at` is below their count"),
        }
    }
    /// A value placed in the current block.
    fn value(&mut self, ty: Type, op: Op) -> IrValue {
        let id = self.def(ty, op);
        self.block().nodes.push(Node::Value(id));
        id
    }
    fn node(&mut self, node: Node) {
        self.block().nodes.push(node);
    }
    fn ty(&self, v: IrValue) -> Option<Type> {
        self.values.get(v.0 as usize).map(|d| d.ty)
    }
}

/// The state of one lift.
pub(crate) struct Lift<'m> {
    pub(crate) m: &'m Module,
    /// The program's functions: the entry, and a hull program's patch-constant function.
    functions: Vec<&'m LlvmFunction>,
    /// The function being lifted.
    pub(crate) f: &'m LlvmFunction,
    pub(crate) setup: Setup,
    g: Graph,
    /// Each local value's translation, by `v - f.first`.
    values: Vec<Option<Lv>>,
    /// The instructions of each basic block, by index into `f.instructions`.
    llvm_blocks: Vec<Vec<usize>>,
    /// The parameters of each block, one per phi, in the order of its phis.
    params: Vec<Vec<(ValueId, IrValue)>>,
    /// The variable of each `alloca` and module global the program reaches as memory.
    memory: BTreeMap<ValueId, VarId>,
    /// The allocas and module globals an amplification program hands to `DispatchMesh`, and the variables laid out
    /// in bytes: those, and a mesh program's task payload.
    payload_roots: BTreeSet<ValueId>,
    bytes: BTreeSet<VarId>,
    /// A mesh program's task payload, made where `GetMeshPayload` first reads it.
    mesh_payload: Option<VarId>,
    /// What the lift approximates or assumes, each once.
    pub(crate) notes: BTreeSet<String>,
    /// The bit offset of the instruction being lifted, for refusals.
    pub(crate) at: usize,
    /// The lane's number where it is the invocation's rank among its wave's, made at the start of the entry block.
    lane: Option<IrValue>,
    /// The block being lifted has ended in an any-hit program's decision (`IgnoreHit`, `AcceptHitAndEndSearch`): what
    /// follows it in the block is not reached.
    decided: bool,
}

impl<'m> Lift<'m> {
    fn new(m: &'m Module, functions: Vec<&'m LlvmFunction>, setup: Setup) -> Result<Self> {
        let Some(&f) = functions.first() else {
            return Err(error(0, "a program of no functions"));
        };
        let mut lift = Lift {
            m,
            functions,
            f,
            setup,
            g: Graph {
                values: Vec::new(),
                vars: Vec::new(),
                blocks: Vec::new(),
                current: 0,
                origin: None,
            },
            values: Vec::new(),
            llvm_blocks: Vec::new(),
            params: Vec::new(),
            memory: BTreeMap::new(),
            payload_roots: BTreeSet::new(),
            bytes: BTreeSet::new(),
            mesh_payload: None,
            notes: BTreeSet::new(),
            at: 0,
            lane: None,
            decided: false,
        };
        lift.begin(f)?;
        Ok(lift)
    }

    /// Start lifting function `f`: no values, blocks or memory of its own yet.
    fn begin(&mut self, f: &'m LlvmFunction) -> Result<()> {
        let mut llvm_blocks = vec![Vec::new(); f.blocks];
        for (i, inst) in f.instructions.iter().enumerate() {
            llvm_blocks
                .get_mut(inst.block)
                .ok_or_else(|| {
                    error(
                        inst.at,
                        format!(
                            "an instruction in basic block {}, where the function has {}",
                            inst.block, f.blocks
                        ),
                    )
                })?
                .push(i);
        }
        self.f = f;
        self.g = Graph {
            values: Vec::new(),
            vars: Vec::new(),
            blocks: Vec::new(),
            current: 0,
            origin: None,
        };
        self.values = vec![None; f.locals.len()];
        self.llvm_blocks = llvm_blocks;
        self.params = vec![Vec::new(); f.blocks];
        self.memory.clear();
        self.at = 0;
        self.lane = None;
        Ok(())
    }

    /// Function `f` of the program lifted as the IR function `name`.
    fn function(&mut self, f: &'m LlvmFunction, name: &str) -> Result<Function> {
        self.begin(f)?;
        self.prepare()?;
        self.lane_at_start();
        self.blocks()?;
        self.prune();
        self.finish_function(name)
    }

    pub(crate) fn refuse<T>(&self, why: impl Into<String>) -> Result<T> {
        Err(error(self.at, why))
    }

    /// The blocks, with a parameter for each phi, before any instruction is lifted (a branch names the blocks after
    /// it); and the module's thread-group and constant memory the function reaches.
    fn prepare(&mut self) -> Result<()> {
        for b in 0..self.f.blocks {
            self.g.blocks.push(Block {
                params: Vec::new(),
                nodes: Vec::new(),
                term: Terminator::Unreachable,
            });
            let phis: Vec<(ValueId, TypeId)> = self
                .llvm_blocks
                .get(b)
                .map(|insts| {
                    insts
                        .iter()
                        .filter_map(|&i| self.f.instructions.get(i))
                        .filter(|inst| matches!(inst.op, LlvmOp::Phi { .. }))
                        .filter_map(|inst| inst.result)
                        .collect()
                })
                .unwrap_or_default();
            for (v, ty) in phis {
                let Some(t) = self.carried_type(ty)? else {
                    // a phi of handles or pointers is the same on every edge, or it is refused where it is read
                    continue;
                };
                let p = self.g.def(t, Op::Param);
                if let Some(block) = self.g.blocks.get_mut(b) {
                    block.params.push(p);
                }
                if let Some(params) = self.params.get_mut(b) {
                    params.push((v, p));
                }
                self.define(v, Lv::Value(p))?;
            }
        }
        if self.g.blocks.is_empty() {
            return self.refuse("a function of no blocks");
        }
        self.declare_memory()?;
        self.ray_parameters()?;
        Ok(())
    }

    /// The blocks in reverse post-order from the entry: each after the blocks that dominate it, so a value is lifted
    /// before every instruction that reads it (LLVM's own order of blocks need not be one), a phi's incoming values
    /// read at the end of its predecessors. A block the entry does not reach comes last.
    fn order(&self) -> Vec<usize> {
        let successors = |b: usize| -> Vec<usize> {
            let last = self
                .llvm_blocks
                .get(b)
                .and_then(|insts| insts.last())
                .and_then(|&i| self.f.instructions.get(i));
            match last.map(|i| &i.op) {
                Some(LlvmOp::Branch { target }) => vec![*target],
                Some(LlvmOp::CondBranch { then, otherwise, .. }) => vec![*then, *otherwise],
                Some(LlvmOp::Switch { default, cases, .. }) => std::iter::once(*default)
                    .chain(cases.iter().map(|&(_, b)| b))
                    .collect(),
                _ => Vec::new(),
            }
        };
        let n = self.f.blocks;
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        // an explicit stack of (block, next successor to visit)
        let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();
        if let Some(s) = seen.first_mut() {
            *s = true;
            stack.push((0, successors(0), 0));
        }
        while let Some((b, next, k)) = stack.last_mut() {
            match next.get(*k).copied() {
                Some(c) => {
                    *k = k.saturating_add(1);
                    if let Some(s) = seen.get_mut(c)
                        && !*s
                    {
                        *s = true;
                        let succ = successors(c);
                        stack.push((c, succ, 0));
                    }
                }
                None => {
                    post.push(*b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        post.extend((0..n).filter(|&b| !seen.get(b).copied().unwrap_or(true)));
        post
    }

    /// Each block's instructions in order, ending in its terminator, the blocks in [`Lift::order`].
    fn blocks(&mut self) -> Result<()> {
        for b in self.order() {
            self.g.current = b;
            self.decided = false;
            let insts = self.llvm_blocks.get(b).cloned().unwrap_or_default();
            for i in insts {
                let Some(inst) = self.f.instructions.get(i) else {
                    continue;
                };
                self.at = inst.at;
                self.g.origin = Some(Origin {
                    unit: Unit::Bit,
                    offset: u32::try_from(inst.at).unwrap_or(u32::MAX),
                });
                self.instruction(inst)?;
            }
        }
        Ok(())
    }

    /// The target of a branch to block `to` from block `from`: the block, and the value each of its phis takes from
    /// `from`.
    fn target(&mut self, from: usize, to: usize) -> Result<Target> {
        let params = self.params.get(to).cloned().unwrap_or_default();
        let mut args = Vec::new();
        for (phi, param) in params {
            let Some(LlvmOp::Phi { incoming }) = self
                .f
                .instructions
                .iter()
                .find(|i| i.result.is_some_and(|(r, _)| r == phi))
                .map(|i| &i.op)
            else {
                return self.refuse("a block parameter that is no phi");
            };
            let Some(&(v, _)) = incoming.iter().find(|(_, b)| *b == from) else {
                return self.refuse(format!("a phi of block {to} with nothing from block {from}"));
            };
            let want = self
                .g
                .ty(param)
                .ok_or_else(|| error(self.at, "a parameter of no type"))?;
            let value = self.value_as(v, want)?;
            args.push(value);
        }
        Ok(Target {
            block: u32::try_from(to).map_err(|_| error(self.at, "a block number past 2^32"))?,
            args,
        })
    }

    fn terminate(&mut self, term: Terminator) {
        self.g.block().term = term;
    }

    fn instruction(&mut self, inst: &Instruction) -> Result<()> {
        let from = inst.block;
        if self.decided {
            return match &inst.op {
                LlvmOp::Unreachable | LlvmOp::Return(_) => Ok(()),
                _ => self.refuse("an instruction after an any-hit program's decision"),
            };
        }
        match &inst.op {
            LlvmOp::Phi { .. } => Ok(()),
            LlvmOp::Return(_) => {
                self.terminate(Terminator::Return);
                Ok(())
            }
            LlvmOp::Unreachable => {
                self.terminate(Terminator::Unreachable);
                Ok(())
            }
            LlvmOp::Branch { target } => {
                let t = self.target(from, *target)?;
                self.terminate(Terminator::Jump(t));
                Ok(())
            }
            LlvmOp::CondBranch {
                condition,
                then,
                otherwise,
            } => {
                let cond = self.value_as(*condition, Type::BOOL)?;
                let accept = self.target(from, *then)?;
                let reject = self.target(from, *otherwise)?;
                self.terminate(Terminator::Branch { cond, accept, reject });
                Ok(())
            }
            LlvmOp::Switch {
                value,
                default,
                cases,
            } => {
                let selector = self.value_as(*value, Type::U32)?;
                let mut out = Vec::new();
                for &(c, b) in cases {
                    let Some(k) = crate::lift::setup::const_int(self.m, self.f, c) else {
                        return self.refuse("a switch case that is not a constant");
                    };
                    let t = self.target(from, b)?;
                    out.push((crate::lift::low32(k as u64), t));
                }
                let default = self.target(from, *default)?;
                self.terminate(Terminator::Switch {
                    selector,
                    cases: out,
                    default,
                });
                Ok(())
            }
            LlvmOp::Call { function, args, .. } => self.call(inst, *function, args),
            _ => self.plain(inst),
        }
    }

    /// Remove the values nothing reads and that change nothing: the fields of an LLVM structure the program never
    /// extracts (a sample's status, the sizes a `GetDimensions` gives that it does not use), whose operations the
    /// naga-path lift never makes either.
    fn prune(&mut self) {
        loop {
            let mut uses = vec![0u32; self.g.values.len()];
            let count = |v: IrValue, uses: &mut Vec<u32>| {
                if let Some(u) = uses.get_mut(v.0 as usize) {
                    *u = u.saturating_add(1);
                }
            };
            for b in &self.g.blocks {
                for n in &b.nodes {
                    match n {
                        Node::Value(v) => {
                            for o in self
                                .g
                                .values
                                .get(v.0 as usize)
                                .map(|d| d.op.operands())
                                .unwrap_or_default()
                            {
                                count(o, &mut uses);
                            }
                        }
                        other => {
                            for o in other.values() {
                                count(o, &mut uses);
                            }
                        }
                    }
                }
                match &b.term {
                    Terminator::Branch { cond, .. } => count(*cond, &mut uses),
                    Terminator::Switch { selector, .. } => count(*selector, &mut uses),
                    _ => {}
                }
                for t in b.term.targets() {
                    for a in &t.args {
                        count(*a, &mut uses);
                    }
                }
            }
            let values = &self.g.values;
            let mut removed = false;
            for b in &mut self.g.blocks {
                let before = b.nodes.len();
                b.nodes.retain(|n| {
                    !matches!(n, Node::Value(v) if uses.get(v.0 as usize).copied().unwrap_or(1) == 0
                        && !values.get(v.0 as usize).is_some_and(|d| d.op.has_effect()))
                });
                removed |= b.nodes.len() != before;
            }
            if !removed {
                break;
            }
        }
    }

    /// The function `name`: its values, variables and blocks.
    fn finish_function(&mut self, name: &str) -> Result<Function> {
        Ok(Function {
            name: name.into(),
            values: std::mem::take(&mut self.g.values),
            vars: std::mem::take(&mut self.g.vars),
            body: Body::Cfg(std::mem::take(&mut self.g.blocks)),
        })
    }

    /// The `dx.op` opcode of a call, if it calls one.
    pub(crate) fn opcode(&self, function: ValueId, args: &[ValueId]) -> Option<u64> {
        dx_opcode(self.m, self.f, function, args)
    }

    /// Record `v`'s translation.
    pub(crate) fn define(&mut self, v: ValueId, lv: Lv) -> Result<()> {
        let Some(i) = v.checked_sub(self.f.first) else {
            return self.refuse("a module value defined by an instruction");
        };
        match self.values.get_mut(i) {
            Some(slot) => {
                *slot = Some(lv);
                Ok(())
            }
            None => self.refuse(format!("value {v}, past the function's")),
        }
    }

    /// The LLVM type at `ty`.
    pub(crate) fn type_at(&self, ty: TypeId) -> Result<&LlvmType> {
        self.m
            .types
            .get(ty)
            .map_or_else(|| self.refuse(format!("type {ty}")), Ok)
    }
}
