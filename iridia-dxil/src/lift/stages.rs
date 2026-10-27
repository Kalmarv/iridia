//! The stages that are functions, not entry points: a hull program's two phases, a domain program, a vertex program
//! that writes no position. What reads their arguments and fills their results (see
//! [`iridia::stage_functions`]).

use iridia::build::{Kind, Val};
use iridia::{IoType, Program, StageFacts, Varying};
use naga::{Expression as E, Handle, LocalVariable};

use super::setup::const_int;
use super::{Lift, Lv};
use crate::Result;
use dxil_reader::module::{Instruction, ValueId};

impl Lift<'_> {
    /// The expression of the current function's argument `index`, made once: a pre-emitted one, readable from any
    /// block.
    pub(super) fn argument_expr(&mut self, index: usize) -> Handle<E> {
        match self.argument_exprs.get(&index) {
            Some(&e) => e,
            None => {
                let e = self
                    .b
                    .app(E::FunctionArgument(u32::try_from(index).unwrap_or(u32::MAX)));
                self.argument_exprs.insert(index, e);
                e
            }
        }
    }

    /// The argument that carries input varying `varying`, and its type.
    pub(super) fn input_argument(&self, varying: usize) -> Result<(usize, IoType)> {
        match &self.sig {
            None => {
                let io = &self.setup.io;
                match io
                    .arguments
                    .iter()
                    .enumerate()
                    .find(|(_, item)| item.varyings.first() == Some(&varying))
                {
                    Some((index, item)) => Ok((index, item.ty)),
                    None => self.refuse("a kept input varying without an argument"),
                }
            }
            Some(sig) => {
                let Some(input) = self.setup.interface.inputs.get(varying) else {
                    return self.refuse(format!(
                        "input varying {varying}, which the interface does not have"
                    ));
                };
                let name = Program::varying_name(&input.label);
                self.argument_named(sig, &name)
            }
        }
    }

    fn argument_named(&self, sig: &iridia::FunctionSig, name: &str) -> Result<(usize, IoType)> {
        match sig.arguments.iter().enumerate().find(|(_, a)| a.name == name) {
            Some((index, argument)) => Ok((index, argument.ty)),
            None => self.refuse(format!("{}: no argument {name}", sig.name)),
        }
    }

    /// The value of the argument the pipeline supplies under this name.
    fn special_argument(&mut self, name: &str) -> Result<Val> {
        let Some(sig) = self.sig.clone() else {
            return self.refuse(format!("{name} outside a stage function"));
        };
        let (index, ty) = self.argument_named(&sig, name)?;
        let IoType::Value(v) = ty else {
            return self.refuse(format!("{name} is not a value"));
        };
        let expr = self.argument_expr(index);
        Ok(Val {
            expr,
            kind: Kind::of(v.scalar),
            width: v.width,
        })
    }

    /// The patch constants of the current program: a hull program's outputs, a domain program's inputs.
    fn patch_constants(&self) -> &[Varying] {
        match &self.setup.interface.facts {
            StageFacts::Hull(h) => &h.patch_constants,
            StageFacts::Domain(d) => &d.patch_constants,
            _ => &[],
        }
    }

    /// The result of a call as the type it has, from a value carried as `kind`.
    fn typed_result(&mut self, inst: &Instruction, value: Val) -> Result<()> {
        let Some((_, ty)) = inst.result else {
            return self.refuse("a call that reads a value but has no result");
        };
        let kind = self.kind_of_type(ty)?;
        let r = self.b.bitcast_to(value, kind);
        self.define_result(inst.result, Lv::Scalar(r))
    }

    /// `LoadPatchConstant(element, row, column)` in a domain program: one lane of a patch constant.
    pub(super) fn load_patch_constant(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (Some(id), Some(row), Some(col)) = (
            const_int(self.m, self.f, self.arg(args, 1)?),
            const_int(self.m, self.f, self.arg(args, 2)?),
            const_int(self.m, self.f, self.arg(args, 3)?),
        ) else {
            return self.refuse("a patch constant read at a computed element, row or column");
        };
        let Some(element) = self
            .setup
            .patch_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("patch constant element {id}"));
        };
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        let Some((varying, position)) = patch_cell(self.patch_constants(), register, lane) else {
            return self.refuse(format!(
                "patch constant {} has no lane at register {register}",
                element.name
            ));
        };
        let Some(patch_constant) = self.patch_constants().get(varying) else {
            return self.refuse(format!(
                "patch constant varying {varying}, which the interface does not have"
            ));
        };
        let name = Program::varying_name(&patch_constant.label);
        let value = self.special_argument(&name)?;
        let lane_value = if value.width > 1 {
            self.b.lane(value, position)
        } else {
            value
        };
        self.typed_result(inst, lane_value)
    }

    /// `StorePatchConstant(element, row, column, value)` in a hull program: one lane of a patch constant, kept as raw
    /// bits in its register's local until the function returns.
    pub(super) fn store_patch_constant(&mut self, args: &[ValueId]) -> Result<()> {
        let (Some(id), Some(row), Some(col)) = (
            const_int(self.m, self.f, self.arg(args, 1)?),
            const_int(self.m, self.f, self.arg(args, 2)?),
            const_int(self.m, self.f, self.arg(args, 3)?),
        ) else {
            return self.refuse("a patch constant written at a computed element, row or column");
        };
        let Some(element) = self
            .setup
            .patch_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("patch constant element {id}"));
        };
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        if patch_cell(self.patch_constants(), register, lane).is_none() {
            return self.refuse(format!(
                "patch constant {} has no lane at register {register}",
                element.name
            ));
        }
        let value = self.scalar(self.arg(args, 4)?)?;
        let bits = self.b.bitcast_to(value, Kind::U);
        let local = self.output_local(register)?;
        let pointer = self.b.local_ptr(local);
        let target = self.b.app(E::AccessIndex {
            base: pointer,
            index: u32::from(lane),
        });
        self.b.store(target, bits);
        Ok(())
    }

    /// `LoadOutputControlPoint(element, row, column, vertex)` in a hull program's patch-constant function: one lane of
    /// an output control point.
    pub(super) fn load_output_control_point(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (Some(id), Some(row), Some(col)) = (
            const_int(self.m, self.f, self.arg(args, 1)?),
            const_int(self.m, self.f, self.arg(args, 2)?),
            const_int(self.m, self.f, self.arg(args, 3)?),
        ) else {
            return self.refuse("an output control point read at a computed element, row or column");
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
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        let Some((varying, position)) = self.setup.output_cell(register, lane) else {
            return self.refuse(format!(
                "output {} has no lane at register {register}",
                element.name
            ));
        };
        let Some(output) = self.setup.interface.outputs.get(varying) else {
            return self.refuse(format!(
                "output varying {varying}, which the interface does not have"
            ));
        };
        let name = format!("out_{}", Program::varying_name(&output.label));
        let Some(sig) = self.sig.clone() else {
            return self.refuse("an output control point read outside a stage function");
        };
        let (index, ty) = self.argument_named(&sig, &name)?;
        let IoType::Array(v, _) = ty else {
            return self.refuse("output control points are not an array");
        };
        let points = self.argument_expr(index);
        let point = self.uint(self.arg(args, 4)?)?;
        let element = Val {
            expr: self.b.app(E::Access {
                base: points,
                index: point.expr,
            }),
            kind: Kind::of(v.scalar),
            width: v.width,
        };
        let lane_value = if v.width > 1 {
            self.b.lane(element, position)
        } else {
            element
        };
        self.typed_result(inst, lane_value)
    }

    /// `DomainLocation(component)`: a component of the vertex's place in its patch.
    pub(super) fn domain_location(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let Some(component) = const_int(self.m, self.f, self.arg(args, 1)?).filter(|c| (0..3).contains(c))
        else {
            return self.refuse("a domain location read at a computed component");
        };
        let location = self.special_argument("SV_DomainLocation")?;
        let lane = self.b.lane(location, super::lane_u8(component));
        self.typed_result(inst, lane)
    }

    /// `PrimitiveID` and `OutputControlPointID`: the patch's number, and the control point a hull program's
    /// control-point phase is for.
    pub(super) fn patch_id(&mut self, inst: &Instruction, name: &str) -> Result<()> {
        let id = self.special_argument(name)?;
        self.typed_result(inst, id)
    }
}

/// One output varying of a geometry program: its array local, the register local it is emitted from, the register's
/// lanes it takes and the type it is carried as.
type GsOutputLocal = (
    Handle<LocalVariable>,
    Handle<LocalVariable>,
    Vec<u8>,
    iridia::Value,
);

/// A geometry program's output so far: how many vertices it has emitted, whether the next begins a strip, and for each
/// output varying and each vertex the value emitted.
#[derive(Clone)]
pub(super) struct GsState {
    count: Handle<LocalVariable>,
    pending: Handle<LocalVariable>,
    strips: Handle<LocalVariable>,
    outputs: Vec<GsOutputLocal>,
    max: u32,
}

impl Lift<'_> {
    /// A geometry program's output locals: an array per output varying, the count, the strip starts.
    pub(super) fn start_geometry(&mut self, max: u32) -> Result<()> {
        let Some(one) = std::num::NonZeroU32::new(max) else {
            return self.refuse("a geometry program that emits no vertices");
        };
        let u1 = self.b.ty_of(Kind::U, 1);
        let strips_ty = self.b.ty(naga::TypeInner::Array {
            base: u1,
            size: naga::ArraySize::Constant(one),
            stride: 4,
        });
        let count = self.b.local("emitted", u1);
        let pending = self.b.local("strip_pending", u1);
        let strips = self.b.local("strip_starts", strips_ty);
        let (zero, first) = (self.b.lit_u(0), self.b.lit_u(1));
        let pointer = self.b.local_ptr(count);
        self.b.store(pointer, zero);
        let pointer = self.b.local_ptr(pending);
        self.b.store(pointer, first);
        let mut outputs = Vec::new();
        let Some(sig) = self.sig.clone() else {
            return self.refuse("a geometry epilogue outside a stage function");
        };
        for (i, v) in self.setup.interface.outputs.clone().into_iter().enumerate() {
            let Some(d3d) = v.provenance.d3d else {
                return self.refuse("an output without a register");
            };
            let Some(&register) = self.outputs.get(&d3d.register) else {
                return self.refuse("an output register that is not declared");
            };
            let Some(IoType::Array(carried, _)) =
                sig.result.iter().find(|r| r.varying == Some(i)).map(|r| r.ty)
            else {
                return self.refuse("an output without its result array");
            };
            let ty = self
                .b
                .io_array(carried, max)
                .map_err(|d| crate::error(self.at, d))?;
            let array = self
                .b
                .local(&format!("emit_{}", Program::varying_name(&v.label)), ty);
            let lanes: Vec<u8> = match v.system {
                Some(iridia::SystemValue::ClipPosition) => vec![0, 1, 2, 3],
                _ => (0..4u8).filter(|l| d3d.mask >> l & 1 != 0).collect(),
            };
            outputs.push((array, register, lanes, carried));
        }
        self.gs = Some(GsState {
            count,
            pending,
            strips,
            outputs,
            max,
        });
        Ok(())
    }

    /// `EmitStream`, `CutStream`, `EmitThenCutStream` (97 to 99) on stream 0: the output registers become the next
    /// vertex (past the most vertices, nothing), and the next vertex begins a strip after a cut.
    pub(super) fn stream_op(&mut self, code: u64, args: &[ValueId]) -> Result<()> {
        if const_int(self.m, self.f, self.arg(args, 1)?) != Some(0) {
            return self.refuse("a vertex emitted to a stream other than 0");
        }
        if code != 98 {
            self.emit_vertex()?;
        }
        if code != 97 {
            let Some(gs) = self.gs.clone() else {
                return self.refuse("a strip cut outside a geometry program");
            };
            let one = self.b.lit_u(1);
            let p = self.b.local_ptr(gs.pending);
            self.b.store(p, one);
        }
        Ok(())
    }

    fn emit_vertex(&mut self) -> Result<()> {
        let Some(gs) = self.gs.clone() else {
            return self.refuse("a vertex emitted outside a geometry program");
        };
        let count = {
            let p = self.b.local_ptr(gs.count);
            self.b.load(p, Kind::U, 1)
        };
        let max = self.b.lit_u(gs.max);
        let inside = self.b.binary(naga::BinaryOperator::Less, count, max);
        self.b.when(inside, |b| {
            let index = {
                let p = b.local_ptr(gs.count);
                b.load(p, Kind::U, 1)
            };
            for (array, register, lanes, carried) in &gs.outputs {
                let p = b.local_ptr(*register);
                let row = b.load(p, Kind::U, 4);
                let picked = b.swizzle(row, lanes);
                let value = b.bitcast(picked, Kind::of(carried.scalar));
                let a = b.local_ptr(*array);
                let target = b.app(E::Access {
                    base: a,
                    index: index.expr,
                });
                b.store(target, value);
            }
            let pending = {
                let p = b.local_ptr(gs.pending);
                b.load(p, Kind::U, 1)
            };
            let s = b.local_ptr(gs.strips);
            let target = b.app(E::Access {
                base: s,
                index: index.expr,
            });
            b.store(target, pending);
            let zero = b.lit_u(0);
            let p = b.local_ptr(gs.pending);
            b.store(p, zero);
            let one = b.lit_u(1);
            let next = b.binary(naga::BinaryOperator::Add, index, one);
            let p = b.local_ptr(gs.count);
            b.store(p, next);
        });
        Ok(())
    }

    /// The geometry program's result: the emitted vertices, how many, and where strips start.
    fn geometry_epilogue(&mut self) -> Result<()> {
        let Some(result) = self.result_type else {
            return self.refuse("a geometry program that returns nothing");
        };
        let Some(gs) = self.gs.clone() else {
            return self.refuse("a geometry program without its vertices");
        };
        let mut parts = Vec::new();
        for (array, ..) in &gs.outputs {
            let p = self.b.local_ptr(*array);
            parts.push(self.b.load(p, Kind::U, 1).expr);
        }
        let p = self.b.local_ptr(gs.count);
        parts.push(self.b.load(p, Kind::U, 1).expr);
        let p = self.b.local_ptr(gs.strips);
        parts.push(self.b.load(p, Kind::U, 1).expr);
        let composed = self.b.app(E::Compose {
            ty: result,
            components: parts,
        });
        self.b.push(naga::Statement::Return {
            value: Some(composed),
        });
        Ok(())
    }

    /// Pack the result registers into the function's result structure and return it.
    pub(super) fn function_epilogue(&mut self) -> Result<()> {
        if self.gs.is_some() {
            return self.geometry_epilogue();
        }
        let Some(result) = self.result_type else {
            self.b.push(naga::Statement::Return { value: None });
            return Ok(());
        };
        let Some(sig) = self.sig.clone() else {
            return self.refuse("a stage function's epilogue outside a stage function");
        };
        let varyings = self.result_varyings();
        let mut parts = Vec::new();
        for item in &sig.result {
            let (IoType::Value(carried), Some(index)) = (item.ty, item.varying) else {
                return self.refuse("a result that is not a varying's value");
            };
            let Some(v) = varyings.get(index) else {
                return self.refuse(format!(
                    "{} names varying {index}, which the interface does not have",
                    item.name
                ));
            };
            let Some(d3d) = v.provenance.d3d else {
                return self.refuse(format!("{} has no register", item.name));
            };
            let Some(&local) = self.outputs.get(&d3d.register) else {
                return self.refuse(format!("{} has no register {}", item.name, d3d.register));
            };
            let lanes: Vec<u8> = match v.system {
                Some(iridia::SystemValue::ClipPosition) => vec![0, 1, 2, 3],
                _ => (0..4u8).filter(|l| d3d.mask >> l & 1 != 0).collect(),
            };
            let p = self.b.local_ptr(local);
            let row = self.b.load(p, Kind::U, 4);
            let picked = self.b.swizzle(row, &lanes);
            parts.push(self.b.bitcast(picked, Kind::of(carried.scalar)).expr);
        }
        let composed = self.b.app(E::Compose {
            ty: result,
            components: parts,
        });
        self.b.push(naga::Statement::Return {
            value: Some(composed),
        });
        Ok(())
    }
}

/// The patch-constant varying whose register and lanes hold this cell, and the lane's position in its value.
fn patch_cell(list: &[Varying], register: u32, lane: u8) -> Option<(usize, u8)> {
    list.iter().enumerate().find_map(|(i, v)| {
        let d3d = v.provenance.d3d?;
        (d3d.register == register && lanes_below(d3d.mask, lane).is_some())
            .then(|| (i, lanes_below(d3d.mask, lane).unwrap_or(0)))
    })
}

/// How many lanes of `mask` come before `lane`, where the mask names `lane`.
pub(super) fn lanes_below(mask: u8, lane: u8) -> Option<u8> {
    let lane = u32::from(lane);
    (mask.checked_shr(lane)? & 1 != 0).then(|| {
        let below = (u32::from(mask) & 1u32.checked_shl(lane).unwrap_or(0).wrapping_sub(1)).count_ones();
        u8::try_from(below).unwrap_or(0)
    })
}
