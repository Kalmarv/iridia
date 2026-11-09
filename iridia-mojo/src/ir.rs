//! Lifting the dialect to an [`iridia_ir::Program`], read and lowered in one pass: declarations, then `main`'s
//! statements, each lowered as it is read (an `if` and a loop open their constructs in the builder).
//!
//! Each register (`ps_r0`, `vs_a0`) and each output (`gl_Position`, a varying a vertex program writes, a target) is a
//! variable; the outputs are stored when `main` ends. The interface, the resources and the notes are [`crate::lift`]'s.
//! GLSL's rules are stated in the operations: `*` is IEEE 754's multiply, `min`, `max` and `clamp` leave the result of
//! a NaN operand open, a derivative's fineness is the driver's, and `discard` demotes the invocation (as WGSL's
//! `discard` and Direct3D's `texkill` do). Where GLSL leaves a result undefined and the dialect reads as Direct3D 9
//! does, the program does as Direct3D 9 does, a policy it states (`Provenance::policies`): a float converted to an
//! integer saturates (`glsl.float_to_int=saturate`); `clamp(x, 0.0, 1.0)`, which MojoShader writes for the `_sat`
//! modifier, is Direct3D 9's saturate, +0 for -0 and 0 for NaN (`glsl.clamp01=saturate`); `inversesqrt` and `log2`, which MojoShader writes for `rsq` and
//! `log` of the operand where Direct3D 9 takes its magnitude, take its magnitude (`glsl.inversesqrt_log2=magnitude`);
//! `normalize` of a vector of length 0 is 0, as `nrm` (`glsl.normalize_zero=zero`); and a row of the uniform array
//! past its end reads zero (`glsl.uniform_oob=zero`). The built-in functions GLSL defines by others are written as
//! their definitions: `clamp(x, a, b)` is `min(max(x, a), b)`, `mix(x, y, a)` is `x * (1 - a) + y * a`, `length(x)` is
//! `sqrt(dot(x, x))`, `normalize(x)` is `x * inversesqrt(dot(x, x))` (scaled by the largest float where the dot is 0)
//! and `cross` its six products.

use std::collections::BTreeMap;

use iridia_ir::interface::{
    BufferLayout, Constant as LayoutConstant, ConstantType, D3dBinding, D3dClass, D3dSignature, Interface,
    InterpMode, InterpSampling, Interpolation, Label, Resource, ResourceId, ResourceKind, ResourceProvenance,
    SampleType, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, TextureDim, Value, Varying,
    VaryingProvenance,
};
use iridia_ir::{
    Address, Axis, Binary, Builder, Cmp, Conventions, End, Fineness, FloatEnv, FloatToInt, LabelId, Level,
    MinMaxNan, MulZero, Op, Origin, OutOfBounds, Program, Provenance, Res, Rounding, SaturateNan, Scalar,
    Type, Unary, Unit, ValueId, VarId, source_hash,
};

use crate::lex::{self, Token};
use crate::parse::{Cursor, swizzle_lanes, varying_semantic};
use crate::{Hints, Result, error};

/// The lifter's name in a program's provenance.
pub const LIFTER: &str = "iridia-mojo";

/// The compatibility policies every program states: where GLSL leaves a result undefined, Direct3D 9's.
const POLICIES: [&str; 5] = [
    "glsl.clamp01=saturate",
    "glsl.float_to_int=saturate",
    "glsl.inversesqrt_log2=magnitude",
    "glsl.normalize_zero=zero",
    "glsl.uniform_oob=zero",
];

/// Increment when the same text lifts differently; gated by `tests/ir.rs` (the programs there lifted and hashed under
/// it, `iridia_ir::gate`, in `tests/ir-lifted.hashes`).
pub const LIFTER_VERSION: u32 = 1;

/// Lift the vertex or pixel program `source` onto the IR. Only [`Stage::Vertex`] and [`Stage::Pixel`] are programs of
/// this dialect; what the program holds that the dialect does not have is refused, naming the line.
pub fn lift(source: &str, stage: Stage, hints: &Hints) -> Result<Program> {
    if !matches!(stage, Stage::Vertex | Stage::Pixel) {
        return Err(error(
            0,
            format!("{} program is not in the dialect", stage.with_article()),
        ));
    }
    let lexed = lex::lex(source)?;
    for e in &lexed.extensions {
        if e != "GL_OES_standard_derivatives : enable" {
            return Err(error(0, format!("the extension {e}")));
        }
    }
    let mut l = Lower {
        b: Builder::new("main"),
        c: Cursor::new(lexed.tokens),
        stage,
        hints,
        symbols: BTreeMap::new(),
        rows: None,
        samplers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        notes: Vec::new(),
        in_main: false,
        loops: Vec::new(),
        ends: vec![None],
    };
    l.prepare_outputs()?;
    l.declarations()?;
    l.main()?;
    l.finish(source)
}

const F32X4: Type = Type::F32X4;
const I32X4: Type = Type::vector(Scalar::I32, 4);

/// What a name stands for.
#[derive(Clone, Debug)]
enum Sym {
    /// A register or an output: a variable of this type.
    Var(VarId, Type),
    /// A constant (`const vec4 ps_c5 = vec4(...)`): its value, and its lanes where they are integer literals (a loop's
    /// bound is read from them).
    Const(V, Option<[i64; 4]>),
    /// The uniform array of constants, of this many rows.
    Constants(u32),
    /// A sampler, by its index in the program's samplers.
    Sampler(usize),
    /// An input of the interface (an attribute, or a varying a pixel program reads), by slot, and its type.
    Input(u32, Type),
    /// `posFixup`, which only the dropped epilogue reads.
    PosFixup,
    /// A loop's counter.
    Loop(V),
}

/// A value, its type, and its number if it is an integer literal or a sum, difference or product of them.
#[derive(Clone, Copy, Debug)]
struct V {
    v: ValueId,
    ty: Type,
    literal: Option<i64>,
}

impl V {
    fn is(&self, scalar: Scalar) -> bool {
        self.ty.scalar == scalar
    }
    fn lanes(&self) -> u8 {
        self.ty.lanes
    }
    fn is_bool_scalar(&self) -> bool {
        self.ty == Type::BOOL
    }
}

/// The name GLSL gives a value's scalar type, as a refusal says it.
fn kind(t: Type) -> &'static str {
    match t.scalar {
        Scalar::F32 => "F",
        Scalar::I32 => "I",
        Scalar::U32 => "U",
        _ => "B",
    }
}

struct Sampler {
    glsl_name: String,
    slot: u32,
    name: String,
    /// `sampler2D`, `samplerCube` or `sampler3D`.
    dim: TextureDim,
}

/// The lanes of a coordinate of a texture of `dim`, and the function that samples it.
fn sampling(dim: TextureDim) -> (u8, &'static str) {
    match dim {
        TextureDim::Cube => (3, "textureCube"),
        TextureDim::D3 => (3, "texture3D"),
        _ => (2, "texture2D"),
    }
}

/// The mask of every lane of a value of `ty`.
fn every(ty: Type) -> u8 {
    (1u8 << ty.lanes.min(4)).wrapping_sub(1)
}

struct Lower<'a> {
    b: Builder,
    c: Cursor,
    stage: Stage,
    hints: &'a Hints,
    symbols: BTreeMap<String, Sym>,
    /// The rows of the uniform array, once declared.
    rows: Option<u32>,
    samplers: Vec<Sampler>,
    inputs: Vec<Varying>,
    /// The outputs in the interface's order, and the variable that holds each.
    outputs: Vec<(Varying, VarId)>,
    notes: Vec<String>,
    /// Whether the declarations are read: the uniforms, samplers, registers and inputs are values only in `main`.
    in_main: bool,
    /// The labels of the open loops, innermost last: what a `break` leaves.
    loops: Vec<LabelId>,
    /// How each open region ends, innermost last: `None` until a `break` ends it.
    ends: Vec<Option<End>>,
}

fn float_value(width: u8) -> Value {
    Value {
        scalar: ScalarType::Float,
        width,
    }
}

fn varying(label: Label, system: Option<SystemValue>, interp: Option<Interpolation>, width: u8) -> Varying {
    Varying {
        label,
        system,
        value: float_value(width),
        interp,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

impl Lower<'_> {
    // ---- the builder ----

    fn op(&mut self, op: Op) -> Result<V> {
        let line = self.c.line();
        let v = self.b.op(op).map_err(|e| error(line, e))?;
        let ty = self.b.ty(v).ok_or_else(|| error(line, "a value with no type"))?;
        Ok(V { v, ty, literal: None })
    }

    fn typed(&mut self, ty: Type, op: Op) -> V {
        V {
            v: self.b.value(ty, op),
            ty,
            literal: None,
        }
    }

    fn unary(&mut self, u: Unary, a: V) -> Result<V> {
        self.op(Op::Unary(u, a.v))
    }

    fn binary(&mut self, op: Binary, a: V, b: V) -> Result<V> {
        self.op(Op::Binary(op, a.v, b.v))
    }

    fn f32(&mut self, x: f32) -> V {
        V {
            v: self.b.f32(x),
            ty: Type::F32,
            literal: None,
        }
    }

    fn i32(&mut self, x: i32) -> V {
        V {
            v: self.b.i32(x),
            ty: Type::I32,
            literal: Some(i64::from(x)),
        }
    }

    /// A scalar repeated in `lanes` lanes; a value already that wide is itself.
    fn splat(&mut self, v: V, lanes: u8) -> V {
        if v.lanes() == lanes {
            return v;
        }
        self.typed(Type::vector(v.ty.scalar, lanes), Op::Splat(v.v))
    }

    /// Lanes of a value by index, a scalar for one; of a scalar, its one lane repeated.
    fn swizzle(&mut self, v: V, lanes: &[u8]) -> Result<V> {
        let n = u8::try_from(lanes.len())
            .ok()
            .filter(|n| (1..=4).contains(n))
            .ok_or_else(|| error(self.c.line(), "a swizzle of more than four lanes"))?;
        if v.lanes() == 1 {
            return Ok(self.splat(v, n));
        }
        if lanes.iter().copied().eq(0..v.lanes()) {
            return Ok(v);
        }
        let mut all = [0u8; 4];
        for (to, from) in all.iter_mut().zip(lanes) {
            *to = *from;
        }
        Ok(self.typed(Type::vector(v.ty.scalar, n), Op::Swizzle(v.v, all)))
    }

    fn lane(&mut self, v: V, n: u8) -> Result<V> {
        self.swizzle(v, &[n])
    }

    /// `a` and `b` of one width: a scalar beside a vector splat to its lanes.
    fn widen(&mut self, a: V, b: V) -> (V, V) {
        match (a.lanes(), b.lanes()) {
            (1, n) if n > 1 => (self.splat(a, n), b),
            (n, 1) if n > 1 => (a, self.splat(b, n)),
            _ => (a, b),
        }
    }

    fn convert(&mut self, v: V, to: Scalar) -> V {
        self.typed(v.ty.with(to), Op::Convert(v.v, FloatToInt::Saturate))
    }

    /// The variable's value.
    fn load(&mut self, var: VarId, ty: Type) -> V {
        self.typed(ty, Op::LoadVar(var))
    }

    /// The resource of the uniform array, the first of the program's.
    fn constants_id() -> ResourceId {
        ResourceId(0)
    }

    /// The resources of sampler `index`: its texture and its sampler, after the uniform array's.
    fn sampler_ids(&self, index: usize) -> Result<(ResourceId, ResourceId)> {
        let base = u32::from(self.rows.is_some());
        let texture = u32::try_from(index)
            .ok()
            .and_then(|i| i.checked_mul(2))
            .and_then(|i| i.checked_add(base))
            .ok_or_else(|| error(self.c.line(), "more than 4294967295 resources"))?;
        let sampler = texture
            .checked_add(1)
            .ok_or_else(|| error(self.c.line(), "more than 4294967295 resources"))?;
        Ok((ResourceId(texture), ResourceId(sampler)))
    }

    // ---- the entry point's edges ----

    /// A pixel program writes `gl_FragColor` or `gl_FragData[n]`, which it never declares: the targets it writes are
    /// found in its text. A vertex program writes `gl_Position`.
    fn prepare_outputs(&mut self) -> Result<()> {
        match self.stage {
            Stage::Vertex => {
                let var = self.b.var("position", F32X4);
                self.symbols.insert("gl_Position".into(), Sym::Var(var, F32X4));
                let label = Label {
                    name: "POSITION".into(),
                    index: 0,
                };
                self.outputs
                    .push((varying(label, Some(SystemValue::ClipPosition), None, 4), var));
            }
            _ => {
                for n in self.c.fragment_targets()? {
                    let var = self.b.var(&format!("target{n}"), F32X4);
                    // `gl_FragColor` is target 0
                    self.symbols
                        .insert(format!("gl_FragData[{n}]"), Sym::Var(var, F32X4));
                    if n == 0 {
                        self.symbols.insert("gl_FragColor".into(), Sym::Var(var, F32X4));
                    }
                    let label = Label {
                        name: "SV_Target".into(),
                        index: n,
                    };
                    self.outputs
                        .push((varying(label, Some(SystemValue::Target(n)), None, 4), var));
                }
            }
        }
        Ok(())
    }

    /// Name `name` as `sym`; a name the program already declared (or the entry point's own outputs) is not declared again.
    fn declare_symbol(&mut self, name: &str, sym: Sym) -> Result<()> {
        if self.symbols.contains_key(name) {
            return self.c.fail(format!("{name} declared twice"));
        }
        self.symbols.insert(name.to_owned(), sym);
        Ok(())
    }

    // ---- declarations ----

    fn declarations(&mut self) -> Result<()> {
        let prefix = if self.stage == Stage::Vertex { "vs_" } else { "ps_" };
        loop {
            if self.c.at_ident("void") {
                return Ok(());
            }
            let storage = self.c.ident()?;
            match storage.as_str() {
                "uniform" => {
                    let ty = self.c.ident()?;
                    let name = self.c.ident()?;
                    match ty.as_str() {
                        "vec4" if self.c.accept("[") => {
                            let rows = self.c.integer()?;
                            self.c.expect("]")?;
                            self.c.expect(";")?;
                            if name != format!("{prefix}uniforms_vec4") {
                                return self.c.fail(format!("the uniform array {name}"));
                            }
                            let rows = u32::try_from(rows).ok().filter(|r| *r > 0).ok_or_else(|| {
                                error(self.c.line(), format!("a uniform array of {rows} rows"))
                            })?;
                            if self.rows.is_some() {
                                return self.c.fail(format!("{name} declared twice"));
                            }
                            self.rows = Some(rows);
                            self.declare_symbol(&name, Sym::Constants(rows))?;
                        }
                        "vec4" if name == "posFixup" => {
                            self.c.expect(";")?;
                            self.declare_symbol(&name, Sym::PosFixup)?;
                        }
                        "sampler2D" | "samplerCube" | "sampler3D" => {
                            self.c.expect(";")?;
                            let dim = match ty.as_str() {
                                "samplerCube" => TextureDim::Cube,
                                "sampler3D" => TextureDim::D3,
                                _ => TextureDim::D2,
                            };
                            self.declare_sampler(&name, dim)?;
                        }
                        other => return self.c.fail(format!("a uniform {other} {name}")),
                    }
                }
                "attribute" if self.stage == Stage::Vertex => {
                    self.c.word("vec4")?;
                    let name = self.c.ident()?;
                    self.c.expect(";")?;
                    self.declare_attribute(&name)?;
                }
                "varying" => {
                    // MojoShader declares the fog varying a float, every other a vec4
                    let width = match self.c.ident()?.as_str() {
                        "vec4" => 4,
                        "float" => 1,
                        other => return self.c.fail(format!("vec4 expected, found {other}")),
                    };
                    let name = self.c.ident()?;
                    self.c.expect(";")?;
                    self.declare_varying(&name, width)?;
                }
                "const" => {
                    let ty = self.c.ident()?;
                    let name = self.c.ident()?;
                    self.c.expect("=")?;
                    self.declare_const(&ty, &name)?;
                }
                "vec4" | "ivec4" | "bvec4" => {
                    let name = self.c.ident()?;
                    self.c.expect(";")?;
                    let ty = match storage.as_str() {
                        "vec4" => F32X4,
                        "ivec4" => I32X4,
                        _ => Type::vector(Scalar::Bool, 4),
                    };
                    let var = self.b.var(&name, ty);
                    self.declare_symbol(&name, Sym::Var(var, ty))?;
                }
                other => return self.c.fail(format!("a declaration beginning {other}")),
            }
        }
    }

    fn declare_sampler(&mut self, name: &str, dim: TextureDim) -> Result<()> {
        let Some(hint) = self.hints.samplers.iter().find(|s| s.glsl_name == name) else {
            return self
                .c
                .fail(format!("the sampler {name}, which the caller does not describe"));
        };
        let index = self.samplers.len();
        self.samplers.push(Sampler {
            glsl_name: name.to_owned(),
            slot: hint.slot,
            name: hint.name.clone(),
            dim,
        });
        self.declare_symbol(name, Sym::Sampler(index))
    }

    fn input_slot(&self) -> Result<u32> {
        u32::try_from(self.inputs.len()).or_else(|_| self.c.fail("more than 4294967295 inputs"))
    }

    fn declare_attribute(&mut self, name: &str) -> Result<()> {
        let Some(hint) = self.hints.attributes.iter().find(|a| a.glsl_name == name) else {
            return self.c.fail(format!(
                "the attribute {name}, which the caller does not describe"
            ));
        };
        let register = name
            .strip_prefix("vs_v")
            .and_then(|n| n.parse::<u32>().ok())
            .map(|register| D3dSignature {
                register,
                mask: 0xF,
                read_write_mask: 0,
            });
        let slot = self.input_slot()?;
        let label = Label {
            name: hint.semantic.clone(),
            index: hint.index,
        };
        self.inputs.push(Varying {
            provenance: VaryingProvenance {
                d3d: register,
                ..Default::default()
            },
            ..varying(label, None, None, 4)
        });
        self.declare_symbol(name, Sym::Input(slot, F32X4))
    }

    fn declare_varying(&mut self, name: &str, width: u8) -> Result<()> {
        let Some((semantic, index)) = varying_semantic(name) else {
            return self.c.fail(format!("the varying {name}"));
        };
        let label = Label {
            name: semantic.into(),
            index,
        };
        if self.stage == Stage::Pixel {
            let slot = self.input_slot()?;
            let interp = Interpolation {
                mode: InterpMode::Perspective,
                sampling: InterpSampling::Center,
            };
            self.inputs.push(varying(label, None, Some(interp), width));
            self.declare_symbol(name, Sym::Input(slot, Type::vector(Scalar::F32, width)))
        } else {
            let ty = Type::vector(Scalar::F32, width);
            let var = self.b.var(name, ty);
            self.outputs.push((varying(label, None, None, width), var));
            self.declare_symbol(name, Sym::Var(var, ty))
        }
    }

    fn declare_const(&mut self, ty: &str, name: &str) -> Result<()> {
        match ty {
            "ivec4" => {
                self.c.word("ivec4")?;
                self.c.expect("(")?;
                let mut lanes = [0i64; 4];
                let mut parts = [0u64; 4];
                for (k, (lane, part)) in lanes.iter_mut().zip(&mut parts).enumerate() {
                    if k > 0 {
                        self.c.expect(",")?;
                    }
                    let negative = self.c.accept("-");
                    let n = self.c.integer()?;
                    *lane = if negative {
                        n.checked_neg()
                            .ok_or_else(|| error(self.c.line(), "an integer constant over 64 bits"))?
                    } else {
                        n
                    };
                    let value = i32::try_from(*lane)
                        .map_err(|_| error(self.c.line(), "an integer constant over 32 bits"))?;
                    *part = u64::from(value.cast_unsigned());
                }
                self.c.expect(")")?;
                self.c.expect(";")?;
                let v = self.typed(I32X4, Op::Literal(parts));
                self.declare_symbol(name, Sym::Const(v, Some(lanes)))?;
            }
            "vec4" => {
                let v = self.expression()?;
                self.c.expect(";")?;
                if v.ty != F32X4 {
                    return self
                        .c
                        .fail(format!("the constant {name}, which is not four floats"));
                }
                self.declare_symbol(name, Sym::Const(v, None))?;
            }
            other => return self.c.fail(format!("a constant of type {other}")),
        }
        Ok(())
    }

    // ---- main ----

    fn main(&mut self) -> Result<()> {
        if let Some(note) = self.c.open_main(self.stage == Stage::Vertex)? {
            self.notes.push(note);
        }
        self.in_main = true;
        while !self.c.main_done() {
            self.statement()?;
        }
        Ok(())
    }

    fn block_or_statement(&mut self) -> Result<()> {
        if self.c.accept("{") {
            while !self.c.at("}") {
                if self.c.peek().is_none() {
                    return self.c.fail("a block that is not closed");
                }
                self.statement()?;
            }
            self.c.expect("}")
        } else {
            self.statement()
        }
    }

    /// Open a region (an `if`'s arm, a loop's body): nothing ends it yet.
    fn open_region(&mut self) {
        self.ends.push(None);
    }

    /// Close the region being read: how it ends, a `break`'s branch or `otherwise`.
    fn close_region(&mut self, otherwise: End) -> End {
        self.ends.pop().flatten().unwrap_or(otherwise)
    }

    fn statement(&mut self) -> Result<()> {
        let line = self.c.line();
        if self.ends.last().is_some_and(Option::is_some) {
            return self.c.fail("a statement after break");
        }
        self.b.at(Some(Origin {
            unit: Unit::Line,
            offset: u32::try_from(line).unwrap_or(u32::MAX),
        }));
        let built = |e: String| error(line, e);
        if self.c.at_ident("if") {
            self.c.pos = self.c.pos.saturating_add(1);
            self.c.expect("(")?;
            let c = self.expression()?;
            self.c.expect(")")?;
            if !c.is_bool_scalar() {
                return self.c.fail("an if on something that is not one boolean");
            }
            self.b.open_if(c.v);
            self.open_region();
            self.block_or_statement()?;
            let accept = self.close_region(End::Yield(vec![]));
            if self.c.at_ident("else") {
                self.c.pos = self.c.pos.saturating_add(1);
                self.b.else_(accept).map_err(built)?;
                self.open_region();
                self.block_or_statement()?;
                let reject = self.close_region(End::Yield(vec![]));
                self.b.close(reject, &[]).map_err(built)?;
            } else {
                self.b.close(accept, &[]).map_err(built)?;
            }
            return Ok(());
        }
        if self.c.at_ident("for") {
            return self.for_loop();
        }
        if self.c.at_ident("discard") {
            if self.stage != Stage::Pixel {
                return self.c.fail("discard in a vertex program");
            }
            self.c.pos = self.c.pos.saturating_add(1);
            self.c.expect(";")?;
            self.b.demote();
            return Ok(());
        }
        if self.c.at_ident("break") {
            self.c.pos = self.c.pos.saturating_add(1);
            self.c.expect(";")?;
            let Some(&label) = self.loops.last() else {
                return self.c.fail("break outside a loop");
            };
            if let Some(end) = self.ends.last_mut() {
                *end = Some(End::Br(label, vec![]));
            }
            return Ok(());
        }
        if self.c.at_ident("return") || self.c.at_ident("while") {
            let word = self.c.ident()?;
            return self.c.fail(word);
        }
        self.assignment()
    }

    /// `for (int repN = A; repN < ps_iM.x; repN++) { .. }`, a `rep` loop of constant count: a loop whose parameter is
    /// the counter, leaving once it reaches the bound.
    fn for_loop(&mut self) -> Result<()> {
        self.c.pos = self.c.pos.saturating_add(1);
        self.c.expect("(")?;
        self.c.word("int")?;
        let counter = self.c.ident()?;
        self.c.expect("=")?;
        let start = self.c.integer()?;
        self.c.expect(";")?;
        if self.c.ident()? != counter {
            return self.c.fail("a loop that tests something other than its counter");
        }
        self.c.expect("<")?;
        let bound = self.c.ident()?;
        self.c.expect(".")?;
        let lane = self.c.ident()?;
        self.c.expect(";")?;
        if self.c.ident()? != counter {
            return self.c.fail("a loop that steps something other than its counter");
        }
        self.c.expect("++")?;
        self.c.expect(")")?;
        let Some(Sym::Const(_, Some(lanes))) = self.symbols.get(&bound).cloned() else {
            return self.c.fail(format!(
                "a loop bounded by {bound}, which is not a constant integer vector"
            ));
        };
        let lane = swizzle_lanes(&lane).and_then(|l| match l[..] {
            [one] => Some(one),
            _ => None,
        });
        let Some(lane) = lane else {
            return self.c.fail("a loop bound that is not one lane");
        };
        let line = self.c.line();
        let end = lanes
            .get(usize::from(lane))
            .copied()
            .filter(|end| end.checked_sub(start).is_some_and(|n| u32::try_from(n).is_ok()))
            .ok_or_else(|| error(line, "a loop that does not count up"))?;
        let start = i32::try_from(start).map_err(|_| error(line, "a loop start over 32 bits"))?;
        let end = i32::try_from(end).map_err(|_| error(line, "a loop bound over 32 bits"))?;
        let first = self.i32(start);
        let (label, params) = self.b.open_loop(&[first.v]).map_err(|e| error(line, e))?;
        let i = V {
            v: *params
                .first()
                .ok_or_else(|| error(line, "a loop with no counter"))?,
            ty: Type::I32,
            literal: None,
        };
        let bound = self.i32(end);
        let done = self.binary(Binary::SCmp(Cmp::Ge), i, bound)?;
        self.b.open_if(done.v);
        self.b
            .close(End::Br(label, vec![]), &[])
            .map_err(|e| error(line, e))?;
        let saved = self.symbols.insert(counter.clone(), Sym::Loop(i));
        self.loops.push(label);
        self.open_region();
        self.block_or_statement()?;
        self.loops.pop();
        match saved {
            Some(s) => self.symbols.insert(counter, s),
            None => self.symbols.remove(&counter),
        };
        let line = self.c.line();
        // a body that ends in a break leaves; any other steps the counter and goes round again
        let end = match self.ends.pop().flatten() {
            Some(end) => end,
            None => {
                let one = self.i32(1);
                let next = self.binary(Binary::IAdd, i, one)?;
                End::Continue(label, vec![next.v])
            }
        };
        self.b.close(end, &[]).map_err(|e| error(line, e))?;
        Ok(())
    }

    fn assignment(&mut self) -> Result<()> {
        let line = self.c.line();
        let name = self.c.ident()?;
        // `gl_FragData[n]` is one name
        let name = if name == "gl_FragData" {
            self.c.expect("[")?;
            let n = self.c.integer()?;
            self.c.expect("]")?;
            format!("gl_FragData[{n}]")
        } else {
            name
        };
        let lanes = if self.c.accept(".") {
            let w = self.c.ident()?;
            swizzle_lanes(&w).ok_or_else(|| error(line, format!("the swizzle {w}")))?
        } else {
            Vec::new()
        };
        let Some(Sym::Var(var, ty)) = self.symbols.get(&name).cloned() else {
            return Err(error(line, format!("an assignment to {name}")));
        };
        let lanes = if lanes.is_empty() {
            (0..ty.lanes).collect()
        } else {
            lanes
        };
        if lanes.iter().any(|l| *l >= ty.lanes)
            || lanes
                .iter()
                .enumerate()
                .any(|(k, lane)| lanes.iter().take(k).any(|earlier| earlier == lane))
        {
            return Err(error(
                line,
                format!("a write mask of {name} that repeats or leaves it"),
            ));
        }
        if !self.c.accept("=") {
            let op = self.c.next()?.text();
            return Err(error(line, format!("the assignment {op}")));
        }
        let mut value = self.expression()?;
        self.c.expect(";")?;
        if value.ty.scalar != ty.scalar {
            return Err(error(
                line,
                format!("a {} stored in a {} register", kind(value.ty), kind(ty)),
            ));
        }
        let count = u8::try_from(lanes.len()).map_err(|_| error(line, "a write mask over four lanes"))?;
        if value.lanes() == 1 && count > 1 {
            value = self.splat(value, count);
        }
        if value.lanes() != count {
            return Err(error(
                line,
                format!("{} lanes stored in {} of {name}", value.lanes(), lanes.len()),
            ));
        }
        let whole = if lanes.iter().copied().eq(0..ty.lanes) {
            value
        } else {
            let old = self.load(var, ty);
            let mut parts = Vec::new();
            for n in 0..ty.lanes {
                let part = match lanes.iter().position(|l| *l == n) {
                    Some(k) => self.lane(value, u8::try_from(k).unwrap_or(0))?,
                    None => self.lane(old, n)?,
                };
                parts.push(part.v);
            }
            self.typed(ty, Op::Compose(parts))
        };
        self.b.store_var(var, whole.v);
        Ok(())
    }

    // ---- expressions ----

    fn expression(&mut self) -> Result<V> {
        self.ternary()
    }

    fn ternary(&mut self) -> Result<V> {
        let c = self.or()?;
        if !self.c.accept("?") {
            return Ok(c);
        }
        let a = self.expression()?;
        self.c.expect(":")?;
        let b = self.expression()?;
        if !c.is_bool_scalar() {
            return self.c.fail("a condition that is not one boolean");
        }
        if a.ty != b.ty {
            return self.c.fail("the two sides of ?: differ in type or width");
        }
        self.op(Op::Select {
            cond: c.v,
            accept: a.v,
            reject: b.v,
        })
    }

    fn or(&mut self) -> Result<V> {
        let mut a = self.and()?;
        while self.c.accept("||") {
            let b = self.and()?;
            a = self.logical(Binary::Or, a, b)?;
        }
        Ok(a)
    }

    fn and(&mut self) -> Result<V> {
        let mut a = self.equality()?;
        while self.c.accept("&&") {
            let b = self.equality()?;
            a = self.logical(Binary::And, a, b)?;
        }
        Ok(a)
    }

    fn logical(&mut self, op: Binary, a: V, b: V) -> Result<V> {
        if !a.is_bool_scalar() || !b.is_bool_scalar() {
            return self
                .c
                .fail("a logical operator on something that is not one boolean");
        }
        self.binary(op, a, b)
    }

    fn equality(&mut self) -> Result<V> {
        let mut a = self.relational()?;
        loop {
            let cmp = if self.c.at("==") {
                Cmp::Eq
            } else if self.c.at("!=") {
                Cmp::Ne
            } else {
                return Ok(a);
            };
            self.c.pos = self.c.pos.saturating_add(1);
            let b = self.relational()?;
            a = self.compare(cmp, a, b)?;
        }
    }

    fn relational(&mut self) -> Result<V> {
        let mut a = self.additive()?;
        loop {
            let cmp = if self.c.at("<") {
                Cmp::Lt
            } else if self.c.at(">") {
                Cmp::Gt
            } else if self.c.at("<=") {
                Cmp::Le
            } else if self.c.at(">=") {
                Cmp::Ge
            } else {
                return Ok(a);
            };
            self.c.pos = self.c.pos.saturating_add(1);
            let b = self.additive()?;
            a = self.compare(cmp, a, b)?;
        }
    }

    /// The comparison of two scalars of one type, or (`lessThan`) two vectors lane by lane.
    fn comparison(&mut self, cmp: Cmp, a: V, b: V) -> Result<V> {
        let op = match a.ty.scalar {
            Scalar::F32 => Binary::FCmp(cmp),
            Scalar::I32 => Binary::SCmp(cmp),
            _ if matches!(cmp, Cmp::Eq | Cmp::Ne) => Binary::UCmp(cmp),
            _ => return self.c.fail("an ordering of booleans"),
        };
        self.binary(op, a, b)
    }

    fn compare(&mut self, cmp: Cmp, a: V, b: V) -> Result<V> {
        if a.ty.scalar != b.ty.scalar || a.lanes() != 1 || b.lanes() != 1 {
            return self
                .c
                .fail("a comparison of values that are not scalars of one type");
        }
        self.comparison(cmp, a, b)
    }

    fn additive(&mut self) -> Result<V> {
        let mut a = self.multiplicative()?;
        loop {
            let op = if self.c.at("+") {
                '+'
            } else if self.c.at("-") {
                '-'
            } else {
                return Ok(a);
            };
            self.c.pos = self.c.pos.saturating_add(1);
            let b = self.multiplicative()?;
            a = self.arithmetic(op, a, b)?;
        }
    }

    fn multiplicative(&mut self) -> Result<V> {
        let mut a = self.unary_expression()?;
        loop {
            let op = if self.c.at("*") {
                '*'
            } else if self.c.at("/") {
                '/'
            } else {
                return Ok(a);
            };
            self.c.pos = self.c.pos.saturating_add(1);
            let b = self.unary_expression()?;
            a = self.arithmetic(op, a, b)?;
        }
    }

    fn arithmetic(&mut self, op: char, a: V, b: V) -> Result<V> {
        if a.ty.scalar != b.ty.scalar || !matches!(a.ty.scalar, Scalar::F32 | Scalar::I32) {
            return self
                .c
                .fail("arithmetic on values of different types (or booleans)");
        }
        let float = a.is(Scalar::F32);
        if !float && op == '/' {
            return self.c.fail("integer division");
        }
        if a.lanes() != b.lanes() && a.lanes() != 1 && b.lanes() != 1 {
            return self.c.fail(format!(
                "arithmetic on vectors of {} and {} lanes",
                a.lanes(),
                b.lanes()
            ));
        }
        let literal = match (a.literal, b.literal, op) {
            (Some(x), Some(y), '+') => x.checked_add(y),
            (Some(x), Some(y), '-') => x.checked_sub(y),
            (Some(x), Some(y), '*') => x.checked_mul(y),
            _ => None,
        };
        let binary = match (op, float) {
            ('+', true) => Binary::FAdd,
            ('-', true) => Binary::FSub,
            ('*', true) => Binary::FMul(MulZero::Ieee),
            ('/', _) => Binary::FDiv,
            ('+', false) => Binary::IAdd,
            ('-', false) => Binary::ISub,
            _ => Binary::IMul,
        };
        let (a, b) = self.widen(a, b);
        let v = self.binary(binary, a, b)?;
        Ok(V { literal, ..v })
    }

    fn unary_expression(&mut self) -> Result<V> {
        if self.c.accept("-") {
            let a = self.unary_expression()?;
            let negate = match a.ty.scalar {
                Scalar::F32 => Unary::FNeg,
                Scalar::I32 => Unary::INeg,
                _ => return self.c.fail("a negation of a boolean"),
            };
            let v = self.unary(negate, a)?;
            return Ok(V {
                literal: a.literal.and_then(i64::checked_neg),
                ..v
            });
        }
        if self.c.accept("!") {
            let a = self.unary_expression()?;
            if !a.is(Scalar::Bool) {
                return self.c.fail("! on something that is not a boolean");
            }
            return self.unary(Unary::Not, a);
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<V> {
        let mut v = self.primary()?;
        loop {
            if self.c.accept(".") {
                let w = self.c.ident()?;
                let Some(lanes) = swizzle_lanes(&w) else {
                    return self.c.fail(format!("the swizzle {w}"));
                };
                if lanes.iter().any(|l| *l >= v.lanes()) {
                    return self
                        .c
                        .fail(format!("the swizzle {w} of a value {} lanes wide", v.lanes()));
                }
                v = self.swizzle(v, &lanes)?;
            } else {
                return Ok(v);
            }
        }
    }

    fn primary(&mut self) -> Result<V> {
        match self.c.next()? {
            Token::Float(f) => Ok(self.f32(f)),
            Token::Int(i) => {
                let value = i32::try_from(i).map_err(|_| error(self.c.line(), "an integer over 32 bits"))?;
                Ok(self.i32(value))
            }
            Token::Punct("(") => {
                let v = self.expression()?;
                self.c.expect(")")?;
                Ok(v)
            }
            Token::Ident(name) => self.name(&name),
            Token::Punct(p) => self.c.fail(format!("{p} where a value was expected")),
        }
    }

    /// Refuse a read of the program's state before `main`: a constant's value is a constant expression.
    fn in_main(&self, what: &str) -> Result<()> {
        if self.in_main {
            Ok(())
        } else {
            self.c.fail(format!("{what} read in a constant's value"))
        }
    }

    fn name(&mut self, name: &str) -> Result<V> {
        if self.c.at("(") {
            return self.call(name);
        }
        if name == "gl_FragData" {
            return self.c.fail("gl_FragData read");
        }
        if matches!(
            name,
            "gl_FragCoord" | "gl_FrontFacing" | "gl_PointCoord" | "gl_FragDepth"
        ) {
            return self.c.fail(name);
        }
        let Some(sym) = self.symbols.get(name).cloned() else {
            return self.c.fail(format!("the name {name}"));
        };
        match sym {
            Sym::Var(var, ty) => {
                self.in_main(name)?;
                Ok(self.load(var, ty))
            }
            Sym::Const(v, _) | Sym::Loop(v) => Ok(v),
            Sym::Input(slot, ty) => {
                self.in_main(name)?;
                Ok(self.typed(ty, Op::LoadInput(slot, every(ty))))
            }
            Sym::Constants(rows) => {
                self.in_main(name)?;
                self.c.expect("[")?;
                let index = self.expression()?;
                self.c.expect("]")?;
                self.constant_row(rows, index)
            }
            Sym::Sampler(_) => self.c.fail(format!("the sampler {name} used as a value")),
            Sym::PosFixup => self
                .c
                .fail("posFixup used outside the clip-space fix MojoShader ends a vertex program with"),
        }
    }

    /// Row `index` of the uniform array as four floats; past the array (or before it), zero: D3D9's relative
    /// addressing.
    fn constant_row(&mut self, rows: u32, index: V) -> Result<V> {
        if index.ty != Type::I32 {
            return self
                .c
                .fail("a constant row indexed by something that is not an integer");
        }
        let address = match index.literal {
            Some(row) => match u32::try_from(row).ok().filter(|r| *r < rows) {
                Some(row) => Address::fixed(row.saturating_mul(16)),
                None => return self.c.fail(format!("constant row {row} of {rows}")),
            },
            None => Address {
                index: Some(index.v),
                stride: 16,
                offset: 0,
            },
        };
        Ok(self.typed(
            F32X4,
            Op::LoadBuffer {
                resource: Res::of(Self::constants_id()),
                address,
                oob: OutOfBounds::Zero,
            },
        ))
    }

    fn arguments(&mut self) -> Result<Vec<V>> {
        self.c.expect("(")?;
        let mut out = Vec::new();
        if self.c.accept(")") {
            return Ok(out);
        }
        loop {
            out.push(self.expression()?);
            if self.c.accept(")") {
                return Ok(out);
            }
            self.c.expect(",")?;
        }
    }

    /// `texture2D`, `textureCube` or `texture3D` of a sampler of that dimension at a coordinate of its lanes.
    fn texture(&mut self, function: &str) -> Result<V> {
        self.c.expect("(")?;
        let sampler = self.c.ident()?;
        self.c.expect(",")?;
        let coordinate = self.expression()?;
        self.c.expect(")")?;
        let Some(Sym::Sampler(index)) = self.symbols.get(&sampler).cloned() else {
            return self
                .c
                .fail(format!("{function} of {sampler}, which is not a sampler"));
        };
        self.in_main("a sampler")?;
        if self.stage != Stage::Pixel {
            return self.c.fail(format!("{function} in a vertex program"));
        }
        let dim = self.samplers.get(index).map_or(TextureDim::D2, |s| s.dim);
        let (lanes, expected) = sampling(dim);
        if expected != function {
            return self
                .c
                .fail(format!("{function} of {sampler}, which {expected} samples"));
        }
        if coordinate.ty != Type::vector(Scalar::F32, lanes) {
            return self
                .c
                .fail(format!("{function} at a coordinate that is not {lanes} floats"));
        }
        let (texture, sampler) = self.sampler_ids(index)?;
        Ok(self.typed(
            F32X4,
            Op::Sample {
                texture: Res::of(texture),
                sampler: Res::of(sampler),
                coord: coordinate.v,
                level: Level::Auto,
                compare: None,
                offset: [0; 3],
                clamp: None,
            },
        ))
    }

    fn call(&mut self, name: &str) -> Result<V> {
        if matches!(name, "texture2D" | "textureCube" | "texture3D") {
            return self.texture(name);
        }
        let args = self.arguments()?;
        match name {
            "vec2" | "vec3" | "vec4" | "ivec2" | "ivec3" | "ivec4" => self.construct(name, &args),
            "float" | "int" => {
                let [a] = args.as_slice() else {
                    return self.c.fail(format!("{name}() of {} arguments", args.len()));
                };
                if a.lanes() != 1 {
                    return self.c.fail(format!("{name}() of a vector"));
                }
                let to = if name == "float" { Scalar::F32 } else { Scalar::I32 };
                self.converted(*a, to)
            }
            "dFdx" | "dFdy" => {
                let [a] = args.as_slice() else {
                    return self.c.fail(format!("{name} of {} arguments", args.len()));
                };
                if self.stage != Stage::Pixel {
                    return self.c.fail(format!("{name} in a vertex program"));
                }
                if !a.is(Scalar::F32) {
                    return self.c.fail(format!("{name} of something that is not a float"));
                }
                let axis = if name == "dFdx" { Axis::X } else { Axis::Y };
                self.unary(Unary::Derivative(axis, Fineness::Any), *a)
            }
            "any" | "all" => {
                let [a] = args.as_slice() else {
                    return self.c.fail(format!("{name} of {} arguments", args.len()));
                };
                if !a.is(Scalar::Bool) {
                    return self.c.fail(format!("{name} of something that is not boolean"));
                }
                if a.lanes() == 1 {
                    return Ok(*a);
                }
                self.unary(if name == "any" { Unary::Any } else { Unary::All }, *a)
            }
            "lessThan" | "lessThanEqual" | "greaterThan" | "greaterThanEqual" | "equal" | "notEqual" => {
                let [a, b] = args.as_slice() else {
                    return self.c.fail(format!("{name} of {} arguments", args.len()));
                };
                if a.ty != b.ty || a.is(Scalar::Bool) {
                    return self
                        .c
                        .fail(format!("{name} of values that differ in type or width"));
                }
                let cmp = match name {
                    "lessThan" => Cmp::Lt,
                    "lessThanEqual" => Cmp::Le,
                    "greaterThan" => Cmp::Gt,
                    "greaterThanEqual" => Cmp::Ge,
                    "equal" => Cmp::Eq,
                    _ => Cmp::Ne,
                };
                self.comparison(cmp, *a, *b)
            }
            _ => self.builtin(name, &args),
        }
    }

    /// A built-in function of floats: its arguments checked, then its operation or its definition.
    fn builtin(&mut self, name: &str, args: &[V]) -> Result<V> {
        let Some((arity, mixes)) = builtin(name) else {
            return self.c.fail(format!("the function {name}"));
        };
        if args.len() != arity {
            return self.c.fail(format!("{name} of {} arguments", args.len()));
        }
        if args.iter().any(|a| !a.is(Scalar::F32)) {
            return self.c.fail(format!("{name} of something that is not a float"));
        }
        let widest = args.iter().map(V::lanes).max().unwrap_or(1);
        // GLSL's `min(vec, float)`, `clamp(vec, float, float)` and `mix(vec, vec, float)` take a scalar beside vectors
        let scalar_ok = |k: usize, a: &V| a.lanes() == 1 && mixes.contains(&k);
        if args
            .iter()
            .enumerate()
            .any(|(k, a)| a.lanes() != widest && !scalar_ok(k, a))
        {
            return self.c.fail(format!("{name} of vectors of different widths"));
        }
        let mut wide = Vec::new();
        for a in args {
            wide.push(self.splat(*a, widest));
        }
        let arg = |k: usize| {
            wide.get(k)
                .copied()
                .ok_or_else(|| error(0, "an argument missing"))
        };
        let x = arg(0)?;
        let unary = |u: Unary| Some(u);
        let simple = match name {
            "abs" => unary(Unary::FAbs),
            "fract" => unary(Unary::FFract),
            "sqrt" => unary(Unary::FSqrt),
            "sin" => unary(Unary::FSin),
            "cos" => unary(Unary::FCos),
            "exp2" => unary(Unary::FExp2),
            "floor" => unary(Unary::FRound(Rounding::Floor)),
            "ceil" => unary(Unary::FRound(Rounding::Ceil)),
            _ => None,
        };
        if let Some(u) = simple {
            return self.unary(u, x);
        }
        match name {
            "min" => self.binary(Binary::FMin(MinMaxNan::Any), x, arg(1)?),
            "max" => self.binary(Binary::FMax(MinMaxNan::Any), x, arg(1)?),
            "pow" => self.binary(Binary::FPow, x, arg(1)?),
            "sign" => self.sign(x),
            // MojoShader writes Direct3D 9's `rsq` and `log`, which take the operand's magnitude, as `inversesqrt` and
            // `log2` of the operand, which GLSL leaves undefined below zero
            "inversesqrt" | "log2" => {
                let magnitude = self.unary(Unary::FAbs, x)?;
                let u = if name == "log2" {
                    Unary::FLog2
                } else {
                    Unary::FInverseSqrt
                };
                self.unary(u, magnitude)
            }
            "clamp"
                if self.float_literal(arg(1)?) == Some(0.0f32.to_bits())
                    && self.float_literal(arg(2)?) == Some(1.0f32.to_bits()) =>
            {
                self.unary(Unary::Saturate(SaturateNan::Zero), x)
            }
            "clamp" => {
                let low = self.binary(Binary::FMax(MinMaxNan::Any), x, arg(1)?)?;
                self.binary(Binary::FMin(MinMaxNan::Any), low, arg(2)?)
            }
            "mix" => {
                let (y, a) = (arg(1)?, arg(2)?);
                let one = self.f32(1.0);
                let one = self.splat(one, widest);
                let inverse = self.binary(Binary::FSub, one, a)?;
                let left = self.binary(Binary::FMul(MulZero::Ieee), x, inverse)?;
                let right = self.binary(Binary::FMul(MulZero::Ieee), y, a)?;
                self.binary(Binary::FAdd, left, right)
            }
            "dot" => self.dot(x, arg(1)?),
            "length" => {
                let squared = self.dot(x, x)?;
                self.unary(Unary::FSqrt, squared)
            }
            "normalize" => {
                // as Direct3D 9's `nrm`, which MojoShader writes it for: a vector of length 0 is 0 (GLSL leaves it
                // undefined), each lane scaled by the largest float where the squared length is 0
                let squared = self.dot(x, x)?;
                let zero = self.f32(0.0);
                let empty = self.binary(Binary::FCmp(Cmp::Eq), squared, zero)?;
                let reciprocal = self.unary(Unary::FInverseSqrt, squared)?;
                let largest = self.f32(f32::MAX);
                let scale = self.op(Op::Select {
                    cond: empty.v,
                    accept: largest.v,
                    reject: reciprocal.v,
                })?;
                let scale = self.splat(scale, widest);
                self.binary(Binary::FMul(MulZero::Ieee), x, scale)
            }
            "cross" => {
                let y = arg(1)?;
                if widest != 3 {
                    return self.c.fail(format!("cross of {widest} lanes"));
                }
                let mut lanes = Vec::new();
                for (p, q) in [(1, 2), (2, 0), (0, 1)] {
                    let xp = self.lane(x, p)?;
                    let yq = self.lane(y, q)?;
                    let xq = self.lane(x, q)?;
                    let yp = self.lane(y, p)?;
                    let left = self.binary(Binary::FMul(MulZero::Ieee), xp, yq)?;
                    let right = self.binary(Binary::FMul(MulZero::Ieee), xq, yp)?;
                    lanes.push(self.binary(Binary::FSub, left, right)?.v);
                }
                Ok(self.typed(Type::vector(Scalar::F32, 3), Op::Compose(lanes)))
            }
            _ => self.c.fail(format!("the function {name}")),
        }
    }

    /// `v` as a value of `to`, as GLSL's constructors convert: a float to an integer truncating, a boolean to 1 where it
    /// holds and 0 where it does not.
    fn converted(&mut self, v: V, to: Scalar) -> Result<V> {
        if v.is(to) {
            return Ok(v);
        }
        if !v.is(Scalar::Bool) {
            return Ok(self.convert(v, to));
        }
        let (one, zero) = if to == Scalar::F32 {
            (self.f32(1.0), self.f32(0.0))
        } else {
            (self.i32(1), self.i32(0))
        };
        let (one, zero) = (self.splat(one, v.lanes()), self.splat(zero, v.lanes()));
        self.op(Op::Select {
            cond: v.v,
            accept: one.v,
            reject: zero.v,
        })
    }

    /// `sign(x)`: -1 below zero, 0 at either zero, 1 otherwise. GLSL's `sign` is 0.0 where `x` is 0, and MojoShader
    /// writes it for Direct3D 9's `sgn`, whose reference compares (a NaN is 1, as `iridia-d3d9-ir` lifts `sgn`); IR's
    /// `FSign` gives a signed zero back as itself.
    fn sign(&mut self, x: V) -> Result<V> {
        let lanes = x.lanes();
        let (zero, one, minus) = (self.f32(0.0), self.f32(1.0), self.f32(-1.0));
        let (zero, one, minus) = (
            self.splat(zero, lanes),
            self.splat(one, lanes),
            self.splat(minus, lanes),
        );
        let below = self.binary(Binary::FCmp(Cmp::Lt), x, zero)?;
        let at_zero = self.binary(Binary::FCmp(Cmp::Eq), x, zero)?;
        let rest = self.op(Op::Select {
            cond: at_zero.v,
            accept: zero.v,
            reject: one.v,
        })?;
        self.op(Op::Select {
            cond: below.v,
            accept: minus.v,
            reject: rest.v,
        })
    }

    /// The bits of the float every lane of `v` holds, where it is a literal or a literal repeated.
    fn float_literal(&self, v: V) -> Option<u32> {
        let values = &self.b.function().values;
        let mut id = v.v;
        loop {
            let def = values.get(usize::try_from(id.0).ok()?)?;
            match &def.op {
                Op::Splat(x) => id = *x,
                Op::Literal(parts) if def.ty.scalar == Scalar::F32 => {
                    let lanes = parts.get(..usize::from(def.ty.lanes))?;
                    let first = *lanes.first()?;
                    return lanes
                        .iter()
                        .all(|l| *l == first)
                        .then(|| u32::try_from(first).ok())
                        .flatten();
                }
                _ => return None,
            }
        }
    }

    /// The dot product of two floats of one width: of scalars, their product.
    fn dot(&mut self, a: V, b: V) -> Result<V> {
        if a.lanes() == 1 {
            return self.binary(Binary::FMul(MulZero::Ieee), a, b);
        }
        self.op(Op::Dot(a.v, b.v))
    }

    /// `vec4(..)`, `ivec4(..)`: the lanes of the arguments in order, or one scalar splat; a float converted to an
    /// integer, as GLSL's constructor does (truncating).
    fn construct(&mut self, name: &str, args: &[V]) -> Result<V> {
        let width = match name.chars().last().and_then(|c| c.to_digit(10)) {
            Some(lanes) => u8::try_from(lanes)
                .ok()
                .filter(|l| (2..=4).contains(l))
                .ok_or_else(|| error(self.c.line(), format!("{name}() of {lanes} lanes")))?,
            None => 4,
        };
        let to = if name.starts_with('i') {
            Scalar::I32
        } else {
            Scalar::F32
        };
        let mut lanes = Vec::new();
        for a in args {
            let v = self.converted(*a, to)?;
            for k in 0..v.lanes() {
                lanes.push(self.lane(v, k)?);
            }
        }
        match lanes[..] {
            [one] => Ok(self.splat(one, width)),
            _ if lanes.len() == usize::from(width) => {
                let parts = lanes.iter().map(|l| l.v).collect();
                Ok(self.typed(Type::vector(to, width), Op::Compose(parts)))
            }
            _ => self.c.fail(format!("{name}() of {} lanes", lanes.len())),
        }
    }

    // ---- the program ----

    fn finish(mut self, source: &str) -> Result<Program> {
        self.b.at(None);
        // the resources: the constants, then each sampler's texture and sampler
        let mut resources = Vec::new();
        let mut notes = std::mem::take(&mut self.notes);
        if let Some(rows) = self.rows {
            let size = rows.saturating_mul(16);
            let mut constants = Vec::new();
            let (mut left_out, mut cut) = (0usize, 0usize);
            for c in &self.hints.constants {
                // the program declares the rows it reads: a constant beyond them is not its, one that reaches past them
                // is cut to the rows it has
                if c.offset >= size {
                    left_out = left_out.saturating_add(1);
                    continue;
                }
                cut = cut.saturating_add(usize::from(c.offset.saturating_add(c.bytes) > size));
                let bytes = c.bytes.min(size.saturating_sub(c.offset));
                let ty = if bytes == 16 {
                    if c.width <= 1 {
                        ConstantType::Scalar(ScalarType::Float)
                    } else {
                        ConstantType::Vector(float_value(c.width.min(4)))
                    }
                } else {
                    ConstantType::Array {
                        element: Box::new(ConstantType::Vector(float_value(4))),
                        count: (bytes / 16).max(1),
                        stride: 16,
                    }
                };
                constants.push(LayoutConstant {
                    name: c.name.clone(),
                    offset: c.offset,
                    ty,
                    default: None,
                });
            }
            if left_out > 0 {
                notes.push(format!(
                    "{left_out} constants the effect names lie past the program's {rows} uniform rows and are left out"
                ));
            }
            if cut > 0 {
                notes.push(format!(
                    "{cut} constants the effect names reach past the program's {rows} uniform rows and are cut to them"
                ));
            }
            resources.push(Resource {
                id: Self::constants_id(),
                kind: ResourceKind::ConstantBuffer {
                    layout: BufferLayout { size, constants },
                },
                count: 1,
                provenance: ResourceProvenance {
                    name: if self.stage == Stage::Vertex {
                        "vs_uniforms_vec4"
                    } else {
                        "ps_uniforms_vec4"
                    }
                    .into(),
                    d3d: Some(D3dBinding {
                        class: D3dClass::ConstantBuffer,
                        register: 0,
                        space: 0,
                    }),
                    counter_of: None,
                    layout: None,
                },
            });
        }
        for (index, s) in self.samplers.iter().enumerate() {
            let (texture, sampler) = self.sampler_ids(index)?;
            resources.push(Resource {
                id: texture,
                kind: ResourceKind::Texture {
                    dim: s.dim,
                    arrayed: false,
                    multisampled: false,
                    sample: SampleType::Float,
                },
                count: 1,
                provenance: ResourceProvenance {
                    name: s.name.clone(),
                    d3d: Some(D3dBinding {
                        class: D3dClass::ShaderResource,
                        register: s.slot,
                        space: 0,
                    }),
                    counter_of: None,
                    layout: None,
                },
            });
            resources.push(Resource {
                id: sampler,
                kind: ResourceKind::Sampler { comparison: false },
                count: 1,
                provenance: ResourceProvenance {
                    name: format!("{} ({})", s.name, s.glsl_name),
                    d3d: Some(D3dBinding {
                        class: D3dClass::Sampler,
                        register: s.slot,
                        space: 0,
                    }),
                    counter_of: None,
                    layout: None,
                },
            });
        }
        // the outputs, from their variables, in the interface's order
        for (slot, (v, var)) in (0u32..).zip(&self.outputs) {
            let ty = Type::vector(Scalar::F32, v.value.width);
            let value = self.b.value(ty, Op::LoadVar(*var));
            self.b.store_output(slot, every(ty), value);
        }
        let entry = self.b.finish(End::Return).map_err(|e| error(0, e))?;
        notes.sort();
        notes.dedup();
        let program = Program {
            stage: self.stage,
            entry,
            patch: None,
            workgroup: None,
            interface: Interface {
                stage: self.stage,
                inputs: self.inputs,
                outputs: self.outputs.into_iter().map(|(v, _)| v).collect(),
                resources,
                facts: StageFacts::None,
            },
            conventions: Conventions::D3D,
            float: FloatEnv::default(),
            provenance: Provenance {
                format: ShaderFormat::Glsl,
                target: self.hints.target.clone(),
                lifter: LIFTER.into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(source_hash(source.as_bytes())),
                instruction_count: None,
                policies: POLICIES.iter().map(|p| (*p).into()).collect(),
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

/// The built-in float functions of the dialect: their number of arguments, and which of them may be a scalar beside
/// vectors (GLSL's `genType f(genType, float)` forms).
fn builtin(name: &str) -> Option<(usize, &'static [usize])> {
    Some(match name {
        "abs" | "fract" | "inversesqrt" | "sqrt" | "sin" | "cos" | "exp2" | "log2" | "normalize" | "sign"
        | "floor" | "ceil" | "length" => (1, &[]),
        "dot" | "pow" | "cross" => (2, &[]),
        "min" | "max" => (2, &[1]),
        "clamp" => (3, &[1, 2]),
        "mix" => (3, &[2]),
        _ => return None,
    })
}
