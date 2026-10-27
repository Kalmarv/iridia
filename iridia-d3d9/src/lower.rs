use std::{collections::BTreeMap, num::NonZeroU32};

use iridia::{
    BufferLayout, Constant as Field, ConstantType, Conventions, D3dBinding, D3dClass, D3dModes, D3dSignature,
    Interface, InterpMode, InterpSampling, Interpolation, Label, Majorness, Program, Provenance, Resource,
    ResourceId, ResourceKind, ResourceProvenance, SampleType, ScalarType, ShaderFormat, Stage, StageFacts,
    SystemValue, TextureDim, Value, Varying, VaryingProvenance,
    build::{Builder, Kind, Val},
    entry_io,
};
use naga::{
    AddressSpace, ArraySize, BinaryOperator as Bo, Expression as E, GlobalVariable, Handle, ImageClass,
    ImageDimension, LocalVariable, MathFunction as Mf, RelationalFunction, ResourceBinding, SampleLevel,
    ScalarKind, Statement, TypeInner, UnaryOperator,
};

use crate::{Destination, Instruction, LIFTER_VERSION, Opcode, Register, Result, Shader, Source, error};

const SEMANTICS: [&str; 14] = [
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE",
];

struct Lower<'a> {
    source: &'a Shader,
    b: Builder,
    interface: Interface,
    bindings: Vec<(ResourceId, Handle<GlobalVariable>)>,
    constants: Option<Handle<GlobalVariable>>,
    samplers: BTreeMap<u32, (Handle<GlobalVariable>, Handle<GlobalVariable>, TextureDim)>,
    locals: BTreeMap<Register, (Handle<LocalVariable>, u8)>,
    inputs: BTreeMap<u32, (Val, u8, usize)>,
    definitions: BTreeMap<u32, Val>,
    notes: Vec<String>,
    at: usize,
}

fn value(width: u8) -> Value {
    Value {
        scalar: ScalarType::Float,
        width,
    }
}
/// How many lanes a write mask names.
fn lanes_in(mask: u8) -> u8 {
    u8::try_from(mask.count_ones()).unwrap_or(4)
}
/// The semantic name of a declaration usage.
fn semantic(usage: u8, at: usize) -> Result<&'static str> {
    SEMANTICS.get(usize::from(usage)).copied().ok_or_else(|| {
        error(
            at,
            format!(
                "declaration usage {usage}, where D3DDECLUSAGE is 0 to {}",
                SEMANTICS.len().saturating_sub(1)
            ),
        )
    })
}
/// The index of an entry point argument.
fn argument(index: usize, at: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| error(at, "more than 4294967295 inputs"))
}
fn varying(name: &str, index: u32, mask: u8, register: u32, system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: value(lanes_in(mask)),
        interp: None,
        used: false,
        provenance: VaryingProvenance {
            d3d: Some(D3dSignature {
                register,
                mask,
                read_write_mask: 0,
            }),
        },
    }
}

/// The bank's register layout is exact even for an optimized partial array. CTAB's unshortened type is retained
/// by `read`; the interface describes only the allocation that the compiled shader actually binds.
fn field(c: &crate::Constant) -> Result<Field> {
    let t = &c.ty;
    if c.register_set != 2
        || t.kind != 3
        || !t.members.is_empty()
        || !(1..=4).contains(&t.rows)
        || !(1..=4).contains(&t.columns)
        || t.elements == 0
    {
        return Err(error(
            0,
            format!("CTAB {}: non-float/structured/empty type {:?}", c.name, t),
        ));
    }
    // rows and columns are 1 to 4, checked above
    let lanes = |n: u16| u8::try_from(n).map_err(|_| error(0, format!("CTAB {}: {n} lanes", c.name)));
    let element = match t.class {
        0 if t.rows == 1 && t.columns == 1 => ConstantType::Scalar(ScalarType::Float),
        1 if t.rows == 1 => ConstantType::Vector(value(lanes(t.columns)?)),
        2 | 3 => ConstantType::Matrix {
            scalar: ScalarType::Float,
            rows: lanes(t.rows)?,
            columns: lanes(t.columns)?,
            major: if t.class == 2 {
                Majorness::Row
            } else {
                Majorness::Column
            },
            stride: 16,
        },
        _ => return Err(error(0, format!("CTAB {}: parameter class {}", c.name, t.class))),
    };
    let registers = if t.class == 2 {
        t.rows
    } else if t.class == 3 {
        t.columns
    } else {
        1
    };
    let Some(count) = c
        .count
        .checked_div(registers)
        .filter(|&n| c.count.is_multiple_of(registers) && n <= t.elements)
    else {
        return Err(error(
            0,
            format!(
                "CTAB {}: an allocation of {} registers is not a whole number of the type's elements ({} registers each, at most {})",
                c.name, c.count, registers, t.elements
            ),
        ));
    };
    let ty = if t.elements > 1 {
        ConstantType::Array {
            element: Box::new(element),
            count: count.into(),
            stride: u32::from(registers).saturating_mul(16),
        }
    } else {
        element
    };
    let default_size = ty
        .size()
        .ok_or_else(|| error(0, "CTAB normalized type size overflow"))? as usize;
    let default = c
        .default
        .as_ref()
        .map(|bytes| {
            bytes.get(..default_size).map(<[u8]>::to_vec).ok_or_else(|| {
                error(
                    0,
                    format!(
                        "CTAB {}: a default of {} bytes, where its type takes {default_size}",
                        c.name,
                        bytes.len()
                    ),
                )
            })
        })
        .transpose()?;
    Ok(Field {
        name: c.name.clone(),
        offset: u32::from(c.register).saturating_mul(16),
        ty,
        default,
    })
}

pub(crate) fn lift(source: &Shader, bytes: &[u8]) -> Result<Program> {
    lower(source, bytes, None, false)
}

pub(crate) fn project(source: &Shader, bytes: &[u8], cut: crate::Projection, tail: bool) -> Result<Program> {
    if source.stage != Stage::Pixel {
        return Err(error(cut.after_word, "register RGB projections are pixel-only"));
    }
    if cut.lanes.iter().any(|&n| n > 3)
        || cut.lanes[0] == cut.lanes[1]
        || cut.lanes[0] == cut.lanes[2]
        || cut.lanes[1] == cut.lanes[2]
        || !source.instructions.iter().any(|i| i.word == cut.after_word)
    {
        return Err(error(
            cut.after_word,
            "invalid projection lanes or instruction boundary",
        ));
    }
    if tail && source.inputs.iter().any(|i| i.register == 31) {
        return Err(error(
            cut.after_word,
            "projection input register v31 already occupied",
        ));
    }
    let mut program = lower(source, bytes, Some((cut, tail)), false)?;
    // A derived value and its original program share the source hash, but are distinct cache artifacts.
    program.provenance.lifter = format!(
        "iridia-d3d9/{}:word={}:r{}:{:?}",
        if tail { "reencode" } else { "projection" },
        cut.after_word,
        cut.temporary,
        cut.lanes
    );
    Ok(program)
}

/// A sliced vertex output runs as an invocation program. Unselected RGBA lanes are placeholders (0,0,0,1).
pub(crate) fn vertex_output(source: &Shader, bytes: &[u8]) -> Result<Program> {
    lower(source, bytes, None, true)
}

fn lower(
    source: &Shader,
    bytes: &[u8],
    projection: Option<(crate::Projection, bool)>,
    pad_output: bool,
) -> Result<Program> {
    let mut l = Lower {
        source,
        b: Builder::new(),
        interface: Interface {
            stage: source.stage,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: Vec::new(),
            facts: StageFacts::None,
        },
        bindings: Vec::new(),
        constants: None,
        samplers: BTreeMap::new(),
        locals: BTreeMap::new(),
        inputs: BTreeMap::new(),
        definitions: BTreeMap::new(),
        notes: Vec::new(),
        at: 0,
    };
    l.prepare()?;
    for i in &source.instructions {
        l.at = i.word;
        l.instruction(i)?;
        if let Some((cut, tail)) = projection
            && i.word == cut.after_word
        {
            let mask = cut.lanes.iter().fold(0, |m, n| m | (1 << n));
            let value = l.register(Register::Temporary(cut.temporary), mask)?;
            l.notes.push(format!(
                "derived register projection after word {}: r{} lanes {:?}; {}",
                cut.after_word,
                cut.temporary,
                cut.lanes,
                if tail {
                    "original tail with replacement input"
                } else {
                    "prefix only; alpha one"
                }
            ));
            if tail {
                let index = l.interface.inputs.len();
                let mut input = varying("DERIVED_RGB", 0, 7, 31, None);
                input.used = true;
                l.interface.inputs.push(input);
                let arg = Val {
                    expr: l.b.app(E::FunctionArgument(argument(index, cut.after_word)?)),
                    kind: Kind::F,
                    width: 3,
                };
                let mut parts = (0..4).map(|n| l.b.lane(value, n)).collect::<Vec<_>>();
                for (n, &lane) in (0u8..).zip(&cut.lanes) {
                    let part = parts.get_mut(usize::from(lane)).ok_or_else(|| {
                        error(
                            cut.after_word,
                            format!("projection lane {lane}, where a register has lanes 0 to 3"),
                        )
                    })?;
                    *part = l.b.lane(arg, n);
                }
                let replaced = l.b.compose(&parts);
                l.write(
                    Destination {
                        register: Register::Temporary(cut.temporary),
                        mask,
                        saturate: false,
                        partial_precision: false,
                    },
                    replaced,
                );
            } else {
                let mut parts = cut.lanes.map(|n| l.b.lane(value, n)).to_vec();
                parts.push(l.b.lit_f(1.));
                let rgb = l.b.compose(&parts);
                l.locals.retain(|r, _| !matches!(r, Register::Target(_)));
                l.write(
                    Destination {
                        register: Register::Target(0),
                        mask: 15,
                        saturate: false,
                        partial_precision: false,
                    },
                    rgb,
                );
                break;
            }
        }
        l.b.check().map_err(|e| error(i.word, e))?;
    }
    if pad_output {
        let &(_, mask) = l
            .locals
            .get(&Register::Target(0))
            .ok_or_else(|| error(0, "projected output has no writes"))?;
        if mask != 15 {
            let zero = l.b.lit_f(0.);
            let one = l.b.lit_f(1.);
            let padding = l.b.compose(&[zero, zero, zero, one]);
            l.write(
                Destination {
                    register: Register::Target(0),
                    mask: 15 & !mask,
                    saturate: false,
                    partial_precision: false,
                },
                padding,
            );
        }
    }
    l.finish(bytes)
}

impl Lower<'_> {
    fn resource(
        &mut self,
        kind: ResourceKind,
        name: String,
        class: D3dClass,
        register: u32,
        global: Handle<GlobalVariable>,
    ) -> Result<()> {
        let id = ResourceId(
            u32::try_from(self.interface.resources.len())
                .map_err(|_| error(self.at, "more than 4294967295 resources"))?,
        );
        self.interface.resources.push(Resource {
            id,
            kind,
            count: 1,
            provenance: ResourceProvenance {
                name,
                d3d: Some(D3dBinding {
                    class,
                    register,
                    space: 0,
                }),
                counter_of: None,
                layout: None,
            },
        });
        self.bindings.push((id, global));
        Ok(())
    }
    fn global(
        &mut self,
        name: String,
        ty: Handle<naga::Type>,
        group: u32,
        binding: u32,
        space: AddressSpace,
    ) -> Handle<GlobalVariable> {
        self.b.global(GlobalVariable {
            name: Some(name),
            space,
            binding: Some(ResourceBinding { group, binding }),
            ty,
            init: None,
            memory_decorations: Default::default(),
        })
    }
    fn prepare(&mut self) -> Result<()> {
        for i in &self.source.inputs {
            let mut v = varying(
                semantic(i.usage, self.at)?,
                i.index.into(),
                i.mask,
                i.register,
                None,
            );
            v.interp = (self.source.stage == Stage::Pixel).then_some(Interpolation {
                mode: InterpMode::Perspective,
                sampling: if i.centroid {
                    InterpSampling::Centroid
                } else {
                    InterpSampling::Center
                },
            });
            if i.partial_precision {
                self.notes
                    .push("partial-precision requests computed at f32".into());
            }
            let index = self.interface.inputs.len();
            let arg = Val {
                expr: self.b.app(E::FunctionArgument(argument(index, self.at)?)),
                kind: Kind::F,
                width: v.value.width,
            };
            let zero = self.b.lit_f(0.0);
            let mut parts = [zero; 4];
            let mut lane = 0;
            for (n, p) in parts.iter_mut().enumerate() {
                if i.mask & (1 << n) != 0 {
                    *p = self.b.lane(arg, lane);
                    lane = lane.saturating_add(1);
                }
            }
            let expanded = self.b.compose(&parts);
            self.inputs.insert(i.register, (expanded, i.mask, index));
            self.interface.inputs.push(v);
        }
        for (&n, words) in &self.source.definitions {
            let parts: Vec<_> = words.iter().map(|w| self.b.lit_f(f32::from_bits(*w))).collect();
            self.definitions.insert(n, self.b.compose(&parts));
        }
        let mut rows = 0;
        for i in &self.source.instructions {
            for s in &i.sources {
                if let Register::Constant(n) = s.register
                    && !self.source.definitions.contains_key(&n)
                {
                    rows = rows.max(n.saturating_add(1));
                }
            }
        }
        let mut fields = Vec::new();
        if let Some(ctab) = &self.source.constants {
            for c in &ctab.constants {
                match c.register_set {
                    2 => {
                        rows = rows.max(u32::from(c.register).saturating_add(u32::from(c.count)));
                        fields.push(field(c)?);
                        let t = &c.ty;
                        let lines = if t.class == 2 {
                            t.rows
                        } else if t.class == 3 {
                            t.columns
                        } else {
                            1
                        };
                        if u32::from(c.count) < u32::from(t.elements).saturating_mul(u32::from(lines)) {
                            self.notes.push(format!("CTAB {} retains its declared type; its optimized allocation binds {} registers",c.name,c.count));
                        }
                    }
                    3 => {
                        if c.count != 1 || c.ty.class != 4 || !matches!(c.ty.kind, 12..=14) {
                            return Err(error(
                                0,
                                format!("CTAB {}: sampler array or type {:?}", c.name, c.ty),
                            ));
                        }
                        let expected = match c.ty.kind {
                            12 => TextureDim::D2,
                            13 => TextureDim::D3,
                            _ => TextureDim::Cube,
                        };
                        if self.source.samplers.get(&u32::from(c.register)) != Some(&expected) {
                            return Err(error(0, format!("CTAB {} sampler disagrees with DCL", c.name)));
                        }
                    }
                    _ => {
                        return Err(error(
                            0,
                            format!("CTAB {}: register set {} not lifted", c.name, c.register_set),
                        ));
                    }
                }
            }
        }
        if let Some(rows) = NonZeroU32::new(rows) {
            let u4 = self.b.ty_of(Kind::U, 4);
            let ty = self.b.ty(TypeInner::Array {
                base: u4,
                size: ArraySize::Constant(rows),
                stride: 16,
            });
            let g = self.global("cb0".into(), ty, 0, 0, AddressSpace::Uniform);
            self.constants = Some(g);
            self.resource(
                ResourceKind::ConstantBuffer {
                    layout: BufferLayout {
                        size: rows.get().saturating_mul(16),
                        constants: fields,
                    },
                },
                "ps_float_constants".into(),
                D3dClass::ConstantBuffer,
                0,
                g,
            )?;
        }
        for (&slot, &dim) in &self.source.samplers {
            let naga_dim = match dim {
                TextureDim::D2 => ImageDimension::D2,
                TextureDim::D3 => ImageDimension::D3,
                TextureDim::Cube => ImageDimension::Cube,
                _ => return Err(error(0, "1D sampler")),
            };
            let ty = self.b.ty(TypeInner::Image {
                dim: naga_dim,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            });
            let texture = self.global(format!("t{slot}"), ty, 1, slot, AddressSpace::Handle);
            let ty = self.b.ty(TypeInner::Sampler { comparison: false });
            let sampler = self.global(format!("s{slot}"), ty, 2, slot, AddressSpace::Handle);
            let name = self
                .source
                .constants
                .as_ref()
                .and_then(|c| {
                    c.constants
                        .iter()
                        .find(|c| c.register_set == 3 && u32::from(c.register) == slot)
                })
                .map_or_else(|| format!("s{slot}"), |c| c.name.clone());
            self.resource(
                ResourceKind::Texture {
                    dim,
                    arrayed: false,
                    multisampled: false,
                    sample: SampleType::Float,
                },
                name.clone(),
                D3dClass::ShaderResource,
                slot,
                texture,
            )?;
            self.resource(
                ResourceKind::Sampler { comparison: false },
                name,
                D3dClass::Sampler,
                slot,
                sampler,
            )?;
            self.samplers.insert(slot, (texture, sampler, dim));
        }
        Ok(())
    }
    fn register(&mut self, r: Register, required: u8) -> Result<Val> {
        match r {
            Register::Input(n) => {
                let &(v, mask, index) = self
                    .inputs
                    .get(&n)
                    .ok_or_else(|| error(self.at, format!("undeclared input v{n}")))?;
                if required & !mask != 0 {
                    return Err(error(
                        self.at,
                        format!("v{n}: read of undeclared lanes {required:#x}"),
                    ));
                }
                let input = self.interface.inputs.get_mut(index).ok_or_else(|| {
                    error(
                        self.at,
                        format!("input v{n} is input {index} of a list that is shorter"),
                    )
                })?;
                input.used = true;
                if let Some(d) = &mut input.provenance.d3d {
                    d.read_write_mask |= required;
                }
                Ok(v)
            }
            Register::Constant(n) => {
                if let Some(&v) = self.definitions.get(&n) {
                    return Ok(v);
                }
                let g = self
                    .constants
                    .ok_or_else(|| error(self.at, "constant bank not allocated"))?;
                let base = self.b.global_ptr(g);
                let p = self.b.app(E::AccessIndex { base, index: n });
                let v = self.b.load(p, Kind::U, 4);
                Ok(self.b.bitcast(v, Kind::F))
            }
            Register::Temporary(_) => {
                let &(local, mask) = self
                    .locals
                    .get(&r)
                    .ok_or_else(|| error(self.at, format!("uninitialized {r:?}")))?;
                if required & !mask != 0 {
                    return Err(error(
                        self.at,
                        format!("{r:?}: uninitialized lanes {:#x}", required & !mask),
                    ));
                }
                let p = self.b.local_ptr(local);
                Ok(self.b.load(p, Kind::F, 4))
            }
            _ => Err(error(self.at, format!("not a readable float register: {r:?}"))),
        }
    }
    fn source(&mut self, s: Source, needed: u8) -> Result<Val> {
        let required = s.read_mask(needed);
        let mut v = self.register(s.register, required)?;
        v = self.b.swizzle(v, &s.swizzle);
        if s.absolute {
            v = self.b.math(Mf::Abs, &[v]);
        }
        if s.negate {
            v = self.b.unary(UnaryOperator::Negate, v);
        }
        Ok(v)
    }
    fn write(&mut self, d: Destination, mut v: Val) {
        if d.partial_precision {
            self.notes
                .push("partial-precision requests computed at f32".into());
        }
        if d.saturate {
            v = self.b.d3d_saturate(v);
        }
        let (local, mask) = *self.locals.entry(d.register).or_insert_with(|| {
            let ty = self.b.ty_of(Kind::F, 4);
            (self.b.local(&format!("{:?}", d.register), ty), 0)
        });
        let p = self.b.local_ptr(local);
        if d.mask == 15 {
            self.b.store(p, v);
        } else {
            for n in 0u8..4 {
                if d.mask & (1 << n) != 0 {
                    let ptr = self.b.app(E::AccessIndex {
                        base: p,
                        index: u32::from(n),
                    });
                    let lane = self.b.lane(v, n);
                    self.b.store(ptr, lane);
                }
            }
        }
        self.locals.insert(d.register, (local, mask | d.mask));
    }
    fn sample(&mut self, i: &Instruction) -> Result<Val> {
        let s = i.source(1)?;
        let Register::Sampler(slot) = s.register else {
            return Err(error(self.at, "TEX source 1 is not a sampler"));
        };
        if s.absolute || s.negate || s.swizzle != [0, 1, 2, 3] {
            return Err(error(self.at, "modified/swizzled TEX sampler"));
        }
        let &(texture, sampler, dim) = self
            .samplers
            .get(&slot)
            .ok_or_else(|| error(self.at, format!("undeclared sampler s{slot}")))?;
        let lanes: &[u8] = if dim == TextureDim::D2 {
            &[0, 1]
        } else {
            &[0, 1, 2]
        };
        let mut required = if dim == TextureDim::D2 { 3 } else { 7 };
        if i.control != 0 {
            required |= 8;
        }
        let coordinates = self.source(i.source(0)?, required)?;
        let mut coord = self.b.swizzle(coordinates, lanes);
        let level = match i.control {
            0 => SampleLevel::Auto,
            1 => {
                let w = self.b.lane(coordinates, 3);
                coord = self.b.binary(Bo::Divide, coord, w);
                SampleLevel::Auto
            }
            2 => SampleLevel::Bias(self.b.lane(coordinates, 3).expr),
            _ => return Err(error(self.at, "TEX control 3 (projection and bias together)")),
        };
        let image = self.b.global_ptr(texture);
        let sampler = self.b.global_ptr(sampler);
        Ok(Val {
            expr: self.b.app(E::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate: coord.expr,
                array_index: None,
                offset: None,
                level,
                depth_ref: None,
                clamp_to_edge: false,
            }),
            kind: Kind::F,
            width: 4,
        })
    }
    fn instruction(&mut self, i: &Instruction) -> Result<()> {
        use Opcode::*;
        if i.opcode == Nop {
            return Ok(());
        }
        let d = i
            .destination
            .ok_or_else(|| error(self.at, "missing destination"))?;
        if matches!(i.opcode, Rcp | Rsq | Exp | Log | Pow)
            && i.sources
                .iter()
                .any(|s| !s.swizzle.iter().all(|&n| n == s.swizzle[0]))
        {
            return Err(error(
                self.at,
                format!("{:?} requires replicate scalar swizzles", i.opcode),
            ));
        }
        if i.opcode == Nrm
            && (!matches!(d.register, Register::Temporary(_)) || i.source(0)?.register == d.register)
        {
            return Err(error(
                self.at,
                "NRM requires a temporary destination distinct from its source",
            ));
        }
        if i.opcode == TexKill {
            // SM2/3 use all four lanes (the SDK's early texkill page describes SM1's xyz only).
            let v = self.register(d.register, 15)?;
            let zero = self.b.lit_f(0.0);
            let negative = self.b.binary(Bo::Less, v, zero);
            let any = Val {
                expr: self.b.app(E::Relational {
                    fun: RelationalFunction::Any,
                    argument: negative.expr,
                }),
                kind: Kind::B,
                width: 1,
            };
            self.b.when(any, |b| b.push(Statement::Kill));
            return Ok(());
        }
        if i.opcode == Tex {
            let v = self.sample(i)?;
            self.write(d, v);
            return Ok(());
        }
        let mut args = Vec::new();
        for (n, &s) in i.sources.iter().enumerate() {
            args.push(self.source(s, i.source_mask(n))?);
        }
        let arg = |n: usize| {
            args.get(n).copied().ok_or_else(|| {
                error(
                    i.word,
                    format!("{:?} reads source {n}, where it has {}", i.opcode, args.len()),
                )
            })
        };
        let a = arg(0)?;
        let v = match i.opcode {
            Mov => a,
            Add | Sub | Mul => self.b.binary(
                match i.opcode {
                    Add => Bo::Add,
                    Sub => Bo::Subtract,
                    _ => Bo::Multiply,
                },
                a,
                arg(1)?,
            ),
            Mad => {
                let product = self.b.binary(Bo::Multiply, a, arg(1)?);
                self.b.binary(Bo::Add, product, arg(2)?)
            }
            Abs => self.b.math(Mf::Abs, &[a]),
            Min | Max => self
                .b
                .d3d_min_max(if i.opcode == Min { Mf::Min } else { Mf::Max }, a, arg(1)?),
            Frc => self.b.math(Mf::Fract, &[a]),
            Cmp => {
                let zero = self.b.lit_f(0.0);
                let cond = self.b.binary(Bo::GreaterEqual, a, zero);
                self.b.select(cond, arg(1)?, arg(2)?)
            }
            Lrp => {
                let one = self.b.lit_f(1.0);
                let inverse = self.b.binary(Bo::Subtract, one, a);
                let left = self.b.binary(Bo::Multiply, a, arg(1)?);
                let right = self.b.binary(Bo::Multiply, inverse, arg(2)?);
                self.b.binary(Bo::Add, left, right)
            }
            Dp3 | Dp4 | Dp2Add => {
                let lanes: &[u8] = if i.opcode == Dp3 {
                    &[0, 1, 2]
                } else if i.opcode == Dp4 {
                    &[0, 1, 2, 3]
                } else {
                    &[0, 1]
                };
                let x = self.b.swizzle(a, lanes);
                let y = self.b.swizzle(arg(1)?, lanes);
                let mut dot = self.b.math(Mf::Dot, &[x, y]);
                if i.opcode == Dp2Add {
                    let z = self.b.lane(arg(2)?, 0);
                    dot = self.b.binary(Bo::Add, dot, z);
                }
                self.b.splat(dot, 4)
            }
            Nrm => {
                let xyz = self.b.swizzle(a, &[0, 1, 2]);
                let squared = self.b.math(Mf::Dot, &[xyz, xyz]);
                let zero = self.b.lit_f(0.0);
                let empty = self.b.binary(Bo::Equal, squared, zero);
                let reciprocal = self.b.math(Mf::InverseSqrt, &[squared]);
                let largest = self.b.lit_f(f32::MAX);
                let scale = self.b.select(empty, largest, reciprocal);
                self.b.binary(Bo::Multiply, a, scale)
            }
            Rcp | Rsq | Exp | Log | Pow => {
                let mut x = self.b.lane(a, 0);
                if matches!(i.opcode, Rsq | Log | Pow) {
                    x = self.b.math(Mf::Abs, &[x]);
                }
                let scalar = match i.opcode {
                    Rcp => {
                        let one = self.b.lit_f(1.0);
                        self.b.binary(Bo::Divide, one, x)
                    }
                    Rsq => self.b.math(Mf::InverseSqrt, &[x]),
                    Exp => self.b.math(Mf::Exp2, &[x]),
                    Log => self.b.math(Mf::Log2, &[x]),
                    _ => {
                        let y = self.b.lane(arg(1)?, 0);
                        self.b.math(Mf::Pow, &[x, y])
                    }
                };
                self.b.splat(scalar, 4)
            }
            _ => {
                return Err(error(
                    self.at,
                    format!("operation {:?} has no lowering", i.opcode),
                ));
            }
        };
        self.write(d, v);
        Ok(())
    }
    fn finish(mut self, bytes: &[u8]) -> Result<Program> {
        let mut outputs = Vec::new();
        let outgoing: Vec<_> = if self.source.stage == Stage::Vertex {
            self.source
                .outputs
                .iter()
                .map(|o| {
                    let system = (o.usage == 0 && o.index == 0).then_some(SystemValue::ClipPosition);
                    Ok((
                        Register::Output(o.register),
                        varying(
                            semantic(o.usage, self.at)?,
                            o.index.into(),
                            o.mask,
                            o.register,
                            system,
                        ),
                    ))
                })
                .collect::<Result<_>>()?
        } else {
            self.locals
                .iter()
                .filter_map(|(&r, &(_, mask))| {
                    if let Register::Target(n) = r {
                        Some((r, varying("SV_Target", n, mask, n, Some(SystemValue::Target(n)))))
                    } else {
                        None
                    }
                })
                .collect()
        };
        if self.source.stage == Stage::Vertex
            && !outgoing
                .iter()
                .any(|(_, v)| v.system == Some(SystemValue::ClipPosition))
        {
            return Err(error(0, "vertex program has no declared POSITION0 clip output"));
        }
        for (r, mut v) in outgoing {
            let mask = v
                .provenance
                .d3d
                .as_ref()
                .ok_or_else(|| error(self.at, "output signature missing"))?
                .mask;
            let &(local, written) = self
                .locals
                .get(&r)
                .ok_or_else(|| error(self.at, format!("uninitialized output {r:?}")))?;
            if mask & !written != 0 {
                return Err(error(
                    self.at,
                    format!("{r:?}: uninitialized output lanes {:#x}", mask & !written),
                ));
            }
            v.used = true;
            self.interface.outputs.push(v);
            let ptr = self.b.local_ptr(local);
            let v = self.b.load(ptr, Kind::F, 4);
            let lanes: Vec<_> = (0..4).filter(|n| mask & (1 << n) != 0).collect();
            outputs.push(self.b.swizzle(v, &lanes).expr);
        }
        let io = entry_io(&self.interface).map_err(|e| error(0, e))?;
        let decl = self.b.declare_entry(&io).map_err(|e| error(0, e))?;
        let result = decl.result.map(|ty| {
            self.b.app(E::Compose {
                ty,
                components: outputs,
            })
        });
        self.b.push(Statement::Return { value: result });
        self.b.check().map_err(|e| error(self.at, e))?;
        let module = self.b.finish(
            "main",
            if self.source.stage == Stage::Vertex {
                naga::ShaderStage::Vertex
            } else {
                naga::ShaderStage::Fragment
            },
            decl.arguments,
            decl.result,
            false,
        );
        self.notes.extend(iridia::interface_notes(&self.interface));
        self.notes.sort();
        self.notes.dedup();
        let program = Program {
            module,
            stage: self.source.stage,
            entry: "main".into(),
            interface: self.interface,
            bindings: self.bindings,
            conventions: Conventions::D3D,
            modes: D3dModes {
                min_max_nan: true,
                saturate_nan: true,
                ..Default::default()
            },
            provenance: Provenance {
                format: ShaderFormat::D3d9,
                target: if self.source.stage == Stage::Vertex {
                    "vs_3_0"
                } else {
                    "ps_3_0"
                }
                .into(),
                lifter: "iridia-d3d9".into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(iridia::source_hash(bytes)),
                instruction_count: None,
            },
            notes: self.notes,
        };
        program.validate().map_err(|problems| {
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
