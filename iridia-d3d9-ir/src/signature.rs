//! What a program's declarations, implied registers and constant table say about its interface: the varyings by
//! semantic, the float, integer and boolean constant banks with the CTAB layout of their registers, a texture and
//! sampler per sampler, and the pipeline state the code reads.

use std::collections::BTreeMap;

use iridia_ir::interface::{
    BufferLayout, Constant as Field, ConstantType, D3dBinding, D3dClass, D3dSignature, Interface, InterpMode,
    InterpSampling, Interpolation, Label, Majorness, Member, Resource, ResourceId, ResourceKind,
    ResourceProvenance, SampleType, ScalarType, Stage, StageFacts, StateParam, SystemValue, TextureDim,
    Value, Varying, VaryingProvenance,
};

use crate::{Constant, Register, Result, Shader, error, model::usage};

/// D3DDECLUSAGE's names, by usage number.
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

/// The float constant registers of a vertex program, the rows a relative read can reach.
pub const VERTEX_CONSTANTS: u32 = 256;

/// The float constant registers of a pixel 3.0 program, the rows a read relative to `aL` can reach.
pub const PIXEL_CONSTANTS: u32 = 224;

/// The partial-precision note: a request the lifters compute at `f32`.
pub const PARTIAL_PRECISION: &str = "partial-precision requests computed at f32";

/// A constant bank: the resource and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    /// The constant buffer resource.
    pub resource: ResourceId,
    /// Its registers: past the last register read or allocated, the whole bank where a read is relative.
    pub rows: u32,
}

/// A sampler's texture and sampler resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sampler {
    /// The sampler state.
    pub sampler: ResourceId,
    /// The texture for each dimension it can have: the declared one, or for a pixel 1.x texture stage whose dimension
    /// is not known, one each for 2D, cube and volume.
    pub textures: Vec<(TextureDim, ResourceId)>,
    /// The stage's `StateParam::TextureDim`, where there is more than one texture to pick from.
    pub dim_state: Option<ResourceId>,
}

/// A program's interface before its code is lowered: the inputs (none used yet), the resources, and no outputs (see
/// [`outputs`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The interface: inputs in declaration order, the banks, a texture (or one per possible dimension) and a sampler
    /// per sampler register, and pipeline state as the lift reads it.
    pub interface: Interface,
    /// The input slot of each input register, with its declared mask.
    pub inputs: BTreeMap<Register, (u32, u8)>,
    /// The float constant bank, where the program reads or allocates a register of it.
    pub bank: Option<Bank>,
    /// The integer constant bank (`iN`, a `vec4<i32>` each), where the program reads one.
    pub integers: Option<Bank>,
    /// The boolean constant bank (`bN`, a 32-bit boolean each, packed), where the program reads one.
    pub booleans: Option<Bank>,
    /// The samplers by register (texture stage, in pixel 1.x).
    pub samplers: BTreeMap<u32, Sampler>,
    /// What the declarations show that the lift cannot state exactly.
    pub notes: Vec<String>,
}

impl Signature {
    /// The pipeline state resource of `param`, declared the first time it is asked for.
    pub fn state(&mut self, param: StateParam) -> Result<ResourceId> {
        let found = self
            .interface
            .resources
            .iter()
            .find(|r| r.kind == ResourceKind::PipelineState(param))
            .map(|r| r.id);
        match found {
            Some(id) => Ok(id),
            None => resource(
                &mut self.interface,
                ResourceKind::PipelineState(param),
                format!("{param:?}"),
                None,
            ),
        }
    }
}

fn value(width: u8) -> Value {
    Value {
        scalar: ScalarType::Float,
        width,
    }
}

/// How many lanes a write mask names.
pub fn lanes_in(mask: u8) -> u8 {
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

/// A float varying of `name` and `index` with the lanes of `mask`, at register `register`, unused.
pub fn varying(name: &str, index: u32, mask: u8, register: u32, system: Option<SystemValue>) -> Varying {
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
            ..Default::default()
        },
    }
}

/// The pipeline's input a misc register reads: `vPos` the pixel's window position, `vFace` its primitive's facing.
/// Neither has a signature register.
fn misc_input(r: Register) -> Option<Varying> {
    let (name, system, value) = match r {
        Register::PixelPosition => ("VPOS", SystemValue::FragCoord, value(4)),
        Register::Face => (
            "VFACE",
            SystemValue::FrontFacing,
            Value {
                scalar: ScalarType::Bool,
                width: 1,
            },
        ),
        _ => return None,
    };
    Some(Varying {
        label: Label {
            name: name.into(),
            index: 0,
        },
        system: Some(system),
        value,
        interp: None,
        used: false,
        provenance: VaryingProvenance::default(),
    })
}

/// A CTAB register set's registers: the bytes each takes in its bank, the scalar its lanes hold, and whether a value
/// takes a register per lane.
#[derive(Clone, Copy)]
struct Set {
    size: u32,
    scalar: ScalarType,
    per_lane: bool,
}

/// The register set `n` (0 boolean, 1 integer, 2 float) of constants.
fn set(n: u16) -> Option<Set> {
    let (size, scalar, per_lane) = match n {
        0 => (4, ScalarType::Bool, true),
        1 => (16, ScalarType::Sint, true),
        2 => (16, ScalarType::Float, false),
        _ => return None,
    };
    Some(Set {
        size,
        scalar,
        per_lane,
    })
}

/// A CTAB type's layout in a register set, and the registers one value of it takes. Every value starts a register:
/// in the float set a scalar or vector takes one, a matrix one per line (a row-major matrix's rows, a column-major
/// one's columns); in the boolean and integer sets each lane takes one, a matrix's lanes counted by rows whatever its
/// majorness (an `int2` takes `i0.x` and `i1.x`). A structure's members follow each other, an
/// array's elements likewise. The lanes are the set's scalar whatever the declared type: D3DX converts a bool or int
/// set in the float registers to floats, and the program reads floats.
fn layout(name: &str, t: &crate::Type, set: Set) -> Result<(ConstantType, u32)> {
    let refuse = || {
        error(
            0,
            format!(
                "CTAB {name}: type {t:?} in a register set of {} lanes",
                set.scalar_name()
            ),
        )
    };
    let lanes = |n: u16| {
        u8::try_from(n)
            .ok()
            .filter(|n| (1..=4).contains(n))
            .ok_or_else(refuse)
    };
    let (element, registers) = match (t.class, t.kind) {
        (5, _) if !t.members.is_empty() => {
            let (mut members, mut at) = (Vec::new(), 0u32);
            for m in &t.members {
                let (ty, registers) = layout(&format!("{name}.{}", m.name), &m.ty, set)?;
                members.push(Member {
                    name: m.name.clone(),
                    offset: at.checked_mul(set.size).ok_or_else(refuse)?,
                    ty,
                });
                at = at.checked_add(registers).ok_or_else(refuse)?;
            }
            (ConstantType::Struct { members }, at)
        }
        (0..=3, 1..=3) if t.members.is_empty() => {
            let (rows, columns) = (lanes(t.rows)?, lanes(t.columns)?);
            let shape = match t.class {
                0 if rows == 1 && columns == 1 => Shape::Scalar,
                1 if rows == 1 => Shape::Vector(columns),
                2 => Shape::Matrix(rows, columns, Majorness::Row),
                3 => Shape::Matrix(rows, columns, Majorness::Column),
                _ => return Err(refuse()),
            };
            let scalar = ConstantType::Scalar(set.scalar);
            let run = |element: ConstantType, count: u8, stride: u32| ConstantType::Array {
                element: Box::new(element),
                count: count.into(),
                stride,
            };
            match shape {
                Shape::Scalar => (scalar, 1),
                // the boolean and integer sets: a register per lane, in its first lane, a matrix's in row order (an
                // allocation can end inside a row, so the lanes are one run)
                Shape::Vector(n) if set.per_lane => (run(scalar, n, set.size), n.into()),
                Shape::Matrix(rows, columns, _) if set.per_lane => {
                    let n = rows.checked_mul(columns).ok_or_else(refuse)?;
                    (run(scalar, n, set.size), n.into())
                }
                Shape::Vector(n) => (
                    ConstantType::Vector(Value {
                        scalar: set.scalar,
                        width: n,
                    }),
                    1,
                ),
                Shape::Matrix(rows, columns, major) => (
                    ConstantType::Matrix {
                        scalar: set.scalar,
                        rows,
                        columns,
                        major,
                        stride: set.size,
                    },
                    u32::from(match major {
                        Majorness::Row => rows,
                        Majorness::Column => columns,
                    }),
                ),
            }
        }
        _ => return Err(refuse()),
    };
    match t.elements {
        0 => Err(refuse()),
        1 => Ok((element, registers)),
        n => Ok((
            ConstantType::Array {
                element: Box::new(element),
                count: n.into(),
                stride: registers.checked_mul(set.size).ok_or_else(refuse)?,
            },
            registers.checked_mul(n.into()).ok_or_else(refuse)?,
        )),
    }
}

/// A leaf CTAB type's shape.
#[derive(Clone, Copy)]
enum Shape {
    Scalar,
    Vector(u8),
    Matrix(u8, u8, Majorness),
}

impl Set {
    fn scalar_name(self) -> &'static str {
        match self.scalar {
            ScalarType::Bool => "bool",
            ScalarType::Sint => "int",
            _ => "float",
        }
    }
}

/// `ty`, which takes `registers` of `size` bytes, cut to its first `keep`: the elements, members and matrix lines an
/// optimized allocation binds. `None` where the cut falls inside an array element that is not a matrix, which no
/// single type describes.
fn cut(ty: &ConstantType, registers: u32, keep: u32, size: u32) -> Option<ConstantType> {
    if keep >= registers {
        return Some(ty.clone());
    }
    match ty {
        ConstantType::Matrix {
            scalar,
            rows,
            columns,
            major,
            stride,
        } => {
            let kept = u8::try_from(keep).ok()?;
            let (rows, columns) = match major {
                Majorness::Row => (kept, *columns),
                Majorness::Column => (*rows, kept),
            };
            Some(ConstantType::Matrix {
                scalar: *scalar,
                rows,
                columns,
                major: *major,
                stride: *stride,
            })
        }
        ConstantType::Array {
            element,
            count,
            stride,
        } => {
            let each = stride.checked_div(size)?;
            let (whole, part) = (keep.checked_div(each)?, keep.checked_rem(each)?);
            match (whole, part) {
                (_, 0) => Some(ConstantType::Array {
                    element: element.clone(),
                    count: whole,
                    stride: *stride,
                }),
                // one element, part of it: as the single value it is
                (0, part) if *count > 0 => cut(element, each, part, size),
                _ => None,
            }
        }
        ConstantType::Struct { members } => {
            let mut kept = Vec::new();
            for (i, m) in members.iter().enumerate() {
                let start = m.offset.checked_div(size)?;
                if start >= keep {
                    break;
                }
                let end = match members.get(i.checked_add(1)?) {
                    Some(n) => n.offset.checked_div(size)?,
                    None => registers,
                };
                kept.push(Member {
                    name: m.name.clone(),
                    offset: m.offset,
                    ty: cut(&m.ty, end.checked_sub(start)?, keep.checked_sub(start)?, size)?,
                });
            }
            Some(ConstantType::Struct { members: kept })
        }
        // a scalar or vector takes one register, which an allocation always binds
        _ => Some(ty.clone()),
    }
}

/// A boolean, integer or float constant's layout: its declared type in its register set, cut to the registers its
/// allocation binds (an optimized allocation binds those up to the last the program reads), and whether it was cut.
fn field(c: &Constant) -> Result<(Field, bool)> {
    let set = set(c.register_set)
        .ok_or_else(|| error(0, format!("CTAB {}: register set {}", c.name, c.register_set)))?;
    let (declared, registers) = layout(&c.name, &c.ty, set)?;
    let keep = u32::from(c.count);
    if keep > registers {
        return Err(error(
            0,
            format!(
                "CTAB {}: an allocation of {keep} registers, where its type takes {registers}",
                c.name
            ),
        ));
    }
    let ty = cut(&declared, registers, keep, set.size).ok_or_else(|| {
        error(
            0,
            format!(
                "CTAB {}: an allocation of {keep} registers ends inside an element of {declared:?}",
                c.name
            ),
        )
    })?;
    Ok((
        with_default(c, ty, u32::from(c.register).saturating_mul(set.size))?,
        keep < registers,
    ))
}

fn with_default(c: &Constant, ty: ConstantType, offset: u32) -> Result<Field> {
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
        offset,
        ty,
        default,
    })
}

fn resource(
    interface: &mut Interface,
    kind: ResourceKind,
    name: String,
    binding: Option<(D3dClass, u32)>,
) -> Result<ResourceId> {
    let id = ResourceId(
        u32::try_from(interface.resources.len()).map_err(|_| error(0, "more than 4294967295 resources"))?,
    );
    interface.resources.push(Resource {
        id,
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name,
            d3d: binding.map(|(class, register)| D3dBinding {
                class,
                register,
                space: 0,
            }),
            counter_of: None,
            layout: None,
        },
    });
    Ok(id)
}

/// A constant bank of `rows` registers of `stride` bytes, named `name`, at `cb{register}`.
fn bank(
    s: &mut Signature,
    name: &str,
    register: u32,
    rows: u32,
    stride: u32,
    fields: Vec<Field>,
) -> Result<Bank> {
    let layout = BufferLayout {
        size: rows.saturating_mul(stride).next_multiple_of(16),
        constants: fields,
    };
    let id = resource(
        &mut s.interface,
        ResourceKind::ConstantBuffer { layout },
        name.into(),
        Some((D3dClass::ConstantBuffer, register)),
    )?;
    Ok(Bank { resource: id, rows })
}

/// The interface the declarations, the implied registers and the constant table state, the CTAB refusals included.
/// `texture_dims` gives the dimension of pixel 1.x texture stages the host has stated; a stage it does not name gets a
/// texture of each dimension and a `StateParam::TextureDim` to pick by.
pub fn signature(source: &Shader, texture_dims: &BTreeMap<u32, TextureDim>) -> Result<Signature> {
    let float_bank = if source.stage == Stage::Vertex {
        VERTEX_CONSTANTS
    } else {
        PIXEL_CONSTANTS
    };
    signature_with_bank(source, texture_dims, float_bank)
}

/// [`signature`] with the rows of the float constant bank a relative read can reach stated: a vertex program's slice run
/// as a pixel invocation still reads the vertex constant file.
pub(crate) fn signature_with_bank(
    source: &Shader,
    texture_dims: &BTreeMap<u32, TextureDim>,
    float_bank: u32,
) -> Result<Signature> {
    let mut s = Signature {
        interface: Interface {
            stage: source.stage,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: Vec::new(),
            facts: StageFacts::None,
        },
        inputs: BTreeMap::new(),
        bank: None,
        integers: None,
        booleans: None,
        samplers: BTreeMap::new(),
        notes: Vec::new(),
    };
    for i in &source.inputs {
        if let Some(v) = misc_input(i.register) {
            let slot = u32::try_from(s.interface.inputs.len())
                .map_err(|_| error(0, "more than 4294967295 inputs"))?;
            s.inputs.insert(i.register, (slot, i.mask));
            s.interface.inputs.push(v);
            continue;
        }
        let register = match i.register {
            Register::Input(n) | Register::Texture(n) => n,
            other => return Err(error(0, format!("{} is not an input register", other.name()))),
        };
        let mut v = varying(semantic(i.usage, 0)?, i.index.into(), i.mask, register, None);
        v.interp = (source.stage == Stage::Pixel).then_some(Interpolation {
            mode: InterpMode::Perspective,
            sampling: if i.centroid {
                InterpSampling::Centroid
            } else {
                InterpSampling::Center
            },
        });
        if i.partial_precision {
            s.notes.push(PARTIAL_PRECISION.into());
        }
        let slot =
            u32::try_from(s.interface.inputs.len()).map_err(|_| error(0, "more than 4294967295 inputs"))?;
        s.inputs.insert(i.register, (slot, i.mask));
        s.interface.inputs.push(v);
    }
    // the rows of each bank the code reads; a relative read can reach the whole bank
    let (mut rows, mut integers, mut booleans) = (0, 0, 0);
    let reads = source
        .instructions
        .iter()
        .flat_map(|i| i.sources.iter().chain(i.predicate.iter()));
    for src in reads {
        match src.register {
            Register::Constant(_) if src.relative.is_some() => rows = rows.max(float_bank),
            Register::Constant(n) if !source.definitions.contains_key(&n) => {
                rows = rows.max(n.saturating_add(1))
            }
            Register::Integer(n) if !source.integers.contains_key(&n) => {
                integers = integers.max(n.saturating_add(1))
            }
            Register::Boolean(n) if !source.booleans.contains_key(&n) => {
                booleans = booleans.max(n.saturating_add(1))
            }
            _ => {}
        }
    }
    // a matrix operation reads the registers after its second source's
    for i in &source.instructions {
        if let (Some(extra), Some(Register::Constant(n))) = (
            i.matrix_rows().map(|r| r.saturating_sub(1)),
            i.sources.get(1).map(|s| s.register),
        ) {
            let last = n.saturating_add(extra);
            if i.sources.get(1).is_some_and(|s| s.relative.is_some()) {
                rows = rows.max(float_bank);
            } else if !source.definitions.contains_key(&last) {
                rows = rows.max(last.saturating_add(1));
            }
        }
    }
    let (mut fields, mut integer_fields, mut boolean_fields) = (Vec::new(), Vec::new(), Vec::new());
    if let Some(ctab) = &source.constants {
        for c in &ctab.constants {
            let end = u32::from(c.register).saturating_add(u32::from(c.count));
            match c.register_set {
                0..=2 => {
                    let (f, cut) = field(c)?;
                    let bank = match c.register_set {
                        0 => (&mut booleans, &mut boolean_fields),
                        1 => (&mut integers, &mut integer_fields),
                        _ => (&mut rows, &mut fields),
                    };
                    *bank.0 = (*bank.0).max(end);
                    bank.1.push(f);
                    if cut {
                        s.notes.push(format!(
                            "CTAB {}: its optimized allocation binds {} registers of its type, which the interface \
                             describes",
                            c.name, c.count
                        ));
                    }
                    let declared = match c.ty.kind {
                        1 => "bool",
                        2 => "int",
                        _ => "",
                    };
                    let held = set(c.register_set).map_or("float", Set::scalar_name);
                    if c.ty.members.is_empty() && !declared.is_empty() && declared != held {
                        s.notes.push(format!(
                            "CTAB {}: a {declared} constant in the {held} registers, which hold its lanes as {held}s",
                            c.name
                        ));
                    }
                }
                3 => {
                    // a sampler array takes a register per element, those up to the last the program samples
                    if c.ty.class != 4
                        || !matches!(c.ty.kind, 5..=14)
                        || !(1..=c.ty.elements).contains(&c.count)
                    {
                        return Err(error(0, format!("CTAB {}: sampler type {:?}", c.name, c.ty)));
                    }
                    // D3DXPARAMETER_TYPE's textures (5 to 9) and samplers (10 to 14), each untyped, 1D, 2D, 3D and
                    // cube; pixel 1.x declares no samplers, so its table is all there is to check against
                    let expected = match c.ty.kind {
                        6 | 11 => Some(TextureDim::D1),
                        7 | 12 => Some(TextureDim::D2),
                        8 | 13 => Some(TextureDim::D3),
                        9 | 14 => Some(TextureDim::Cube),
                        _ => None,
                    };
                    if let Some(expected) = expected.filter(|_| !source.pixel_1x()) {
                        let first = u32::from(c.register);
                        for slot in first..first.saturating_add(c.count.into()) {
                            // a register of an array the program does not sample is not declared
                            let declared = source.samplers.get(&slot);
                            if declared.is_some_and(|&d| d != expected)
                                || (c.ty.elements == 1 && declared.is_none())
                            {
                                return Err(error(0, format!("CTAB {} sampler disagrees with DCL", c.name)));
                            }
                        }
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
    // the float bank keeps the name iridia-d3d9 gives it in both stages, so the two lifters' interfaces are equal; the
    // integer and boolean banks, which that lifter does not have, are named after it
    if rows > 0 {
        s.bank = Some(bank(&mut s, "ps_float_constants", 0, rows, 16, fields)?);
    }
    if integers > 0 {
        s.integers = Some(bank(&mut s, "ps_int_constants", 1, integers, 16, integer_fields)?);
    }
    if booleans > 0 {
        s.booleans = Some(bank(&mut s, "ps_bool_constants", 2, booleans, 4, boolean_fields)?);
    }
    for (&slot, &declared) in &source.samplers {
        if declared == TextureDim::D1 {
            return Err(error(0, "1D sampler"));
        }
        let name = source
            .constants
            .as_ref()
            .and_then(|c| {
                c.constants.iter().find_map(|c| {
                    let first = u32::from(c.register);
                    let i = slot.checked_sub(first).filter(|&i| i < u32::from(c.count))?;
                    (c.register_set == 3).then(|| {
                        if c.ty.elements > 1 {
                            format!("{}[{i}]", c.name)
                        } else {
                            c.name.clone()
                        }
                    })
                })
            })
            .unwrap_or_else(|| format!("s{slot}"));
        let dims = if !source.pixel_1x() {
            vec![declared]
        } else if let Some(&dim) = texture_dims.get(&slot) {
            vec![dim]
        } else {
            vec![TextureDim::D2, TextureDim::Cube, TextureDim::D3]
        };
        let mut textures = Vec::new();
        for &dim in &dims {
            let kind = ResourceKind::Texture {
                dim,
                arrayed: false,
                multisampled: false,
                sample: SampleType::Float,
            };
            let texture_name = if dims.len() > 1 {
                format!("{name}_{dim:?}")
            } else {
                name.clone()
            };
            let texture = resource(
                &mut s.interface,
                kind,
                texture_name,
                Some((D3dClass::ShaderResource, slot)),
            )?;
            textures.push((dim, texture));
        }
        let sampler = resource(
            &mut s.interface,
            ResourceKind::Sampler { comparison: false },
            name,
            Some((D3dClass::Sampler, slot)),
        )?;
        let dim_state = if textures.len() > 1 {
            Some(s.state(StateParam::TextureDim(slot))?)
        } else {
            None
        };
        s.samplers.insert(
            slot,
            Sampler {
                sampler,
                textures,
                dim_state,
            },
        );
    }
    Ok(s)
}

/// The outputs of a program whose registers were written in the lanes `written` gives: a vertex program's declared
/// or implied outputs in order, or a pixel program's written colour targets in register order (`r0` in pixel 1.x)
/// and its depth. Each is the register it is read from with its varying, marked used; a vertex program with no
/// `POSITION0`, and a lane of an output that is never written, are refused (`at` is where the refusal points).
pub fn outputs(
    source: &Shader,
    written: &BTreeMap<Register, u8>,
    at: usize,
) -> Result<Vec<(Register, Varying)>> {
    let outgoing: Vec<_> = if source.stage == Stage::Vertex {
        source
            .outputs
            .iter()
            .map(|o| {
                let system =
                    (o.usage == usage::POSITION && o.index == 0).then_some(SystemValue::ClipPosition);
                let register = match o.register {
                    Register::Output(n) | Register::Color(n) | Register::TexCoord(n) => n,
                    _ => 0,
                };
                Ok((
                    o.register,
                    varying(semantic(o.usage, at)?, o.index.into(), o.mask, register, system),
                ))
            })
            .collect::<Result<_>>()?
    } else if source.pixel_1x() {
        let mut out = vec![(
            Register::Temporary(0),
            varying("SV_Target", 0, 15, 0, Some(SystemValue::Target(0))),
        )];
        if written.contains_key(&Register::Depth) {
            out.push((
                Register::Depth,
                varying("SV_Depth", 0, 1, 0, Some(SystemValue::Depth)),
            ));
        }
        out
    } else {
        written
            .iter()
            .filter_map(|(&r, &mask)| match r {
                Register::Target(n) => {
                    Some((r, varying("SV_Target", n, mask, n, Some(SystemValue::Target(n)))))
                }
                Register::Depth => Some((r, varying("SV_Depth", 0, 1, 0, Some(SystemValue::Depth)))),
                _ => None,
            })
            .collect()
    };
    if source.stage == Stage::Vertex
        && !outgoing
            .iter()
            .any(|(_, v)| v.system == Some(SystemValue::ClipPosition))
    {
        let what = if source.fixed_outputs() {
            "vertex program never writes oPos"
        } else {
            "vertex program has no declared POSITION0 clip output"
        };
        return Err(error(0, what));
    }
    let mut out = Vec::new();
    for (r, mut v) in outgoing {
        let mask = v
            .provenance
            .d3d
            .as_ref()
            .ok_or_else(|| error(at, "output signature missing"))?
            .mask;
        let &done = written
            .get(&r)
            .ok_or_else(|| error(at, format!("uninitialized output {}", r.name())))?;
        if mask & !done != 0 {
            return Err(error(
                at,
                format!("{}: uninitialized output lanes {:#x}", r.name(), mask & !done),
            ));
        }
        v.used = true;
        out.push((r, v));
    }
    Ok(out)
}

/// Mark lanes `required` of input register `register` read: refused where the program does not declare them.
pub fn read_input(s: &mut Signature, register: Register, required: u8, at: usize) -> Result<u32> {
    let &(slot, mask) = s
        .inputs
        .get(&register)
        .ok_or_else(|| error(at, format!("undeclared input {}", register.name())))?;
    if required & !mask != 0 {
        return Err(error(
            at,
            format!("{}: read of undeclared lanes {required:#x}", register.name()),
        ));
    }
    let input = s.interface.inputs.get_mut(slot as usize).ok_or_else(|| {
        error(
            at,
            format!(
                "input {} is input {slot} of a list that is shorter",
                register.name()
            ),
        )
    })?;
    input.used = true;
    if let Some(d) = &mut input.provenance.d3d {
        d.read_write_mask |= required;
    }
    Ok(slot)
}
