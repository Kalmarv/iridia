//! What a program's declarations and constant table say about its interface: the varyings by semantic, the float
//! constant bank with the CTAB layout of its registers, and a texture and sampler per declared sampler. Both lifters
//! build their interface here, so they describe a program the same way.

use std::collections::BTreeMap;

use iridia_interface::{
    BufferLayout, Constant as Field, ConstantType, D3dBinding, D3dClass, D3dSignature, Interface, InterpMode,
    InterpSampling, Interpolation, Label, Majorness, Resource, ResourceId, ResourceKind, ResourceProvenance,
    SampleType, ScalarType, Stage, StageFacts, SystemValue, TextureDim, Value, Varying, VaryingProvenance,
};

use d3d9_reader::Constant;

use crate::{Register, Result, Shader, error};

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

/// The float constant registers of a `vs_3_0` program, the rows a relative read can reach.
pub const VERTEX_CONSTANTS: u32 = 256;

/// The partial-precision note: a request the lifters compute at `f32`.
pub const PARTIAL_PRECISION: &str = "partial-precision requests computed at f32";

/// The float constant bank: the resource and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    /// The constant buffer resource.
    pub resource: ResourceId,
    /// Its rows of four lanes: past the last register read or allocated, all 256 of a vertex program that reads
    /// relative to `a0`.
    pub rows: u32,
}

/// A declared sampler's texture and sampler resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampler {
    /// The texture.
    pub texture: ResourceId,
    /// The sampler state.
    pub sampler: ResourceId,
    /// The texture's dimension, as declared.
    pub dim: TextureDim,
}

/// A program's interface before its code is lowered: the inputs (none used yet), the resources, and no outputs (see
/// [`outputs`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The interface: inputs in declaration order, the bank, then a texture and a sampler per sampler register.
    pub interface: Interface,
    /// The input slot of each declared input register, with its declared mask.
    pub inputs: BTreeMap<u32, (u32, u8)>,
    /// The float constant bank, where the program reads or allocates a register of it.
    pub bank: Option<Bank>,
    /// The samplers by register.
    pub samplers: BTreeMap<u32, Sampler>,
    /// What the declarations show that the lift cannot state exactly.
    pub notes: Vec<String>,
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

/// The bank's register layout is exact even for an optimized partial array. CTAB's unshortened type is retained
/// by `read`; the interface describes only the allocation that the compiled shader actually binds.
fn field(c: &Constant) -> Result<Field> {
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
    // a single matrix the compiler allocated fewer lines than it has (its last rows or columns are never read) is
    // described by the lines allocated, as a partial array is
    let element = match element {
        ConstantType::Matrix {
            scalar,
            rows,
            columns,
            major,
            stride,
        } if t.elements == 1 && (1..registers).contains(&c.count) => {
            let kept = lanes(c.count)?;
            let (rows, columns) = match major {
                Majorness::Row => (kept, columns),
                Majorness::Column => (rows, kept),
            };
            ConstantType::Matrix {
                scalar,
                rows,
                columns,
                major,
                stride,
            }
        }
        element => element,
    };
    let partial_matrix = t.elements == 1 && (1..registers).contains(&c.count);
    let Some(count) = c
        .count
        .checked_div(registers)
        .filter(|&n| partial_matrix || (c.count.is_multiple_of(registers) && n <= t.elements))
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

fn resource(
    interface: &mut Interface,
    kind: ResourceKind,
    name: String,
    class: D3dClass,
    register: u32,
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
            d3d: Some(D3dBinding {
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

/// The interface the declarations and constant table state, the CTAB refusals included.
pub fn signature(source: &Shader) -> Result<Signature> {
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
        samplers: BTreeMap::new(),
        notes: Vec::new(),
    };
    for i in &source.inputs {
        let mut v = varying(semantic(i.usage, 0)?, i.index.into(), i.mask, i.register, None);
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
    let mut rows = 0;
    for i in &source.instructions {
        for src in &i.sources {
            if src.relative.is_some() {
                // a relative read can reach any row of the vertex constant file
                rows = rows.max(VERTEX_CONSTANTS);
            } else if let Register::Constant(n) = src.register
                && !source.definitions.contains_key(&n)
            {
                rows = rows.max(n.saturating_add(1));
            }
        }
    }
    let mut fields = Vec::new();
    if let Some(ctab) = &source.constants {
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
                        s.notes.push(format!(
                            "CTAB {} retains its declared type; its optimized allocation binds {} registers",
                            c.name, c.count
                        ));
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
                    if source.samplers.get(&u32::from(c.register)) != Some(&expected) {
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
    if rows > 0 {
        let layout = BufferLayout {
            size: rows.saturating_mul(16),
            constants: fields,
        };
        let id = resource(
            &mut s.interface,
            ResourceKind::ConstantBuffer { layout },
            "ps_float_constants".into(),
            D3dClass::ConstantBuffer,
            0,
        )?;
        s.bank = Some(Bank { resource: id, rows });
    }
    for (&slot, &dim) in &source.samplers {
        if dim == TextureDim::D1 {
            return Err(error(0, "1D sampler"));
        }
        let name = source
            .constants
            .as_ref()
            .and_then(|c| {
                c.constants
                    .iter()
                    .find(|c| c.register_set == 3 && u32::from(c.register) == slot)
            })
            .map_or_else(|| format!("s{slot}"), |c| c.name.clone());
        let kind = ResourceKind::Texture {
            dim,
            arrayed: false,
            multisampled: false,
            sample: SampleType::Float,
        };
        let texture = resource(
            &mut s.interface,
            kind,
            name.clone(),
            D3dClass::ShaderResource,
            slot,
        )?;
        let sampler = resource(
            &mut s.interface,
            ResourceKind::Sampler { comparison: false },
            name,
            D3dClass::Sampler,
            slot,
        )?;
        s.samplers.insert(
            slot,
            Sampler {
                texture,
                sampler,
                dim,
            },
        );
    }
    Ok(s)
}

/// The outputs of a program whose registers were written in the lanes `written` gives: a vertex program's declared
/// outputs in declaration order, or a pixel program's written colour targets in register order. Each is the register
/// it is read from with its varying, marked used; a vertex program with no `POSITION0`, and a declared lane that is
/// never written, are refused (`at` is where the refusal points).
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
                let system = (o.usage == 0 && o.index == 0).then_some(SystemValue::ClipPosition);
                Ok((
                    Register::Output(o.register),
                    varying(semantic(o.usage, at)?, o.index.into(), o.mask, o.register, system),
                ))
            })
            .collect::<Result<_>>()?
    } else {
        written
            .iter()
            .filter_map(|(&r, &mask)| match r {
                Register::Target(n) => {
                    Some((r, varying("SV_Target", n, mask, n, Some(SystemValue::Target(n)))))
                }
                _ => None,
            })
            .collect()
    };
    if source.stage == Stage::Vertex
        && !outgoing
            .iter()
            .any(|(_, v)| v.system == Some(SystemValue::ClipPosition))
    {
        return Err(error(0, "vertex program has no declared POSITION0 clip output"));
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
            .ok_or_else(|| error(at, format!("uninitialized output {r:?}")))?;
        if mask & !done != 0 {
            return Err(error(
                at,
                format!("{r:?}: uninitialized output lanes {:#x}", mask & !done),
            ));
        }
        v.used = true;
        out.push((r, v));
    }
    Ok(out)
}
