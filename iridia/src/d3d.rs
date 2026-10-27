//! D3D's enum numbers and packing rules, mapped into the vocabulary ([`iridia_interface`]). The DXBC and DXIL
//! readers share them: both state signatures and constant buffers in D3D's terms. Nothing here is evaluation; a
//! number the table does not map is refused with its name (an `Err` the reader reports), never guessed.

use iridia_interface::{
    ConstantType, D3dSignature, InterpMode, InterpSampling, Interpolation, Label, Majorness, ScalarType,
    Stage, SystemValue, Value, Varying, VaryingProvenance,
};

/// Why a D3D number, element or type is not mapped. Every message says what was expected.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A number the enum has no name for.
    #[error("{what} {value} is not a known value")]
    Unknown { what: &'static str, value: u32 },
    /// A number the enum has, which the vocabulary does not carry.
    #[error("{what} {value} ({name}) is not mapped")]
    Unmapped {
        what: &'static str,
        value: u32,
        name: &'static str,
    },
    /// A vector or matrix of a number of lanes or lines D3D does not have.
    #[error("{what} of {value} is not within 1 to 4")]
    LaneCount { what: &'static str, value: u32 },
    #[error("an array of no elements")]
    EmptyArray,
    #[error("array element size overflow")]
    ArrayElementSize,
    #[error("array size overflow")]
    ArraySize,
    /// A signature element (its semantic name and index), and what is wrong with it.
    #[error("{element}: {problem}")]
    Element {
        element: String,
        problem: ElementProblem,
    },
}

/// What is wrong with a signature element ([`Error::Element`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ElementProblem {
    #[error("lane masks {mask:#x} and {read_write_mask:#x} are not within the register's four lanes")]
    LaneMasks { mask: u8, read_write_mask: u8 },
    #[error("register {register} is not one of the {limit} a signature has (0 to {last})", last = limit.saturating_sub(1))]
    Register { register: u32, limit: u32 },
    #[error(
        "{} as {} of the {stage:?} stage",
        describe("D3D_NAME", *value, *name),
        if *output { "an output" } else { "an input" }
    )]
    SystemValue {
        value: u32,
        name: Option<&'static str>,
        stage: Stage,
        output: bool,
    },
    #[error("the semantic disagrees with its stored D3D_NAME {stored}")]
    SemanticDisagrees { stored: u32 },
    #[error("system-value semantic {name} is not mapped")]
    SemanticUnmapped { name: String },
    #[error("{}", describe("signature component type", *value, None))]
    ComponentType { value: u32 },
}

/// One element of a D3D signature as it is stored (`ISGN`, `OSGN`, `ISG1`, `OSG1`).
#[derive(Clone, Copy, Debug)]
pub struct Element<'a> {
    pub name: &'a str,
    pub index: u32,
    /// `D3D_NAME`.
    pub system_value: u32,
    /// `D3D_REGISTER_COMPONENT_TYPE`.
    pub component_type: u32,
    pub register: u32,
    pub mask: u8,
    /// The signature's second mask: for an input the lanes always read, for an output the lanes never written.
    pub read_write_mask: u8,
}

/// `D3D_NAME`'s name for a value, where the enum has one.
pub fn name_of(value: u32) -> Option<&'static str> {
    Some(match value {
        0 => "UNDEFINED",
        1 => "POSITION",
        2 => "CLIP_DISTANCE",
        3 => "CULL_DISTANCE",
        4 => "RENDER_TARGET_ARRAY_INDEX",
        5 => "VIEWPORT_ARRAY_INDEX",
        6 => "VERTEX_ID",
        7 => "PRIMITIVE_ID",
        8 => "INSTANCE_ID",
        9 => "IS_FRONT_FACE",
        10 => "SAMPLE_INDEX",
        11..=23 => "FINAL_*_TESSFACTOR",
        64 => "TARGET",
        65 => "DEPTH",
        66 => "COVERAGE",
        67 => "DEPTH_GREATER_EQUAL",
        68 => "DEPTH_LESS_EQUAL",
        69 => "STENCIL_REF",
        70 => "INNER_COVERAGE",
        _ => return None,
    })
}

/// A refusal of `value` of the enum `what`, in words.
fn describe(what: &str, value: u32, name: Option<&str>) -> String {
    match name {
        Some(name) => format!("{what} {value} ({name}) is not mapped"),
        None => format!("{what} {value} is not a known value"),
    }
}

/// The system value a `D3D_NAME` number stands for in `stage`'s input or output signature; `Ok(None)` for a plain
/// varying (0).
fn system_from_number(
    stage: Stage,
    output: bool,
    value: u32,
    index: u32,
) -> Result<Option<SystemValue>, ElementProblem> {
    use SystemValue::*;
    Ok(Some(match (stage, output, value) {
        (_, _, 0) => return Ok(None),
        // between a vertex program, a hull program and a domain program the signatures carry plain varyings (a
        // position the vertex program wrote is one the hull program reads), and the tessellation factors are named by
        // their semantic
        (Stage::Hull | Stage::Domain, false, _) | (Stage::Hull, true, _) | (Stage::Domain, true, 11..=16) => {
            return Ok(None);
        }
        // a geometry program reads plain varyings and writes a vertex program's system values
        (Stage::Geometry, false, _) => return Ok(None),
        (Stage::Geometry, true, 1) => ClipPosition,
        (Stage::Geometry, true, 2) => ClipDistance,
        (Stage::Geometry, true, 3) => CullDistance,
        (Stage::Geometry, true, 4) => RenderTargetArrayIndex,
        (Stage::Geometry, true, 5) => ViewportArrayIndex,
        (Stage::Geometry, true, 7) => PrimitiveIndex,
        // a mesh program writes a vertex program's system values per vertex (its per-primitive ones are not carried)
        (Stage::Mesh, false, _) => return Ok(None),
        (Stage::Mesh, true, 1) => ClipPosition,
        (Stage::Mesh, true, 2) => ClipDistance,
        (Stage::Mesh, true, 3) => CullDistance,
        (Stage::Domain, true, 1) => ClipPosition,
        (Stage::Domain, true, 2) => ClipDistance,
        (Stage::Domain, true, 3) => CullDistance,
        (Stage::Domain, true, 4) => RenderTargetArrayIndex,
        (Stage::Domain, true, 5) => ViewportArrayIndex,
        (Stage::Vertex, true, 1) => ClipPosition,
        (Stage::Pixel, false, 1) => FragCoord,
        (Stage::Vertex, true, 2) | (Stage::Pixel, false, 2) => ClipDistance,
        (Stage::Vertex, true, 3) | (Stage::Pixel, false, 3) => CullDistance,
        (Stage::Vertex, true, 4) | (Stage::Pixel, false, 4) => RenderTargetArrayIndex,
        (Stage::Vertex, true, 5) | (Stage::Pixel, false, 5) => ViewportArrayIndex,
        (Stage::Vertex, false, 6) => VertexIndex,
        (Stage::Pixel, false, 7) => PrimitiveIndex,
        (Stage::Vertex, false, 8) => InstanceIndex,
        (Stage::Pixel, false, 9) => FrontFacing,
        (Stage::Pixel, false, 10) => SampleIndex,
        (Stage::Pixel, true, 64) => Target(index),
        (Stage::Pixel, true, 65) => Depth,
        (Stage::Pixel, _, 66) => Coverage,
        (Stage::Pixel, true, 67) => DepthGreaterEqual,
        (Stage::Pixel, true, 68) => DepthLessEqual,
        (Stage::Pixel, true, 69) => StencilRef,
        (Stage::Pixel, false, 70) => InnerCoverage,
        _ => {
            return Err(ElementProblem::SystemValue {
                value,
                name: name_of(value),
                stage,
                output,
            });
        }
    }))
}

/// The `D3D_NAME` an `SV_` semantic name stands for, if it is a system value of the graphics pipeline.
fn number_of_name(name: &str) -> Option<u32> {
    Some(match name[3..].to_ascii_lowercase().as_str() {
        "position" => 1,
        "clipdistance" => 2,
        "culldistance" => 3,
        "rendertargetarrayindex" => 4,
        "viewportarrayindex" => 5,
        "vertexid" => 6,
        "primitiveid" => 7,
        "instanceid" => 8,
        "isfrontface" => 9,
        "sampleindex" => 10,
        "target" => 64,
        "depth" => 65,
        "coverage" => 66,
        "depthgreaterequal" => 67,
        "depthlessequal" => 68,
        "stencilref" => 69,
        "innercoverage" => 70,
        // the tessellation factors of a hull program's patch (the triangle's; the numbers differ by domain, the
        // semantic does not)
        "tessfactor" => 13,
        "insidetessfactor" => 14,
        _ => return None,
    })
}

/// `D3D_REGISTER_COMPONENT_TYPE`.
fn component(value: u32) -> Result<ScalarType, ElementProblem> {
    match value {
        1 => Ok(ScalarType::Uint),
        2 => Ok(ScalarType::Sint),
        3 => Ok(ScalarType::Float),
        _ => Err(ElementProblem::ComponentType { value }),
    }
}

/// The registers a stage's signature has in Direct3D 10 and later.
const MAX_SIGNATURE_REGISTERS: u32 = 32;

/// One signature element of `stage`, unpacked from its register: the label, the system value (the stored
/// `D3D_NAME`, or the `SV_` name when the compiler states it by name), the value type from the component type and the
/// lanes it occupies, and the register and lanes kept in provenance. `interp` is left for the caller, who has the
/// program's declarations.
pub fn varying(stage: Stage, output: bool, e: &Element<'_>) -> Result<Varying, Error> {
    let at = |problem: ElementProblem| Error::Element {
        element: format!("{}{}", e.name, e.index),
        problem,
    };
    if e.mask == 0 || e.mask > 15 || e.read_write_mask > 15 {
        return Err(at(ElementProblem::LaneMasks {
            mask: e.mask,
            read_write_mask: e.read_write_mask,
        }));
    }
    // a stage has 32 registers to pass values between stages (`D3D11_VS_OUTPUT_REGISTER_COUNT` and its kin); a system value
    // that takes none states `u32::MAX`
    if e.register != u32::MAX && e.register >= MAX_SIGNATURE_REGISTERS {
        return Err(at(ElementProblem::Register {
            register: e.register,
            limit: MAX_SIGNATURE_REGISTERS,
        }));
    }
    let named = e.name.len() > 3
        && e.name
            .as_bytes()
            .get(..3)
            .is_some_and(|p| p.eq_ignore_ascii_case(b"SV_"));
    let system = match system_from_number(stage, output, e.system_value, e.index).map_err(at)? {
        Some(s) => {
            if named && number_of_name(e.name) != Some(e.system_value) {
                return Err(at(ElementProblem::SemanticDisagrees {
                    stored: e.system_value,
                }));
            }
            Some(s)
        }
        // the compiler states some system values (SV_Target, SV_Depth) by name alone, with D3D_NAME undefined. A name
        // that is no graphics system value (SV_GroupID) is refused. A name that is one, but none this stage takes in
        // this direction, is refused for an output and is a plain varying for an input: the compiler did not make
        // it a system value, and some engines' pixel programs carry an `SV_InstanceID` that way.
        None if named => {
            let number = number_of_name(e.name).ok_or_else(|| {
                at(ElementProblem::SemanticUnmapped {
                    name: e.name.to_owned(),
                })
            })?;
            match system_from_number(stage, output, number, e.index) {
                Ok(system) => system,
                Err(_) if !output => None,
                Err(detail) => return Err(at(detail)),
            }
        }
        None => None,
    };
    let written = e.mask & !e.read_write_mask;
    Ok(Varying {
        label: Label {
            name: e.name.to_owned(),
            index: e.index,
        },
        system,
        value: Value {
            scalar: component(e.component_type).map_err(at)?,
            width: u8::try_from(e.mask.count_ones()).unwrap_or(4),
        },
        interp: None,
        used: if output { written } else { e.read_write_mask } != 0,
        provenance: VaryingProvenance {
            d3d: Some(D3dSignature {
                register: e.register,
                mask: e.mask,
                read_write_mask: e.read_write_mask,
            }),
        },
    })
}

/// A pixel input's interpolation from a program declaration's `D3D_INTERPOLATION_MODE` (the instruction token's
/// bits 11 to 14).
pub fn interpolation(mode: u32) -> Result<Interpolation, Error> {
    let (mode, sampling) = match mode {
        1 => (InterpMode::Flat, InterpSampling::Center),
        2 => (InterpMode::Perspective, InterpSampling::Center),
        3 => (InterpMode::Perspective, InterpSampling::Centroid),
        4 => (InterpMode::Linear, InterpSampling::Center),
        5 => (InterpMode::Linear, InterpSampling::Centroid),
        6 => (InterpMode::Perspective, InterpSampling::Sample),
        7 => (InterpMode::Linear, InterpSampling::Sample),
        _ => {
            return Err(Error::Unknown {
                what: "interpolation mode",
                value: mode,
            });
        }
    };
    Ok(Interpolation { mode, sampling })
}

/// `D3D_SHADER_VARIABLE_TYPE` of a constant-buffer scalar.
pub fn scalar_type(code: u32) -> Result<ScalarType, Error> {
    match code {
        1 => Ok(ScalarType::Bool),
        2 => Ok(ScalarType::Sint),
        3 => Ok(ScalarType::Float),
        19 => Ok(ScalarType::Uint),
        39 => Ok(ScalarType::Double),
        _ => Err(Error::Unknown {
            what: "constant scalar type",
            value: code,
        }),
    }
}

/// Bytes of one D3D constant register.
const REGISTER: u32 = 16;

/// The two D3D buffer layouts. Structured elements align to their scalar components, not constant registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    Constant,
    Structured,
}

impl Packing {
    /// Alignment of a structured value (including the tail padding of a structure in an array).
    pub fn alignment(self, ty: &ConstantType) -> u32 {
        match (self, ty) {
            (Self::Constant, _) => REGISTER,
            (_, ConstantType::Scalar(s)) => s.bytes(),
            (_, ConstantType::Vector(v)) => v.scalar.bytes(),
            (_, ConstantType::Matrix { scalar, .. }) => scalar.bytes(),
            (_, ConstantType::Array { element, .. }) => self.alignment(element),
            (_, ConstantType::Struct { members }) => {
                members.iter().map(|m| self.alignment(&m.ty)).max().unwrap_or(1)
            }
        }
    }

    /// A matrix's stored lines, with the packing of its containing buffer.
    pub fn matrix(
        self,
        scalar: ScalarType,
        rows: u32,
        columns: u32,
        major: Majorness,
    ) -> Result<ConstantType, Error> {
        let mut ty = matrix(scalar, rows, columns, major)?;
        if self == Self::Structured
            && let ConstantType::Matrix { stride, .. } = &mut ty
        {
            *stride = scalar.bytes().saturating_mul(match major {
                Majorness::Row => columns,
                Majorness::Column => rows,
            });
        }
        Ok(ty)
    }

    /// An array with a checked stride; structured arrays do not acquire constant-register padding.
    pub fn array(self, element: ConstantType, count: u32) -> Result<ConstantType, Error> {
        if count == 0 {
            return Err(Error::EmptyArray);
        }
        let alignment = self.alignment(&element);
        let stride = element
            .size()
            .ok_or(Error::ArrayElementSize)?
            .checked_next_multiple_of(alignment)
            .ok_or(Error::ArrayElementSize)?;
        let ty = ConstantType::Array {
            element: Box::new(element),
            count,
            stride,
        };
        ty.size().ok_or(Error::ArraySize)?;
        Ok(ty)
    }
}

fn lanes(what: &'static str, n: u32) -> Result<u8, Error> {
    u8::try_from(n)
        .ok()
        .filter(|n| (1..=4).contains(n))
        .ok_or(Error::LaneCount { what, value: n })
}

/// A vector of `width` lanes (1 to 4); a vector of one lane is a scalar.
pub fn vector(scalar: ScalarType, width: u32) -> Result<ConstantType, Error> {
    Ok(match lanes("vector width", width)? {
        1 => ConstantType::Scalar(scalar),
        width => ConstantType::Vector(Value { scalar, width }),
    })
}

/// A matrix under D3D packing: each line (a row for `Row`, a column for `Column`) starts a 16-byte register.
pub fn matrix(scalar: ScalarType, rows: u32, columns: u32, major: Majorness) -> Result<ConstantType, Error> {
    Ok(ConstantType::Matrix {
        scalar,
        rows: lanes("matrix rows", rows)?,
        columns: lanes("matrix columns", columns)?,
        major,
        stride: REGISTER,
    })
}

/// An array under D3D packing: each element starts a 16-byte register, and the last takes only its own bytes.
pub fn array(element: ConstantType, count: u32) -> Result<ConstantType, Error> {
    Packing::Constant.array(element, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, sv: u32, mask: u8, rw: u8) -> Element<'_> {
        Element {
            name,
            index: 0,
            system_value: sv,
            component_type: 3,
            register: 1,
            mask,
            read_write_mask: rw,
        }
    }

    /// Pixel inputs: the stored `D3D_NAME` or the `SV_` name makes the system value, a plain semantic has none.
    #[test]
    fn signature_elements_map_by_number_and_by_name() {
        let position = varying(Stage::Pixel, false, &element("SV_Position", 1, 15, 15)).unwrap();
        assert_eq!(position.system, Some(SystemValue::FragCoord));
        assert_eq!(position.value.width, 4);
        let uv = varying(Stage::Pixel, false, &element("TEXCOORD", 0, 0b1100, 0b0100)).unwrap();
        assert_eq!((uv.system, uv.value.width, uv.used), (None, 2, true));
        let d3d = uv.provenance.d3d.unwrap();
        assert_eq!((d3d.register, d3d.mask, d3d.read_write_mask), (1, 0b1100, 0b0100));
        // the compiler states SV_Target by name alone, in the output signature
        let mut target = element("SV_Target", 0, 15, 0);
        target.index = 2;
        let target = varying(Stage::Pixel, true, &target).unwrap();
        assert_eq!(target.system, Some(SystemValue::Target(2)));
        assert!(target.used);
        // an output's second mask is the lanes never written: all four never written means unused
        assert!(
            !varying(Stage::Pixel, true, &element("TEXCOORD", 0, 15, 15))
                .unwrap()
                .used
        );
        let front = varying(
            Stage::Pixel,
            false,
            &Element {
                component_type: 1,
                ..element("SV_IsFrontFace", 9, 1, 1)
            },
        )
        .unwrap();
        assert_eq!(front.system, Some(SystemValue::FrontFacing));
        assert_eq!(front.value.scalar, ScalarType::Uint);
        let clip = varying(Stage::Vertex, true, &element("SV_Position", 1, 15, 0)).unwrap();
        assert_eq!(clip.system, Some(SystemValue::ClipPosition));
    }

    /// An input whose D3D_NAME is undefined and whose semantic is a graphics system value that stage's inputs do not
    /// have is a plain varying (some engines' pixel programs carry `SV_InstanceID` so); one a stage does take by name keeps
    /// its system value.
    #[test]
    fn an_undefined_name_with_a_foreign_semantic_is_a_plain_input() {
        let instance = varying(Stage::Pixel, false, &element("SV_InstanceID", 0, 1, 1)).unwrap();
        assert_eq!(instance.system, None);
        let position = varying(Stage::Pixel, false, &element("SV_Position", 0, 15, 15)).unwrap();
        assert_eq!(position.system, Some(SystemValue::FragCoord));
    }

    #[test]
    fn what_the_table_does_not_map_is_refused_by_name() {
        let refusal = |stage, output, e: Element<'_>| varying(stage, output, &e).unwrap_err().to_string();
        // a tessellation factor, a name that disagrees with its number, an unknown SV_ name, a position written by a
        // pixel stage, an unknown component type and an empty mask
        assert!(
            refusal(Stage::Pixel, false, element("SV_TessFactor", 11, 1, 1)).contains("FINAL_*_TESSFACTOR")
        );
        assert!(refusal(Stage::Pixel, false, element("SV_IsFrontFace", 1, 1, 1)).contains("disagrees"));
        assert!(refusal(Stage::Pixel, false, element("SV_GroupID", 0, 1, 1)).contains("SV_GroupID"));
        // a graphics system value no pixel stage outputs, named with its D3D_NAME undefined
        assert!(refusal(Stage::Pixel, true, element("SV_InstanceID", 0, 1, 0)).contains("INSTANCE_ID"));
        assert!(refusal(Stage::Pixel, true, element("SV_Position", 1, 15, 0)).contains("POSITION"));
        let mut odd = element("TEXCOORD", 0, 15, 15);
        odd.component_type = 7;
        assert!(refusal(Stage::Pixel, false, odd).contains("component type 7"));
        assert!(refusal(Stage::Pixel, false, element("TEXCOORD", 0, 0, 0)).contains("lane masks"));
    }

    #[test]
    fn interpolation_modes_split_into_mode_and_sampling() {
        assert_eq!(
            interpolation(5).unwrap(),
            Interpolation {
                mode: InterpMode::Linear,
                sampling: InterpSampling::Centroid
            }
        );
        assert_eq!(interpolation(1).unwrap().mode, InterpMode::Flat);
        assert!(interpolation(0).is_err() && interpolation(8).is_err());
    }

    /// D3D packing: matrix lines and array elements start a 16-byte register, the last taking only its bytes.
    #[test]
    fn packing_strides_follow_the_registers() {
        let f = ScalarType::Float;
        assert_eq!(matrix(f, 3, 3, Majorness::Row).unwrap().size(), Some(44));
        assert_eq!(matrix(f, 4, 4, Majorness::Column).unwrap().size(), Some(64));
        let offsets = array(vector(f, 2).unwrap(), 3).unwrap();
        assert_eq!(offsets.size(), Some(40));
        let ConstantType::Array { stride, .. } = offsets else {
            panic!("an array")
        };
        assert_eq!(stride, 16);
        // a 64-byte matrix needs no padding between elements
        let ConstantType::Array { stride, .. } = array(matrix(f, 4, 4, Majorness::Row).unwrap(), 2).unwrap()
        else {
            panic!("an array")
        };
        assert_eq!(stride, 64);
        assert!(
            vector(f, 5).is_err()
                && matrix(f, 0, 1, Majorness::Row).is_err()
                && array(vector(f, 1).unwrap(), 0).is_err()
        );
        assert!(scalar_type(39).is_ok() && scalar_type(10).is_err());
    }
}
