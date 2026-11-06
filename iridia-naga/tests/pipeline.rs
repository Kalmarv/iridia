//! Programs written for one pipeline ([`iridia_naga::wgsl_for`]): a vertex program's outputs where the pixel program
//! it feeds reads them, read back from the written WGSL.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying,
        VaryingProvenance,
    },
};
use iridia_naga::{BindingPlan, Pipeline, Profile, wgsl_for};

fn varying(name: &str, index: u32, value: (ScalarType, u8), system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: Value {
            scalar: value.0,
            width: value.1,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

fn program(stage: Stage, text: &str, inputs: Vec<Varying>, outputs: Vec<Varying>) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let p = Program {
        stage,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage,
            inputs,
            outputs,
            resources: Vec::new(),
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::D3d9,
            target: "fixture".into(),
            lifter: "text".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    };
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    p
}

const F2: (ScalarType, u8) = (ScalarType::Float, 2);
const F3: (ScalarType, u8) = (ScalarType::Float, 3);
const F4: (ScalarType, u8) = (ScalarType::Float, 4);

/// A vertex program writing the position, `TEXCOORD1` (two lanes), `COLOR0` and `TEXCOORD0`, in that order.
fn vertex() -> Program {
    program(
        Stage::Vertex,
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  output 0.xyzw = %0
  %1: vec2<f32> = input 0.xy
  output 1.xy = %1
  output 2.xyzw = %0
  %2: vec3<f32> = input 0.yzw
  output 3.xyz = %2
  return
}
",
        vec![varying("POSITION", 0, F4, None)],
        vec![
            varying("POSITION", 0, F4, Some(SystemValue::ClipPosition)),
            varying("TEXCOORD", 1, F2, None),
            varying("COLOR", 0, F4, None),
            varying("TEXCOORD", 0, F3, None),
        ],
    )
}

/// A pixel program reading `color0` (another case) and `TEXCOORD1` with four lanes, in the other order, and `extra`.
fn pixel(extra: Option<Varying>) -> Program {
    let mut inputs = vec![varying("color", 0, F4, None), varying("TEXCOORD", 1, F4, None)];
    let mut text = String::from(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = input 1.xyzw
  %2: vec4<f32> = fadd %0, %1
",
    );
    if let Some(extra) = extra {
        inputs.push(extra);
        text.push_str(
            "  %3: vec4<f32> = input 2.xyzw\n  %4: vec4<f32> = fadd %2, %3\n  output 0.xyzw = %4\n",
        );
    } else {
        text.push_str("  output 0.xyzw = %2\n");
    }
    text.push_str("  return\n}\n");
    program(
        Stage::Pixel,
        &text,
        inputs,
        vec![varying("SV_Target", 0, F4, Some(SystemValue::Target(0)))],
    )
}

/// Each location the entry point of `wgsl` passes (a vertex program's result members, a pixel program's arguments),
/// with its type, in location order; the module validates.
fn locations(wgsl: &str) -> Vec<(u32, naga::TypeInner)> {
    let module = naga::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{wgsl}"));
    let entry = &module.entry_points[0];
    let mut out = Vec::new();
    let mut push = |binding: &Option<naga::Binding>, ty: naga::Handle<naga::Type>| {
        if let Some(naga::Binding::Location { location, .. }) = binding {
            out.push((*location, module.types[ty].inner.clone()));
        }
    };
    match entry.stage {
        naga::ShaderStage::Vertex => {
            let result = entry.function.result.as_ref().unwrap();
            let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
                panic!("{wgsl}");
            };
            for m in members {
                push(&m.binding, m.ty);
            }
        }
        _ => {
            for a in &entry.function.arguments {
                push(&a.binding, a.ty);
            }
        }
    }
    out.sort_by_key(|(l, _)| *l);
    out
}

fn written(p: &Program, feeds: Option<&Interface>) -> Result<iridia_naga::Written, iridia_naga::Error> {
    let pipeline = Pipeline {
        plan: BindingPlan::d3d(&p.interface).unwrap(),
        feeds,
    };
    wgsl_for(p, &Profile::BROWSER, &pipeline)
}

#[test]
fn a_vertex_program_written_for_its_pixel_program_passes_what_it_reads_where_it_reads_it() {
    let (vs, ps) = (vertex(), pixel(None));
    // each program on its own puts its varyings at locations in its own order: the two do not link
    let alone = locations(&iridia_naga::wgsl(&vs, &Profile::BROWSER).unwrap());
    let read = locations(&written(&ps, None).unwrap().text);
    assert_ne!(alone[..read.len()], read[..], "the fixture needs a link");

    let linked = written(&vs, Some(&ps.interface)).unwrap();
    let passed = locations(&linked.text);
    // what the pixel program reads, at its locations and types; TEXCOORD0, unread, after them
    assert_eq!(passed[..read.len()], read[..], "{}", linked.text);
    assert_eq!(
        passed[read.len()..],
        [(
            2,
            naga::TypeInner::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar::F32
            }
        )],
        "{}",
        linked.text
    );
    assert_eq!(linked.next_location, 3);
    // TEXCOORD1's two lanes padded to the four read, with zeros
    assert!(linked.text.contains("vec4<f32>(_e"), "{}", linked.text);
    assert_eq!(linked.notes.len(), 1, "{:?}", linked.notes);
    assert!(
        linked.notes[0].contains("more lanes than written"),
        "{:?}",
        linked.notes
    );
}

#[test]
fn a_pixel_input_no_vertex_output_carries_is_refused_by_name() {
    let ps = pixel(Some(varying("TEXCOORD", 5, F4, None)));
    let e = written(&vertex(), Some(&ps.interface)).unwrap_err();
    assert!(e.0.contains("reads TEXCOORD5"), "{e}");
    assert!(e.0.contains("POSITION0") || e.0.contains("TEXCOORD1"), "{e}");
}

#[test]
fn an_output_read_as_another_scalar_kind_passes_the_same_bits() {
    const U4: (ScalarType, u8) = (ScalarType::Uint, 4);
    let ps = program(
        Stage::Pixel,
        "iridia-ir 1
fn main {
  %0: vec4<u32> = input 0.xyzw
  output 0.xyzw = %0
  return
}
",
        vec![varying("COLOR", 0, U4, None)],
        vec![varying("SV_Target", 0, U4, Some(SystemValue::Target(0)))],
    );
    let read = locations(&written(&ps, None).unwrap().text);
    let linked = written(&vertex(), Some(&ps.interface)).unwrap();
    assert_eq!(locations(&linked.text)[..1], read[..], "{}", linked.text);
    assert!(linked.text.contains("bitcast<vec4<u32>>"), "{}", linked.text);
    assert!(
        linked.notes.iter().any(|n| n.contains("another scalar type")),
        "{:?}",
        linked.notes
    );
}

#[test]
fn only_a_vertex_program_feeds_another() {
    let ps = pixel(None);
    let e = written(&ps, Some(&ps.interface)).unwrap_err();
    assert!(e.0.contains("feeds no consumer"), "{e}");
}

#[test]
fn an_output_read_with_fewer_lanes_passes_its_first_ones() {
    let ps = program(
        Stage::Pixel,
        "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  output 0.xy = %0
  return
}
",
        vec![varying("TEXCOORD", 0, F2, None)],
        vec![varying("SV_Target", 0, F4, Some(SystemValue::Target(0)))],
    );
    let read = locations(&written(&ps, None).unwrap().text);
    let linked = written(&vertex(), Some(&ps.interface)).unwrap();
    assert_eq!(locations(&linked.text)[..1], read[..], "{}", linked.text);
    assert!(
        linked.notes.iter().any(|n| n.contains("fewer lanes")),
        "{:?}",
        linked.notes
    );
}
