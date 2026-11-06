//! Programs in IR text lowered to naga and written as WGSL: what each construct and rule becomes.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, Resource, ResourceId, ResourceKind, ResourceProvenance, ScalarType, ShaderFormat,
        Stage, StageFacts, StateParam, SystemValue, Value, Varying, VaryingProvenance,
    },
};

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

/// A pixel program of the text's function, with `TEXCOORD` inputs and `SV_Target` outputs of the shapes given.
fn program(
    text: &str,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let n = |i: usize| u32::try_from(i).unwrap();
    let p = Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("TEXCOORD", n(i), *v, None))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("SV_Target", n(i), *v, Some(SystemValue::Target(n(i)))))
                .collect(),
            resources,
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

const F4: (ScalarType, u8) = (ScalarType::Float, 4);

#[test]
fn masked_loads_and_stores_lower_to_swizzles_and_merges() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.yw
  output 0.xz = %0
  %1: f32 = input 0.x
  output 0.w = %1
  output 0.x = %1
  return
}
",
        &[F4],
        &[F4],
        Vec::new(),
    );
    // the input (a, b, c, d): (b, 0, d, 0), then (b, 0, d, a), then (a, 0, d, a); a lane never stored is zero
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    let body = "    let _e1 = TEXCOORD0_.yw;
    let _e7 = vec4<f32>(_e1.x, vec4<f32>().y, _e1.y, vec4<f32>().w);
    let _e13 = vec4<f32>(_e7.x, _e7.y, _e7.z, TEXCOORD0_.x);
    return Out(vec4<f32>(TEXCOORD0_.x, _e13.y, _e13.z, _e13.w));
";
    assert!(wgsl.contains(body), "{wgsl}");
}

#[test]
fn a_zero_wins_multiplication_of_vectors_validates() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = input 1.xyzw
  %2: vec4<f32> = fmul.zerowins %0, %1
  output 0.xyzw = %2
  %3: f32 = input 0.x
  %4: f32 = fmul.zerowins %3, %3
  output 1.x = %4
  return
}
",
        &[F4, F4],
        &[F4, (ScalarType::Float, 1)],
        Vec::new(),
    );
    // each lane: zero where either operand is zero, the product otherwise; `|` joins the bool vectors
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(
        wgsl.contains("select((TEXCOORD0_ * TEXCOORD1_), _e4, ((TEXCOORD0_ == _e4) | (TEXCOORD1_ == _e4)))"),
        "{wgsl}"
    );
}

#[test]
fn unspecialised_pipeline_state_is_refused_by_name() {
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::Projected(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "projected0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = program(
        "iridia-ir 1
fn main {
  %0: bool = state r0
  %1: f32 = input 0.x
  if %0 {
    %2: f32 = fneg %1
    yield %2
  } else {
    yield %1
  } -> %3: f32
  output 0.x = %3
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        vec![state],
    );
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap_err();
    assert!(e.0.contains("specialise"), "{e}");
}

#[test]
fn a_lane_used_twice_is_computed_once() {
    // %2 is a lane of a sum nothing else uses: naga gives an access no `let` of its own, so unnamed the sum would be
    // written at both uses
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = fadd %0, %0
  %2: f32 = %1.x
  %3: vec2<f32> = compose %2, %2
  output 0.xy = %3
  return
}
",
        &[F4],
        &[(ScalarType::Float, 2)],
        Vec::new(),
    );
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert_eq!(wgsl.matches(" + ").count(), 1, "{wgsl}");
    assert!(wgsl.contains("let v2_ = (TEXCOORD0_ + TEXCOORD0_).x;"), "{wgsl}");
}

#[test]
fn a_capability_the_profile_lacks_is_refused_before_lowering() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f64 = convert.undef %0
  %2: f64 = fadd %1, %1
  %3: f32 = convert.undef %2
  output 0.x = %3
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        Vec::new(),
    );
    let needs = iridia_naga::requirements(&p);
    assert_eq!(needs.len(), 1, "{needs:?}");
    assert_eq!(needs[0].capability, naga::valid::Capabilities::FLOAT64);
    // browser WGSL has no f64: the preflight names the value, before any naga module is built
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: value %1 is f64"),
        "{e}"
    );
    // naga's dialect has it, and naga's validator, run with the profile's capabilities alone, agrees
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(wgsl.contains("f64"), "{wgsl}");
    // a module that needs more than its profile allows is rejected by naga's validator itself
    let plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
    let module = iridia_naga::module(&p, &plan).unwrap();
    assert!(iridia_naga::validate(&module, &iridia_naga::Profile::BROWSER).is_err());
}
