//! Each operation's lowering against its definition: a pixel program that applies one operation to the bits of its
//! inputs, run on edge and seeded random words in `iridia-exec-ir` (the IR's own definition) and, written as WGSL and
//! read back by naga's front end, in `iridia-exec`; the output bits must agree (two NaNs agree whatever their bits).

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_exec::{PixelShader, ProgramShader, Sample, Sampler};
use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        D3dSignature, Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value,
        Varying, VaryingProvenance,
    },
};

/// Three `vec4<u32>` inputs in registers 0 to 2, one `vec4<u32>` target.
fn program(body: &str) -> Program {
    let text = format!("iridia-ir 1\nfn main {{\n{body}  return\n}}\n");
    let entry = iridia_ir::text::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let varying = |name: &str, register: u32, system: Option<SystemValue>| Varying {
        label: Label {
            name: name.into(),
            index: if system.is_some() { 0 } else { register },
        },
        system,
        value: Value {
            scalar: ScalarType::Uint,
            width: 4,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance {
            d3d: Some(D3dSignature {
                register,
                mask: 0b1111,
                read_write_mask: 0,
            }),
        },
    };
    let p = Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: (0..3).map(|r| varying("TEXCOORD", r, None)).collect(),
            outputs: vec![varying("SV_Target", 0, Some(SystemValue::Target(0)))],
            resources: Vec::new(),
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::Dxbc,
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

/// The WGSL of `p` read back by naga's front end, as a program `iridia-exec` runs: the IR's interface, the entry
/// point's argument and result names and bindings put back as the interface gives them.
fn from_wgsl(p: &Program, wgsl: &str) -> iridia::Program {
    let mut module = naga_fork::front::wgsl::parse_str(wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let io = iridia::entry_io(&p.interface).unwrap();
    let ep = module.entry_points.first_mut().unwrap();
    for (a, item) in ep.function.arguments.iter_mut().zip(&io.arguments) {
        a.name = Some(item.name.clone());
        a.binding = Some(item.binding.clone());
    }
    let result = ep.function.result.as_ref().unwrap().ty;
    let mut t = module.types[result].clone();
    let naga_fork::TypeInner::Struct { members, .. } = &mut t.inner else {
        panic!("the result is not a structure")
    };
    for (m, item) in members.iter_mut().zip(&io.results) {
        m.name = Some(item.name.clone());
        m.binding = Some(item.binding.clone());
    }
    module.types.replace(result, t);
    iridia::Program {
        module,
        stage: Stage::Pixel,
        entry: "main".into(),
        interface: p.interface.clone(),
        bindings: Vec::new(),
        conventions: iridia::Conventions::D3D,
        modes: iridia::D3dModes::default(),
        provenance: iridia::Provenance {
            format: ShaderFormat::Dxbc,
            target: "fixture".into(),
            lifter: "iridia-naga through WGSL".into(),
            lifter_version: 0,
            source_hash: None,
            instruction_count: None,
        },
        notes: Vec::new(),
    }
}

struct NoTextures;
impl Sampler for NoTextures {
    fn sample(&mut self, _: Sample) -> Result<[f32; 4], String> {
        Err("no textures".into())
    }
}

/// Words that find the edges: zeros and ones, signs, f32 and f16 boundaries, NaNs and infinities, small counts.
const EDGES: [u32; 24] = [
    0,
    1,
    2,
    31,
    32,
    33,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0xffff_ffff,
    0xffff_fffe,
    0x0000_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7f80_0001,
    0x477f_e000,
    0x477f_f000,
    0x3880_0000,
    0x3380_0000,
    0x0000_7c00,
    0x0000_03ff,
];

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The inputs of run `n`: edge words in turn, then seeded random ones (some of them small, for counts).
fn inputs(n: u64) -> [[u32; 4]; 3] {
    let mut out = [[0; 4]; 3];
    for (k, w) in out.iter_mut().flatten().enumerate() {
        let r = mix(n.wrapping_mul(16).wrapping_add(k as u64));
        *w = if n < 64 {
            EDGES[(r % EDGES.len() as u64) as usize]
        } else if r & 7 == 0 {
            (r >> 8) as u32 & 63
        } else {
            (r >> 16) as u32
        };
    }
    out
}

/// Where the IR and the WGSL disagree on one of `runs` seeded inputs: the first such input and the two outputs.
fn disagreement(body: &str, runs: u64) -> Option<String> {
    let p = program(body);
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}\n{body}"));
    let shader = ProgramShader::new(from_wgsl(&p, &wgsl)).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let alike = |a: u32, b: u32| a == b || f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan();
    for n in 0..runs {
        let given = inputs(n);
        let b = iridia_exec_ir::run(
            &p,
            &mut iridia_exec_ir::Bound {
                inputs: &given,
                buffers: &[],
                state: &[],
                sampler: &mut |_| Err("no textures".into()),
                denormals: iridia_ir::Denormals::Preserve,
            },
        )
        .unwrap_or_else(|e| panic!("{e}\n{body}"));
        let b = b.outputs[0].unwrap_or([0; 4]);
        let c = shader.execute(&given, &[], &mut NoTextures).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
        let c = c.outputs.first().copied().unwrap_or([0; 4]);
        if !b.iter().zip(&c).all(|(&x, &y)| alike(x, y)) {
            return Some(format!("{given:x?}: the IR gives {b:x?}, the WGSL {c:x?}\n{wgsl}"));
        }
    }
    None
}

/// A unary operation `name` on the first input as `ty` (`f32` or `u32` or `i32`, four lanes), its result `result`.
fn unary(name: &str, ty: &str, result: &str) -> String {
    let cast = |t: &str, v: &str| if t == "u32" { String::new() } else { format!("  %{v}: vec4<{t}> = bitcast %0\n") };
    let a = if ty == "u32" { "0" } else { "1" };
    format!(
        "  %0: vec4<u32> = input 0.xyzw\n{}  %2: vec4<{result}> = {name} %{a}\n  %3: vec4<u32> = bitcast %2\n  output 0.xyzw = %3\n",
        cast(ty, "1")
    )
}

/// A binary operation on the first two inputs as `ty`.
fn binary(name: &str, ty: &str, result: &str) -> String {
    binary_by(name, ty, ty, result)
}

/// A binary operation on the first input as `ty` and the second as `by` (a shift's amount is unsigned).
fn binary_by(name: &str, ty: &str, by: &str, result: &str) -> String {
    format!(
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<{ty}> = bitcast %0\n  %3: vec4<{by}> = bitcast %1\n  %4: vec4<{result}> = {name} %2, %3\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
    )
}

const RUNS: u64 = 4096;

fn check(cases: &[String]) {
    let failed: Vec<String> = cases.iter().filter_map(|c| disagreement(c, RUNS)).collect();
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
}

#[test]
fn unary_operations_lower_to_their_definitions() {
    let mut cases = Vec::new();
    for op in ["ftan", "fasin", "facos", "fatan", "fsinh", "fcosh", "ftanh", "fsign"] {
        cases.push(unary(op, "f32", "f32"));
    }
    for op in ["countones", "bitreverse", "firstbithigh", "firstbitlow", "not"] {
        cases.push(unary(op, "u32", "u32"));
    }
    cases.push(unary("firstbithighsigned", "i32", "i32"));
    cases.push(unary("firstbithigh", "i32", "i32"));
    cases.push(unary("f32tof16", "f32", "u32"));
    cases.push(unary("f16tof32", "u32", "f32"));
    check(&cases);
}

#[test]
fn binary_operations_lower_to_their_definitions() {
    let mut cases = Vec::new();
    for op in ["umulhigh", "uaddcarry", "usubborrow", "umin", "umax", "shl", "shru"] {
        cases.push(binary(op, "u32", "u32"));
    }
    cases.push(binary_by("shrs", "i32", "u32", "i32"));
    for op in ["smulhigh", "sdiv.allones", "srem.allones", "sdiv.byone", "srem.byone"] {
        cases.push(binary(op, "i32", "i32"));
    }
    for op in ["udiv.allones", "urem.allones", "udiv.byone", "urem.byone"] {
        cases.push(binary(op, "u32", "u32"));
    }
    cases.push(binary("frem", "f32", "f32"));
    check(&cases);
}

#[test]
fn ternary_and_bit_insert_lower_to_their_definitions() {
    let three = |name: &str, ty: &str| {
        format!(
            "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<{ty}> = bitcast %0\n  %4: vec4<{ty}> = {name} %3, %1, %2\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
        )
    };
    let cases = vec![
        three("bitextract.unsigned", "u32"),
        three("bitextract.signed", "i32"),
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<u32> = bitinsert %0, %1, %2, %0\n  output 0.xyzw = %3\n".into(),
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<f32> = bitcast %0\n  %4: vec4<f32> = bitcast %1\n  %5: vec4<f32> = bitcast %2\n  %6: vec4<f32> = fma %3, %4, %5\n  %7: vec4<u32> = bitcast %6\n  output 0.xyzw = %7\n".into(),
    ];
    check(&cases);
}
