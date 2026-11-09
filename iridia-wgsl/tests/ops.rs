//! Every operation written as WGSL that upstream naga reads back and validates: a pixel program that applies one operation to
//! the bits of its inputs, four lanes of `u32`, `i32` or `f32`. What each writes is held to the IR's definition by the WGSL
//! run on WARP (`iridia-dxbc`'s `exec_wgsl`) and, for the helper functions, by the module's own validation test.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        D3dSignature, Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value,
        Varying, VaryingProvenance,
    },
};
use iridia_wgsl::Profile;

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
            ..Default::default()
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

/// A unary operation `name` on the first input as `ty` (`f32` or `u32` or `i32`, four lanes), its result `result`.
fn unary(name: &str, ty: &str, result: &str) -> String {
    let cast = |t: &str, v: &str| {
        if t == "u32" {
            String::new()
        } else {
            format!("  %{v}: vec4<{t}> = bitcast %0\n")
        }
    };
    let a = if ty == "u32" { "0" } else { "1" };
    format!(
        "  %0: vec4<u32> = input 0.xyzw\n{}  %2: vec4<{result}> = {name} %{a}\n  %3: vec4<u32> = bitcast %2\n  output 0.xyzw = %3\n",
        cast(ty, "1")
    )
}

/// A binary operation on the first input as `ty` and the second as `by` (a shift's amount is unsigned).
fn binary_by(name: &str, ty: &str, by: &str, result: &str) -> String {
    format!(
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<{ty}> = bitcast %0\n  %3: vec4<{by}> = bitcast %1\n  %4: vec4<{result}> = {name} %2, %3\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
    )
}

fn binary(name: &str, ty: &str, result: &str) -> String {
    binary_by(name, ty, ty, result)
}

/// The text of each case written and validated by naga with every capability (naga's dialect).
fn check(cases: &[String]) {
    use naga::valid::Capabilities as C;
    let mut failed = Vec::new();
    for body in cases {
        let p = program(body);
        let wgsl = match iridia_wgsl::wgsl(&p, &Profile::NAGA) {
            Ok(w) => w,
            Err(e) => {
                failed.push(format!("{e}\n{body}"));
                continue;
            }
        };
        let read = naga::front::wgsl::parse_str(&wgsl).map_err(|e| e.emit_to_string(&wgsl));
        let valid = read.and_then(|m| {
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), C::all())
                .validate(&m)
                .map(|_| ())
                .map_err(|e| format!("{e:?}"))
        });
        if let Err(e) = valid {
            failed.push(format!("{e}\n{body}\n{wgsl}"));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
}

#[test]
fn unary_operations_are_written_as_valid_wgsl() {
    let mut cases = Vec::new();
    for op in [
        "fneg",
        "fabs",
        "ffract",
        "fsqrt",
        "finversesqrt",
        "fexp2",
        "flog2",
        "fsin",
        "fcos",
        "ftan",
        "fasin",
        "facos",
        "fatan",
        "fsinh",
        "fcosh",
        "ftanh",
        "fsign",
        "fround.floor",
        "fround.ceil",
        "fround.trunc",
        "fround.nearesteven",
        "fround.nearestaway",
        "quantize.8.nearestaway",
        "quantize.4.floor",
        "deriv.x.coarse",
        "deriv.y.fine",
        "deriv.x.any",
        "saturate.zero",
        "saturate.any",
        "isnan",
        "isinf",
    ] {
        let result = if matches!(op, "isnan" | "isinf") {
            "bool"
        } else {
            "f32"
        };
        if result == "bool" {
            cases.push(format!(
                "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<f32> = bitcast %0\n  %2: vec4<bool> = {op} %1\n  %3: vec4<u32> = select %2, %0, %0\n  output 0.xyzw = %3\n"
            ));
        } else {
            cases.push(unary(op, "f32", "f32"));
        }
    }
    for op in [
        "countones",
        "bitreverse",
        "firstbithigh",
        "firstbitlow",
        "not",
        "ineg",
    ] {
        cases.push(unary(op, "u32", "u32"));
    }
    cases.push(unary("firstbithighsigned", "i32", "i32"));
    cases.push(unary("firstbithigh", "i32", "i32"));
    cases.push(unary("ineg", "i32", "i32"));
    cases.push(unary("f32tof16", "f32", "u32"));
    cases.push(unary("f16tof32", "u32", "f32"));
    check(&cases);
}

#[test]
fn binary_operations_are_written_as_valid_wgsl() {
    let mut cases = Vec::new();
    for op in [
        "umulhigh",
        "uaddcarry",
        "usubborrow",
        "umin",
        "umax",
        "shl",
        "shru",
        "iadd",
        "isub",
        "imul",
        "and",
        "or",
        "xor",
    ] {
        cases.push(binary(op, "u32", "u32"));
    }
    cases.push(binary_by("shrs", "i32", "u32", "i32"));
    cases.push(binary_by("shru", "i32", "u32", "i32"));
    cases.push(binary_by("shrs", "u32", "u32", "u32"));
    for op in [
        "smulhigh",
        "sdiv.allones",
        "srem.allones",
        "sdiv.byone",
        "srem.byone",
        "smin",
        "smax",
    ] {
        cases.push(binary(op, "i32", "i32"));
    }
    for op in ["udiv.allones", "urem.allones", "udiv.byone", "urem.byone"] {
        cases.push(binary(op, "u32", "u32"));
    }
    for op in [
        "fadd",
        "fsub",
        "fmul.ieee",
        "fmul.zerowins",
        "fdiv",
        "frem",
        "fpow",
        "fmin.minnum",
        "fmax.minnum",
        "fmin.propagate",
        "fmax.any",
    ] {
        cases.push(binary(op, "f32", "f32"));
    }
    check(&cases);
}

#[test]
fn ternary_and_bit_insert_are_written_as_valid_wgsl() {
    let three = |name: &str, ty: &str| {
        format!(
            "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<{ty}> = bitcast %0\n  %4: vec4<{ty}> = {name} %3, %1, %2\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
        )
    };
    let cases = vec![
        three("bitextract.unsigned", "u32"),
        three("bitextract.signed", "i32"),
        three("bitextract.signed", "u32"),
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<u32> = bitinsert %0, %1, %2, %0\n  output 0.xyzw = %3\n".into(),
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<f32> = bitcast %0\n  %4: vec4<f32> = bitcast %1\n  %5: vec4<f32> = bitcast %2\n  %6: vec4<f32> = fma %3, %4, %5\n  %7: vec4<u32> = bitcast %6\n  output 0.xyzw = %7\n".into(),
    ];
    check(&cases);
}
