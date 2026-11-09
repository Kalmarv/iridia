//! Each operation's lowering against its definition: a pixel program that applies one operation to the bits of its
//! inputs, run on edge and seeded random words in `iridia-exec-ir` (the IR's own definition). The definition is held in
//! two ways: a committed digest of its outputs for every case (`fixtures/ops.digests`), which fails when the executor's
//! answers change, and the program written as WGSL by both writers, run on Direct3D 12's WARP through `tools.wgsl_run`
//! (where it is configured), whose output bits must agree (two NaNs agree whatever their bits). Upstream naga reads each
//! written text and validates it either way.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

mod inputs;
#[path = "support/warp.rs"]
mod warp;
use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        D3dBinding, D3dClass, D3dSignature, Interface, Label, Resource, ResourceId, ResourceKind,
        ResourceProvenance, SampleType, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, TextureDim,
        Value, Varying, VaryingProvenance,
    },
};

/// Three `vec4<u32>` inputs in registers 0 to 2, one `vec4<u32>` target.
fn program(body: &str) -> Program {
    program_with(body, Vec::new())
}

/// [`program`] with these resources.
fn program_with(body: &str, resources: Vec<Resource>) -> Program {
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
            resources,
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

/// What the IR executor gives for each of `runs` inputs: the definition of the program.
fn ir_outputs(p: &Program, runs: u64) -> Vec<[u32; 4]> {
    (0..runs)
        .map(|n| {
            let given = inputs(n);
            let run = iridia_exec_ir::run(
                p,
                &mut iridia_exec_ir::Bound {
                    inputs: &given,
                    buffers: &[],
                    state: &[],
                    sampler: &mut |_| Err("no textures".into()),
                    denormals: iridia_ir::Denormals::Preserve,
                },
            )
            .unwrap_or_else(|e| panic!("{e}\n{p:?}"));
            run.outputs[0].unwrap_or([0; 4])
        })
        .collect()
}

/// A result word as a comparison reads it: every NaN of a float result is one NaN, since two NaNs agree whatever their bits.
fn canonical(float: bool, word: u32) -> u32 {
    if float && f32::from_bits(word).is_nan() {
        0x7fc0_0000
    } else {
        word
    }
}

/// FNV-1a of the outputs' words (a float result's NaNs made one), the digest the committed table holds.
fn digest(float: bool, outputs: &[[u32; 4]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for word in outputs.iter().flatten() {
        for byte in canonical(float, *word).to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Whether a float is zero or denormal: WARP flushes these, so a result in this range agrees with any other.
fn tiny(word: u32) -> bool {
    word & 0x7f80_0000 == 0
}

/// How far apart two finite floats are, as a fraction of the larger of their magnitudes and one: the error of a result
/// whose size is its own and of one near zero, which an absolute error of a thousandth of a unit is not.
fn relative_gap(a: u32, b: u32) -> f64 {
    let (x, y) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
    (x - y).abs() / x.abs().max(y.abs()).max(1.0)
}

/// Whether two results agree: the same bits; for a float result also two NaNs (whatever their bits), two zeros or
/// denormals (WARP flushes denormals, the IR keeps them), and finite floats within `tolerance` of each other
/// ([`relative_gap`]).
fn within(float: bool, tolerance: f64, a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    if !float {
        return false;
    }
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    if x.is_nan() && y.is_nan() || tiny(a) && tiny(b) {
        return true;
    }
    x.is_finite() && y.is_finite() && relative_gap(a, b) <= tolerance
}

/// How far WARP's answers to an operation may be from the IR's. WARP flushes denormals, does not fuse a multiply-add
/// the IR fuses, and computes the transcendental functions to its own accuracy (a sinh past 89 and a tan of a huge
/// angle come out nothing like the IR's). What this holds the lowering to is the operation: a lane whose float operands
/// include a denormal or a magnitude past `limit` is not compared, and the others agree to `tolerance`.
#[derive(Clone, Copy)]
struct Slack {
    tolerance: f64,
    limit: Option<f32>,
    /// The operands are a multiply-add's: a lane where fusing it changes the result is not compared either.
    fused: bool,
}

impl Slack {
    /// The same bits.
    const EXACT: Slack = Slack {
        tolerance: 0.0,
        limit: None,
        fused: false,
    };

    /// Whether a lane with these float operands is left out.
    fn skips(self, operands: &[u32]) -> bool {
        let Some(limit) = self.limit else { return false };
        let float = |w: u32| f32::from_bits(w);
        if operands
            .iter()
            .any(|&w| tiny(w) && w & 0x7fff_ffff != 0 || float(w).abs() > limit)
        {
            return true;
        }
        if let (true, &[a, b, c]) = (self.fused, operands) {
            let (fused, plain) = (
                float(a).mul_add(float(b), float(c)),
                float(a) * float(b) + float(c),
            );
            // a product that is denormal is flushed by WARP before the addition
            let product = f64::from(float(a)) * f64::from(float(b));
            let underflows = product != 0.0 && product.abs() < f64::from(f32::MIN_POSITIVE);
            return underflows || fused.to_bits() != plain.to_bits() && !(fused.is_nan() && plain.is_nan());
        }
        false
    }
}

/// Where WARP, run on the WGSL `wgsl` of `p`, disagrees with the IR's answers `want` to the same inputs: the first
/// such input and the two outputs. `None` where the harness is not configured.
fn warp_disagreement(
    warp: &mut warp::Warp,
    wgsl: &str,
    want: &[[u32; 4]],
    arity: usize,
    float: bool,
    slack: Slack,
) -> Option<String> {
    let given: Vec<Vec<[u32; 4]>> = (0..want.len() as u64)
        .map(|n| inputs(n).into_iter().take(arity).collect())
        .collect();
    let got = warp.pixel(wgsl, &given).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let mut differ = Vec::new();
    let mut widest = 0f64;
    for ((given, want), got) in given.iter().zip(want).zip(got) {
        let got = got.unwrap_or([0; 4]);
        for lane in 0..4 {
            let operands: Vec<u32> = given.iter().map(|g| g[lane]).collect();
            if slack.skips(&operands) {
                continue;
            }
            let (x, y) = (want[lane], got[lane]);
            if float
                && !(tiny(x) && tiny(y))
                && f32::from_bits(x).is_finite()
                && f32::from_bits(y).is_finite()
            {
                widest = widest.max(relative_gap(x, y));
            }
            if !within(float, slack.tolerance, x, y) {
                differ.push(format!("{operands:x?}: the IR gives {x:x}, WARP {y:x}"));
            }
        }
    }
    if differ.is_empty() {
        return None;
    }
    let shown = differ.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ");
    Some(format!(
        "{} lanes differ (widest finite gap {widest:e}), the first:\n  {shown}\n{wgsl}",
        differ.len()
    ))
}

/// The WGSL of `p` in a profile, read by upstream naga and validated as the profile states.
fn written(p: &Program, profile: &iridia_naga::Profile) -> String {
    let text = iridia_naga::wgsl(p, profile).unwrap_or_else(|e| panic!("{e}\n{p:?}"));
    let module =
        naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    iridia_naga::validate(&module, profile).unwrap_or_else(|e| panic!("{e}\n{text}"));
    text
}

/// Every text WARP is given for `p`: this crate's WGSL and `iridia-wgsl`'s, both for a browser.
fn warp_texts(p: &Program) -> [String; 2] {
    [
        written(p, &iridia_naga::Profile::BROWSER),
        iridia_wgsl::wgsl(p, &iridia_wgsl::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}\n{p:?}")),
    ]
}

/// A reduced specialised pixel program: its constants already contain the pinned feature words.
fn specialised_fixture(text: &str) -> Program {
    let mut p = program("");
    p.entry = iridia_ir::text::parse(text).unwrap();
    p.interface.inputs.clear();
    p.interface.outputs[0].value = Value {
        scalar: ScalarType::Float,
        width: 1,
    };
    iridia_ir::validate(&p).unwrap();
    p
}

/// What the fixtures' domain operations give: `log2(0)` is -infinity, `1 / rsqrt(0)` is zero, `rsqrt(-1)` is a NaN and `1 / 0`
/// is infinity.
const EXPECT_LOG_ZERO: u32 = 0xff80_0000;
const EXPECT_INVERSE_ZERO: u32 = 0;
const EXPECT_INVERSE_NEGATIVE: u32 = 0x7fc0_0000;
const EXPECT_DIVIDE_ZERO: u32 = 0x7f80_0000;

/// A specialised fixture whose one output word is `expect` (two NaNs agree whatever their bits): the IR executor gives
/// it, and WARP agrees on the WGSL of both writers in both profiles.
fn check_literal_fixture(p: &Program, expect: u32) {
    // a pixel program is run four invocations at a time
    let want = ir_outputs(p, 4);
    assert!(
        want.iter().all(|w| within(true, 0.0, w[0], expect)),
        "the IR gives {:x?}, expected {expect:x}",
        want[0]
    );
    let mut warp = warp::Warp::shared();
    for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
        let text = written(p, &profile);
        if let Some(warp) = warp.as_mut() {
            assert_eq!(warp_disagreement(warp, &text, &want, 0, true, Slack::EXACT), None);
        }
    }
    if let Some(warp) = warp.as_mut() {
        let text = iridia_wgsl::wgsl(p, &iridia_wgsl::Profile::BROWSER).unwrap();
        assert_eq!(warp_disagreement(warp, &text, &want, 0, true, Slack::EXACT), None);
    }
}

#[test]
fn specialised_literal_domain_operations_stay_runtime_in_both_profiles() {
    for (text, expect) in [
        (include_str!("fixtures/log_zero.ir"), EXPECT_LOG_ZERO),
        (include_str!("fixtures/inverse_zero.ir"), EXPECT_INVERSE_ZERO),
        (
            include_str!("fixtures/inverse_negative.ir"),
            EXPECT_INVERSE_NEGATIVE,
        ),
        (include_str!("fixtures/divide_zero.ir"), EXPECT_DIVIDE_ZERO),
    ] {
        check_literal_fixture(&specialised_fixture(text), expect);
    }
}

#[test]
fn nonfinite_literal_bits_and_signed_zero_survive_the_written_runtime_binding() {
    let mut p =
        specialised_fixture("iridia-ir 1\nfn main {\n  %0: f32 = f32(0.0)\n  output 0.x = %0\n  return\n}\n");
    for bits in [
        0x8000_0000u32,
        1,
        f32::MAX.to_bits(),
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x7fc0_1234,
    ] {
        p.entry.values[0].op = iridia_ir::Op::Literal([u64::from(bits), 0, 0, 0]);
        check_literal_fixture(&p, bits);
    }
}

/// A chain of `depth` additions and exclusive ors of input 1 on input 0, as IR text, and what it gives for the first
/// `runs` inputs worked out here: values 2, 4, 6... add input 1 and 3, 5, 7... xor it.
fn chain(depth: usize, runs: u64) -> (Program, Vec<[u32; 4]>) {
    let mut body = String::from("  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n");
    let mut previous = 0;
    for n in 2..depth + 2 {
        let op = if n % 2 == 0 { "iadd" } else { "xor" };
        body.push_str(&format!("  %{n}: vec4<u32> = {op} %{previous}, %1\n"));
        previous = n;
    }
    body.push_str(&format!("  output 0.xyzw = %{previous}\n"));
    let want = (0..runs)
        .map(|n| {
            let given = inputs(n);
            let mut out = given[0];
            for step in 2..depth + 2 {
                for (o, b) in out.iter_mut().zip(given[1]) {
                    *o = if step % 2 == 0 { o.wrapping_add(b) } else { *o ^ b };
                }
            }
            out
        })
        .collect();
    (program(&body), want)
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(32))]
    #[test]
    fn binding_deep_chains_preserves_each_inputs_bits(depth in 24usize..320) {
        let (p, want) = chain(depth, 32);
        proptest::prop_assert_eq!(ir_outputs(&p, 32), want);
        // both writers' text is read and validated; WARP runs a few depths of its own, below
        for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
            written(&p, &profile);
        }
        iridia_wgsl::wgsl(&p, &iridia_wgsl::Profile::BROWSER).unwrap();
    }
}

#[test]
fn deep_chains_run_on_warp_as_the_ir_runs_them() {
    let mut warp = warp::Warp::shared();
    for depth in [24, 160, 319] {
        let (p, want) = chain(depth, 32);
        assert_eq!(ir_outputs(&p, 32), want);
        if let Some(warp) = warp.as_mut() {
            for text in warp_texts(&p) {
                assert_eq!(
                    warp_disagreement(warp, &text, &want, 3, false, Slack::EXACT),
                    None,
                    "depth {depth}"
                );
            }
        }
    }
}

/// One operation's program: a name for the digest table, the body, whether the result is a float (its NaNs are one) and
/// how many units in the last place WARP's answer may differ by (the operations WARP does not give bit for bit).
struct Case {
    name: String,
    body: String,
    float: bool,
    slack: Slack,
}

/// A case; `float` is whether the result is `f32`.
fn case(name: impl Into<String>, body: impl Into<String>, float: bool) -> Case {
    Case {
        name: name.into(),
        body: body.into(),
        float,
        slack: Slack::EXACT,
    }
}

impl Case {
    /// WARP's answers may be [`Slack`] from the IR's: operands up to `limit`, results within `tolerance`.
    fn near(mut self, limit: f32, tolerance: f64) -> Case {
        self.slack = Slack {
            tolerance,
            limit: Some(limit),
            fused: false,
        };
        self
    }

    /// The operands are a multiply-add's.
    fn fused(mut self) -> Case {
        self.slack.fused = true;
        self
    }
}

/// A unary operation `name` on the first input as `ty` (`f32` or `u32` or `i32`, four lanes), its result `result`.
fn unary(name: &str, ty: &str, result: &str) -> Case {
    let cast = |t: &str, v: &str| {
        if t == "u32" {
            String::new()
        } else {
            format!("  %{v}: vec4<{t}> = bitcast %0\n")
        }
    };
    let a = if ty == "u32" { "0" } else { "1" };
    let body = format!(
        "  %0: vec4<u32> = input 0.xyzw\n{}  %2: vec4<{result}> = {name} %{a}\n  %3: vec4<u32> = bitcast %2\n  output 0.xyzw = %3\n",
        cast(ty, "1")
    );
    case(format!("{name} {ty} -> {result}"), body, result == "f32")
}

/// A binary operation on the first two inputs as `ty`.
fn binary(name: &str, ty: &str, result: &str) -> Case {
    binary_by(name, ty, ty, result)
}

/// A binary operation on the first input as `ty` and the second as `by` (a shift's amount is unsigned).
fn binary_by(name: &str, ty: &str, by: &str, result: &str) -> Case {
    let body = format!(
        "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<{ty}> = bitcast %0\n  %3: vec4<{by}> = bitcast %1\n  %4: vec4<{result}> = {name} %2, %3\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
    );
    case(format!("{name} {ty} {by} -> {result}"), body, result == "f32")
}

/// The committed digests of the IR executor's outputs (`fixtures/ops.digests`): a case's name, a tab, the digest in hex.
const DIGESTS: &str = include_str!("fixtures/ops.digests");

const RUNS: u64 = 4096;

/// Each case: upstream naga reads and validates its WGSL in both profiles, the IR executor's outputs on [`RUNS`] inputs
/// are the digest committed for it, and WARP, given the WGSL of both writers, gives the same bits (to the case's slack).
/// The digests found are printed in the table's own form where any differs.
fn check(cases: &[Case]) {
    let mut warp = warp::Warp::shared();
    let mut failures = Vec::new();
    let mut found = String::new();
    for c in cases {
        let p = program(&c.body);
        for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
            written(&p, &profile);
        }
        let want = ir_outputs(&p, RUNS);
        let d = digest(c.float, &want);
        found.push_str(&format!("{}\t{d:016x}\n", c.name));
        let committed = DIGESTS
            .lines()
            .find_map(|l| l.strip_prefix(c.name.as_str())?.strip_prefix('\t'));
        if committed != Some(format!("{d:016x}").as_str()) {
            failures.push(format!(
                "{}: the IR executor's digest is {d:016x}, committed {committed:?}",
                c.name
            ));
        }
        if let Some(warp) = warp.as_mut() {
            for text in warp_texts(&p) {
                if let Some(why) = warp_disagreement(warp, &text, &want, 3, c.float, c.slack) {
                    failures.push(format!("{}: {why}", c.name));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}\n\nthe digests found:\n{found}",
        failures.join("\n\n")
    );
}

#[test]
fn expanded_kernels_are_shared_across_repeated_uses() {
    let chain = |count: u32| {
        let mut body = String::from("  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<f32> = bitcast %0\n");
        let mut previous = 1;
        for n in 0..count {
            let half = 2 + n * 2;
            let float = half + 1;
            body.push_str(&format!(
                "  %{half}: vec4<u32> = f32tof16 %{previous}\n  %{float}: vec4<f32> = f16tof32 %{half}\n"
            ));
            previous = float;
        }
        let result = 2 + count * 2;
        body.push_str(&format!(
            "  %{result}: vec4<u32> = bitcast %{previous}\n  output 0.xyzw = %{result}\n"
        ));
        program(&body)
    };
    // rounding to a half is idempotent: a chain of any length gives what one link gives
    let one = ir_outputs(&chain(1), 64);
    let mut warp = warp::Warp::shared();
    for count in [2, 16, 64] {
        let p = chain(count);
        let want = ir_outputs(&p, 64);
        assert!(
            want.iter()
                .flatten()
                .zip(one.iter().flatten())
                .all(|(&a, &b)| within(true, 0.0, a, b)),
            "a chain of {count} links is not one link"
        );
        let plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
        let module = iridia_naga::module(&p, &plan).unwrap();
        assert_eq!(module.functions.len(), 2);
        for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
            written(&p, &profile);
        }
        if let Some(warp) = warp.as_mut() {
            for text in warp_texts(&p) {
                assert_eq!(warp_disagreement(warp, &text, &want, 3, true, Slack::EXACT), None);
            }
        }
    }
}

#[test]
fn unary_operations_lower_to_their_definitions() {
    let mut cases = Vec::new();
    // (operation, largest operand compared, tolerance): WARP's tan is a hundredth off near its poles, its asin, acos and
    // atan a few parts in ten thousand, and its sinh, cosh and tanh come out wrong from an operand of about 5 up
    for (op, limit, tolerance) in [
        ("ftan", 16.0, 1.6e-2),
        ("fasin", 16.0, 2.5e-4),
        ("facos", 16.0, 2.5e-4),
        ("fatan", 16.0, 2.5e-4),
        ("fsinh", 4.0, 4e-3),
        ("fcosh", 4.0, 4e-3),
        ("ftanh", 4.0, 4e-3),
        ("fsign", f32::MAX, 0.0),
    ] {
        cases.push(unary(op, "f32", "f32").near(limit, tolerance));
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
    for op in [
        "umulhigh",
        "uaddcarry",
        "usubborrow",
        "umin",
        "umax",
        "shl",
        "shru",
    ] {
        cases.push(binary(op, "u32", "u32"));
    }
    cases.push(binary_by("shrs", "i32", "u32", "i32"));
    for op in [
        "smulhigh",
        "sdiv.allones",
        "srem.allones",
        "sdiv.byone",
        "srem.byone",
    ] {
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
        case(
            format!("{name} {ty}"),
            format!(
                "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<{ty}> = bitcast %0\n  %4: vec4<{ty}> = {name} %3, %1, %2\n  %5: vec4<u32> = bitcast %4\n  output 0.xyzw = %5\n"
            ),
            false,
        )
    };
    let cases = vec![
        three("bitextract.unsigned", "u32"),
        three("bitextract.signed", "i32"),
        case(
            "bitinsert",
            "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<u32> = bitinsert %0, %1, %2, %0\n  output 0.xyzw = %3\n",
            false,
        ),
        case(
            "fma f32",
            "  %0: vec4<u32> = input 0.xyzw\n  %1: vec4<u32> = input 1.xyzw\n  %2: vec4<u32> = input 2.xyzw\n  %3: vec4<f32> = bitcast %0\n  %4: vec4<f32> = bitcast %1\n  %5: vec4<f32> = bitcast %2\n  %6: vec4<f32> = fma %3, %4, %5\n  %7: vec4<u32> = bitcast %6\n  output 0.xyzw = %7\n",
            true,
        )
        .near(f32::MAX, 0.0).fused(),
    ];
    check(&cases);
}

#[test]
fn a_sample_position_is_direct3d_s_standard_one_for_every_count() {
    let texture = Resource {
        id: ResourceId(0),
        kind: ResourceKind::Texture {
            dim: TextureDim::D2,
            arrayed: false,
            multisampled: true,
            sample: SampleType::Float,
        },
        count: 1,
        provenance: ResourceProvenance {
            name: "msaa".into(),
            d3d: Some(D3dBinding {
                class: D3dClass::ShaderResource,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    };
    let p = program_with(
        "  %0: vec4<u32> = input 0.xyzw\n  %1: u32 = %0.x\n  %2: vec2<f32> = query.sampleposition r0 index %1\n  %3: vec2<u32> = bitcast %2\n  output 0.xy = %3\n",
        vec![texture],
    );
    // Direct3D's four-sample pattern, in sixteenths of a pixel, as its documentation gives it
    let four = [(-2, -6), (6, -2), (-6, 2), (2, 6)]
        .map(|(x, y): (i8, i8)| [f32::from(x) / 16.0, f32::from(y) / 16.0]);
    for (index, want) in (0..).zip(four) {
        assert_eq!(iridia_ir::standard_sample_position(4, index), Some(want));
    }
    // 31 standard positions: 1 + 2 + 4 + 8 + 16
    let standard: Vec<[f32; 2]> = [1u32, 2, 4, 8, 16]
        .into_iter()
        .flat_map(|count| {
            (0..count).map(move |index| iridia_ir::standard_sample_position(count, index).unwrap())
        })
        .collect();
    assert_eq!(standard.len(), 31);
    // the written table holds them, in the order the lowering indexes them (count n starts at row n - 1)
    for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
        let text = written(&p, &profile);
        let table = text
            .split("sample_positions")
            .nth(1)
            .and_then(|t| t.split(';').next())
            .unwrap_or_else(|| panic!("no sample_positions\n{text}"));
        // the lanes of every `vec2<f32>(x, y)` of the initialiser, an `f` suffix off
        let numbers: Vec<f32> = table
            .split("vec2<f32>(")
            .skip(1)
            .flat_map(|rest| rest.split(')').next().unwrap_or("").split(','))
            .map(|n| n.trim().trim_end_matches('f').parse::<f32>().unwrap_or(f32::NAN))
            .collect();
        let flat: Vec<f32> = standard.iter().flatten().copied().collect();
        assert_eq!(numbers, flat, "the table is not the standard positions\n{text}");
    }
    // and WARP, given a multisampled texture of each count it supports for the format (2, 4, 8 and 16; a count with no
    // standard pattern has no such texture), answers each index as the IR defines it: the sample's position, or (0, 0)
    // for an index past the count
    let Some(mut warp) = warp::Warp::shared() else {
        return;
    };
    for text in warp_texts(&p) {
        assert!(
            text.contains("@group(1) @binding(0)"),
            "the texture is not at group 1 binding 0\n{text}"
        );
        for count in [2u32, 4, 8, 16] {
            let mut indices: Vec<u32> = (0..count + 2).chain([u32::MAX]).collect();
            // a pixel program runs four invocations at a time
            while !indices.len().is_multiple_of(4) {
                indices.push(0);
            }
            let given: Vec<Vec<[u32; 4]>> = indices
                .iter()
                .map(|&i| vec![[i, 0, 0, 0], [0; 4], [0; 4]])
                .collect();
            let resources = serde_json::json!([
                {"group": 1, "binding": 0, "texture": {"texel": [0, 0, 0, 0], "samples": count}}
            ]);
            let got = warp
                .pixel_with(&text, &given, &resources)
                .unwrap_or_else(|e| panic!("{e}\n{text}"));
            for (&index, got) in indices.iter().zip(got) {
                let want = iridia_ir::standard_sample_position(count, index)
                    .unwrap_or([0.0, 0.0])
                    .map(f32::to_bits);
                let got = got.unwrap_or_else(|| panic!("sample {index} of {count} did not return"));
                assert_eq!([got[0], got[1]], want, "sample {index} of {count}\n{text}");
            }
        }
    }
}
