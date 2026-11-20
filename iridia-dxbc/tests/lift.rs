#![allow(clippy::unwrap_used)]
//! Lifting the D3D behaviour fixture (tests/fixtures/modes.hlsl, our own HLSL, compiled with fxc /T ps_5_0 /E main):
//! the interface and resources as worked out from the HLSL, and the D3D behaviours present in the lifted body as
//! plain operations.
use iridia_dxbc::ir::lift;
use iridia_ir::interface::{
    Access, BufferKind, D3dClass, ResourceKind, SampleType, ScalarType, Stage, StageFacts, StorageTarget,
    SystemValue, TextureDim,
};
use iridia_ir::{Op, Unary};
use iridia_wgsl::{Profile, wgsl};

/// The program in the form the writers take, written for naga's WGSL reader.
fn written(program: &mut iridia_ir::Program) -> String {
    iridia_passes::prepare(program).unwrap_or_else(|e| panic!("{e}"));
    wgsl(program, &Profile::NAGA).unwrap_or_else(|e| panic!("{e}"))
}

const MODES: &[u8] = include_bytes!("fixtures/modes.dxbc");

#[test]
fn the_fixture_lifts_validates_and_writes_wgsl() {
    let mut program = lift(MODES).unwrap();
    assert_eq!(program.stage, Stage::Pixel);
    assert_eq!(program.provenance.target, "ps_5_0");
    iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{p:?}"));

    // the program takes SV_Position as a parameter but never reads it: the signature lists it, and it is unused
    let inputs: Vec<_> = program
        .interface
        .inputs
        .iter()
        .map(|v| (v.system, v.used))
        .collect();
    assert_eq!(inputs, [(Some(SystemValue::FragCoord), false)]);
    // five colour targets out: a float4, three uint4 and a float4 (HLSL: struct O)
    let targets: Vec<_> = program
        .interface
        .outputs
        .iter()
        .map(|v| (v.system, v.value.scalar, v.value.width))
        .collect();
    assert_eq!(
        targets,
        [
            (Some(SystemValue::Target(0)), ScalarType::Float, 4),
            (Some(SystemValue::Target(1)), ScalarType::Uint, 4),
            (Some(SystemValue::Target(2)), ScalarType::Uint, 4),
            (Some(SystemValue::Target(3)), ScalarType::Uint, 4),
            (Some(SystemValue::Target(4)), ScalarType::Float, 4),
        ]
    );

    // cbuffer C : register(b0) { float4 f; uint4 u; int4 i; }: 48 bytes, named by the container's reflection
    let [cb] = program.interface.resources.as_slice() else {
        panic!("{:?}", program.interface.resources)
    };
    assert_eq!(cb.provenance.name, "C");
    let ResourceKind::ConstantBuffer { layout } = &cb.kind else {
        panic!("{cb:?}")
    };
    assert_eq!(layout.size, 48);
    let names: Vec<_> = layout
        .constants
        .iter()
        .map(|c| (c.name.as_str(), c.offset))
        .collect();
    assert_eq!(names, [("f", 0), ("u", 16), ("i", 32)]);
    let _ = BufferKind::Raw;

    let wgsl = written(&mut program);
    // NaN is tested on its bits (never `x != x`, which a driver may fold away): |x| bits above the infinity's
    assert!(
        wgsl.contains("2147483647u") && wgsl.contains("2139095040u"),
        "{wgsl}"
    );
    // D3D's unsigned division by zero: a division by a divisor made safe, all ones where the divisor was zero
    assert!(wgsl.contains("4294967295u"), "{wgsl}");
}

#[test]
fn the_provenance_names_what_the_lift_decided() {
    // the IR states D3D's behaviours as plain operations (the WGSL checks above find them), and what the lift chose
    // between implementation-defined ones as policies beside where the program came from
    let provenance = lift(MODES).unwrap().provenance;
    assert_eq!(provenance.format, iridia_ir::interface::ShaderFormat::Dxbc);
    assert_eq!(provenance.target, "ps_5_0");
    assert_eq!(provenance.lifter_version, iridia_dxbc::ir::LIFTER_VERSION);
    assert_eq!(provenance.policies, ["dxbc.mul=ieee"]);
}

/// A ps_5_0 container around `body` (declarations and instructions), with empty signatures.
fn pixel_container(body: &[Vec<u32>]) -> Vec<u8> {
    let mut code = vec![0x50u32, 0];
    for b in body {
        code.extend(b);
    }
    code[1] = code.len() as u32;
    let empty_signature: Vec<u8> = [0u32, 8].iter().flat_map(|w| w.to_le_bytes()).collect();
    let shex: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let chunks: [(&[u8; 4], &[u8]); 3] = [
        (b"ISGN", &empty_signature),
        (b"OSGN", &empty_signature),
        (b"SHEX", &shex),
    ];
    let directory = 32 + 4 * chunks.len();
    let mut offsets = Vec::new();
    let mut at = directory;
    for (_, data) in &chunks {
        offsets.push(at as u32);
        at += 8 + data.len();
    }
    let mut out = b"DXBC".to_vec();
    out.extend([0u8; 16]);
    for w in [1, at as u32, chunks.len() as u32] {
        out.extend(w.to_le_bytes());
    }
    for o in offsets {
        out.extend(o.to_le_bytes());
    }
    for (tag, data) in chunks {
        out.extend(tag);
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
    }
    out
}

fn instruction(token: u32, operands: &[u32]) -> Vec<u32> {
    let mut v = vec![token | (((operands.len() + 1) as u32) << 24)];
    v.extend_from_slice(operands);
    v
}

/// `sincos r0.x, r1.x, r0.x` writes the sine to r0.x and the cosine to r1.x, both of the original r0.x: the
/// instruction reads its source once. (A lift that re-read r0.x for the cosine gave cos(sin(x)): found by the
/// executor-against-evaluator comparison on a Unity program.)
#[test]
fn a_destination_that_aliases_the_source_does_not_change_the_second_result() {
    let temp = |register: u32, mask: u32| [2 | mask << 4 | 1 << 20, register];
    let scalar = |register: u32, lane: u32| [2 | 2 << 2 | lane << 4 | 1 << 20, register];
    let operands = [temp(0, 1), temp(1, 1), scalar(0, 0)].concat();
    let container = pixel_container(&[
        instruction(104, &[2]),
        instruction(77, &operands),
        instruction(62, &[]),
    ]);
    let program = lift(&container).unwrap();
    iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{p:?}"));
    let argument_of = |wanted: Unary| {
        program.entry.values.iter().find_map(|v| match v.op {
            Op::Unary(op, arg) if op == wanted => Some(arg),
            _ => None,
        })
    };
    let (sin, cos) = (
        argument_of(Unary::FSin).unwrap(),
        argument_of(Unary::FCos).unwrap(),
    );
    assert_eq!(sin, cos, "both read the one loaded source");
}

/// A typed unordered-access texture view, stored to and loaded from: `dcl_uav_typed_texture2d (float,float,float,
/// float) u3; store_uav_typed u3.xyzw, r0.xyxx, r1.xyzw; ld_uav_typed r1.xyzw, r0.xyxx, u3.xyzw`. It lifts to a
/// storage image of four 32-bit float lanes, read and written, bound as `u3`, and the instructions to an image store
/// and an image load.
#[test]
fn a_typed_view_is_a_storage_image_stored_and_loaded() {
    let temp_mask = |register: u32| [2 | 15 << 4 | 1 << 20, register];
    let temp = |register: u32, swizzle: u32| [2 | 1 << 2 | swizzle << 4 | 1 << 20, register];
    let view = |slot: u32, mask_or_swizzle: u32, swizzled: bool| {
        [
            2 | u32::from(swizzled) << 2 | mask_or_swizzle << 4 | 30 << 12 | 1 << 20,
            slot,
        ]
    };
    let xyxx = 0b00_00_01_00;
    let xyzw = 0b11_10_01_00;
    let declaration = [30 << 12 | 1 << 20, 3, 0x5555];
    let store = [view(3, 15, false).as_slice(), &temp(0, xyxx), &temp(1, xyzw)].concat();
    let load = [temp_mask(1).as_slice(), &temp(0, xyxx), &view(3, xyzw, true)].concat();
    let container = pixel_container(&[
        instruction(156 | 3 << 11, &declaration),
        instruction(104, &[2]),
        instruction(164, &store),
        instruction(163, &load),
        instruction(62, &[]),
    ]);
    let mut program = lift(&container).unwrap();
    iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{p:?}"));
    let [resource] = program.interface.resources.as_slice() else {
        panic!("{:?}", program.interface.resources)
    };
    assert_eq!(resource.provenance.name, "u3");
    assert_eq!(
        resource.provenance.d3d.map(|b| (b.class, b.register)),
        Some((D3dClass::UnorderedAccess, 3))
    );
    assert_eq!(
        resource.kind,
        ResourceKind::Storage {
            target: StorageTarget::Image {
                dim: TextureDim::D2,
                arrayed: false,
                sample: SampleType::Float,
            },
            access: Access::ReadWrite,
        }
    );
    // the load's result is unused, so the writers' form drops it: it is in the lift
    assert!(
        program
            .entry
            .values
            .iter()
            .any(|v| matches!(v.op, Op::LoadTexel { .. })),
        "{:?}",
        program.entry.values
    );
    let wgsl = written(&mut program);
    assert!(wgsl.contains("@group(3) @binding(3)"), "{wgsl}");
    assert!(
        wgsl.contains("texture_storage_2d<rgba32float, read_write>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("textureStore(u3"), "{wgsl}");
}

/// A store through a view with a partial write mask would keep the other lanes, which a store of four lanes cannot:
/// refused by name rather than lanes lost.
#[test]
fn a_partial_write_mask_on_a_view_is_refused() {
    let declaration = [30 << 12 | 1 << 20, 0, 0x4444];
    let store = [
        [2 | 1 << 4 | 30 << 12 | 1 << 20, 0],
        [2 | 1 << 2 | 0b00_00_01_00 << 4 | 1 << 20, 0],
        [2 | 1 << 2 | 0b11_10_01_00 << 4 | 1 << 20, 1],
    ]
    .concat();
    let container = pixel_container(&[
        instruction(156 | 3 << 11, &declaration),
        instruction(104, &[2]),
        instruction(164, &store),
        instruction(62, &[]),
    ]);
    let e = lift(&container).unwrap_err();
    assert!(e.detail.contains("a typed store whose mask is not .xyzw"), "{e}");
}

const COMPUTE: [&[u8]; 2] = [
    include_bytes!("fixtures/compute_basic.dxbc"),
    include_bytes!("fixtures/compute_basic.od.dxbc"),
];

/// The compute fixture (tests/fixtures/compute_basic.hlsl, fxc /T cs_5_0): a thread group of 8 by 4 by 2, the four
/// ids, shared memory with atomics, and a raw, a structured and a typed buffer.
#[test]
fn the_compute_fixture_lifts_validates_and_writes_wgsl() {
    for (name, container) in ["default", "/Od"].into_iter().zip(COMPUTE) {
        let mut program = lift(container).unwrap_or_else(|e| panic!("{name}: {e}"));
        iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{name}: {p:?}"));
        assert_eq!(program.stage, Stage::Compute, "{name}");
        assert_eq!(program.provenance.target, "cs_5_0", "{name}");
        assert_eq!(program.workgroup, Some([8, 4, 2]), "{name}");
        let systems: Vec<_> = program.interface.inputs.iter().filter_map(|v| v.system).collect();
        for id in [
            SystemValue::GroupId,
            SystemValue::GroupThreadId,
            SystemValue::DispatchThreadId,
            SystemValue::GroupIndex,
        ] {
            assert!(systems.contains(&id), "{name}: no {id:?} in {systems:?}");
        }
        // u0 raw, u1 structured of uint, u2 typed: all read and written
        let storage: Vec<_> = program
            .interface
            .resources
            .iter()
            .filter_map(|r| match &r.kind {
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(kind),
                    ..
                } => Some((r.provenance.d3d.unwrap().register, *kind)),
                _ => None,
            })
            .collect();
        assert!(
            storage.iter().any(|(r, k)| *r == 0 && *k == BufferKind::Raw),
            "{name}: {storage:?}"
        );
        assert!(
            storage
                .iter()
                .any(|(r, k)| *r == 1 && matches!(k, BufferKind::Structured { stride: 4 })),
            "{name}: {storage:?}"
        );
        assert!(
            storage
                .iter()
                .any(|(r, k)| *r == 2 && matches!(k, BufferKind::Typed(_))),
            "{name}: {storage:?}"
        );
        // the resources are named and laid out by the program's reflection, as the source declares them
        let named: Vec<(&str, D3dClass, u32)> = program
            .interface
            .resources
            .iter()
            .filter_map(|r| {
                let d = r.provenance.d3d?;
                Some((r.provenance.name.as_str(), d.class, d.register))
            })
            .collect();
        for expected in [
            ("raw", D3dClass::UnorderedAccess, 0),
            ("words", D3dClass::UnorderedAccess, 1),
            ("typed", D3dClass::UnorderedAccess, 2),
            ("Values", D3dClass::ConstantBuffer, 0),
        ] {
            assert!(named.contains(&expected), "{name}: {expected:?} not in {named:?}");
        }
        let values = program
            .interface
            .resources
            .iter()
            .find_map(|r| match &r.kind {
                ResourceKind::ConstantBuffer { layout } if r.provenance.name == "Values" => Some(layout),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            values
                .constants
                .iter()
                .map(|c| (c.name.as_str(), c.offset))
                .collect::<Vec<_>>(),
            [("scale", 0)],
            "{name}"
        );
        let wgsl = written(&mut program);
        for needle in [
            "@compute @workgroup_size(8, 4, 2)",
            "var<workgroup>",
            "atomicAdd(",
            "workgroupBarrier()",
            "var<storage, read_write>",
        ] {
            assert!(wgsl.contains(needle), "{name}: {needle:?} is not in\n{wgsl}");
        }
    }
}

/// The hull fixture whose phases declare different numbers of temporaries (tests/fixtures/hull_phases.hlsl, fxc /T hs_5_0:
/// `dcl_temps 2` in the control-point phase, 1 in the last fork phase) lifts: each phase may use the temporaries it
/// declared, so the program has the most any of them declares.
#[test]
fn a_hull_program_whose_phases_declare_different_temporaries_lifts() {
    let mut program = lift(include_bytes!("fixtures/hull_phases.hs.dxbc")).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(program.stage, Stage::Hull);
    // each phase's function compiles for the executor
    iridia_passes::prepare(&mut program).unwrap_or_else(|e| panic!("{e}"));
    iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{p:?}"));
    for name in ["main", "patch_constant"] {
        iridia_exec::Executable::function_from_ir(&program, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

/// A hull and a domain program with a `min16float` in the patch-constant structure (tests/fixtures/min16_patch.hlsl,
/// fxc /T hs_5_0 and ds_5_0): fxc writes their patch-constant signature as a `PSG1` part, not `PCSG`. Compiled with the
/// `min16float`s as `float` (the `.plain` twins) the programs have a `PCSG` part and are otherwise the same source, so the
/// two lift to the same program and the same WGSL.
#[test]
fn a_patch_constant_signature_in_a_psg1_part_lifts_as_the_pcsg_one_does() {
    for (min16, plain, stage) in [
        (
            &include_bytes!("fixtures/min16_patch.hs.dxbc")[..],
            &include_bytes!("fixtures/min16_patch.plain.hs.dxbc")[..],
            Stage::Hull,
        ),
        (
            &include_bytes!("fixtures/min16_patch.ds.dxbc")[..],
            &include_bytes!("fixtures/min16_patch.plain.ds.dxbc")[..],
            Stage::Domain,
        ),
    ] {
        let mut from_psg1 = lift(min16).unwrap_or_else(|e| panic!("{stage:?}: {e}"));
        let from_pcsg = lift(plain).unwrap_or_else(|e| panic!("{stage:?}: {e}"));
        iridia_ir::validate(&from_psg1).unwrap_or_else(|p| panic!("{p:?}"));
        assert_eq!(from_psg1.stage, stage);
        // both are tessellation programs of the same source: the same functions, interface and patch constants (the
        // PCDATA0 varying); they differ in the container's digest and in the note that the minimum precision operands
        // are computed at 32 bits
        assert_eq!(from_psg1.interface, from_pcsg.interface, "{stage:?}");
        assert_eq!(
            iridia_ir::text::print(&from_psg1.entry),
            iridia_ir::text::print(&from_pcsg.entry),
            "{stage:?}"
        );
        assert_eq!(
            from_psg1.patch.as_ref().map(iridia_ir::text::print),
            from_pcsg.patch.as_ref().map(iridia_ir::text::print),
            "{stage:?}"
        );
        let patch_constants = match &from_psg1.interface.facts {
            StageFacts::Hull(h) => &h.patch_constants,
            StageFacts::Domain(d) => &d.patch_constants,
            other => panic!("{stage:?}: {other:?}"),
        };
        assert!(
            patch_constants.iter().any(|v| v.label.name == "PCDATA"),
            "{stage:?}"
        );
        assert!(
            from_psg1.notes.iter().any(|n| n.contains("min-precision")),
            "{stage:?}"
        );
        // neither IR writer writes the functions of a stage with no entry point yet (the naga-path writer did): the
        // refusal names the stage, so a writer that learns to write it fails here and this test gets its text check back
        iridia_passes::prepare(&mut from_psg1).unwrap_or_else(|e| panic!("{stage:?}: {e}"));
        let refusal = wgsl(&from_psg1, &Profile::NAGA).unwrap_err().to_string();
        assert!(
            refusal.contains(&format!("{stage:?} program is not")),
            "{stage:?}: {refusal}"
        );
    }
}

/// A float compared with itself is the test for NaN (`ne x, x` is what fxc writes for `isnan(x)`), and a compiler that
/// assumes no NaN writes a constant for a comparison of two reads of one value: where an instruction's two sources
/// are one value both lifts make the test on the bits (`Unary::IsNan`; the older lift's `Builder::is_nan`, whose
/// last operation is the only `>` here), and no comparison. A lane is one value where both sources read the same
/// component of the same register at the same index under the same modifier; an instruction of both kinds of lane
/// (fxc packs them) has the test and the comparison; `lt x, x` and every comparison of two values stay comparisons.
#[test]
fn a_float_compared_with_itself_lifts_as_the_test_for_nan_lane_by_lane() {
    use iridia_ir::Binary;
    use naga::{BinaryOperator as Bo, Expression};
    let (ne, eq, ge, lt, ieq) = (57, 24, 29, 49, 32);
    let lanes = |kind: u32, index: u32, l: [u32; 4]| {
        vec![
            2 | 1 << 2 | (l[0] | l[1] << 2 | l[2] << 4 | l[3] << 6) << 4 | kind << 12 | 1 << 20,
            index,
        ]
    };
    let scalar = |index: u32, lane: u32| vec![2 | 2 << 2 | lane << 4 | 1 << 20, index];
    let modified = |source: Vec<u32>, modifier: u32| vec![source[0] | 1 << 31, 1 | modifier << 6, source[1]];
    let literal = |words: [u32; 4]| [&[2 | 4 << 12][..], &words].concat();
    // cb0[row], and cb0[r0.<lane>]: a row named by a register
    let row = |row: u32| vec![2 | 1 << 2 | 0xe4 << 4 | 8 << 12 | 2 << 20, 0, row];
    let row_at = |lane: u32| {
        vec![
            2 | 1 << 2 | 0xe4 << 4 | 8 << 12 | 2 << 20 | 3 << 25,
            0,
            0,
            2 | 2 << 2 | lane << 4 | 1 << 20,
            0,
        ]
    };
    let xyzw = [0, 1, 2, 3];
    let r0 = || lanes(0, 0, xyzw);
    // the comparison, r1's write mask, its two sources, and how many tests for NaN and how many comparisons it lifts to
    let cases = vec![
        ("ne r0, r0", ne, 15, r0(), r0(), (1, 0)),
        ("eq r0, r0", eq, 15, r0(), r0(), (1, 0)),
        ("ge r0, r0", ge, 15, r0(), r0(), (1, 0)),
        ("lt r0, r0", lt, 15, r0(), r0(), (0, 1)),
        ("ne r0.y, r0.y", ne, 1, scalar(0, 1), scalar(0, 1), (1, 0)),
        ("ne r0.y, r0.z", ne, 1, scalar(0, 1), scalar(0, 2), (0, 1)),
        // a lane selected as a scalar and by a swizzle is the same lane
        (
            "ne r0.y, r0.yyyy",
            ne,
            1,
            scalar(0, 1),
            lanes(0, 0, [1; 4]),
            (1, 0),
        ),
        // the swizzles differ only in the lane the destination .xyz does not write
        (
            "ne r0.xyzx, r0.xyzw",
            ne,
            7,
            lanes(0, 0, [0, 1, 2, 0]),
            r0(),
            (1, 0),
        ),
        // the same destination .xyzw writes that lane, a comparison of two values beside three tests
        (
            "ne r0.xyzx, r0.xyzw (.xyzw)",
            ne,
            15,
            lanes(0, 0, [0, 1, 2, 0]),
            r0(),
            (1, 1),
        ),
        (
            "eq r0.yzyy, r0.yyyy",
            eq,
            3,
            lanes(0, 0, [1, 2, 1, 1]),
            lanes(0, 0, [1; 4]),
            (1, 1),
        ),
        // no written lane against itself, though the swizzles agree in the lanes .xy does not write
        (
            "ne r0.xyxx, r0.yxxx",
            ne,
            3,
            lanes(0, 0, [0, 1, 0, 0]),
            lanes(0, 0, [1, 0, 0, 0]),
            (0, 1),
        ),
        (
            "ne -r0, -r0",
            ne,
            15,
            modified(r0(), 1),
            modified(r0(), 1),
            (1, 0),
        ),
        (
            "ne |r0|, |r0|",
            ne,
            15,
            modified(r0(), 2),
            modified(r0(), 2),
            (1, 0),
        ),
        ("ne r0, -r0", ne, 15, r0(), modified(r0(), 1), (0, 1)),
        ("eq |r0|, r0", eq, 15, modified(r0(), 2), r0(), (0, 1)),
        ("ne r0, r2", ne, 15, r0(), lanes(0, 2, xyzw), (0, 1)),
        ("ne cb0[1], cb0[1]", ne, 15, row(1), row(1), (1, 0)),
        ("ne cb0[0], cb0[1]", ne, 15, row(0), row(1), (0, 1)),
        ("ne cb0[r0.x], cb0[r0.x]", ne, 15, row_at(0), row_at(0), (1, 0)),
        ("ne cb0[r0.x], cb0[r0.y]", ne, 15, row_at(0), row_at(1), (0, 1)),
        (
            "ne l(1, 2, 3, 4), l(1, 2, 3, 4)",
            ne,
            15,
            literal([1, 2, 3, 4]),
            literal([1, 2, 3, 4]),
            (1, 0),
        ),
        (
            "ne l(1, 2, 3, 4), l(1, 2, 3, 5)",
            ne,
            15,
            literal([1, 2, 3, 4]),
            literal([1, 2, 3, 5]),
            (0, 1),
        ),
        // an integer has no NaN: its comparison with itself is a comparison
        ("ieq r0, r0", ieq, 15, r0(), r0(), (0, 1)),
    ];
    for (name, opcode, mask, first, second, expected) in cases {
        let destination = [2 | mask << 4 | 1 << 20, 1];
        let container = pixel_container(&[
            // dcl_constantbuffer cb0[2], dcl_temps 3
            instruction(89 | 0x800, &[2 | 15 << 4 | 8 << 12 | 2 << 20, 0, 2]),
            instruction(104, &[3]),
            instruction(opcode, &[&destination[..], &first, &second].concat()),
            instruction(62, &[]),
        ]);

        let program = lift(&container).unwrap_or_else(|e| panic!("{name}: {e}"));
        iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{name}: {p:?}"));
        let count =
            |wanted: &dyn Fn(&Op) -> bool| program.entry.values.iter().filter(|v| wanted(&v.op)).count();
        let tests = count(&|op| matches!(op, Op::Unary(Unary::IsNan, _)));
        let comparisons = count(&|op| {
            matches!(
                op,
                Op::Binary(Binary::FCmp(_) | Binary::UCmp(_) | Binary::SCmp(_), _, _)
            )
        });
        assert_eq!((tests, comparisons), expected, "{name}: the IR lift");

        let older = iridia_dxbc::lift(&container).unwrap_or_else(|e| panic!("{name}: {e}"));
        let [entry] = older.module.entry_points.as_slice() else {
            panic!("{name}: {:?}", older.module.entry_points)
        };
        let count = |wanted: &[Bo]| {
            entry
                .function
                .expressions
                .iter()
                .filter(|(_, e)| matches!(e, Expression::Binary { op, .. } if wanted.contains(op)))
                .count()
        };
        // the instruction's own operator: a row named by a register is read behind a comparison of its own
        let own = match opcode {
            57 => Bo::NotEqual,
            29 => Bo::GreaterEqual,
            49 => Bo::Less,
            _ => Bo::Equal,
        };
        assert_eq!(
            (count(&[Bo::Greater]), count(&[own])),
            expected,
            "{name}: the older lift"
        );
    }
}
