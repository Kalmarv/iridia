#![allow(clippy::unwrap_used)]
//! A float compared with itself, the test for NaN (`self_compare.hlsl`, our own HLSL, fxc /T ps_5_0): `isnan(x)` and
//! `x != x` are `ne x, x` in what fxc writes, and it packs a lane against itself beside a lane against another into one
//! instruction. A compiler that assumes no NaN writes a constant for a comparison of two reads of one value, and the
//! program's guard against a NaN is gone; so both lifts make the test on the value's bits in those lanes. The programs
//! they lift give Direct3D's values with a NaN and without one, and the IR and the text written of it hold the test.

use iridia_exec::{PixelShader, ProgramShader, Sample};
use iridia_ir::{Binary, Cmp, Op, Unary};

const FIXTURE: &[u8] = include_bytes!("fixtures/self_compare.dxbc");

fn no_sample(_: Sample) -> Result<[f32; 4], String> {
    panic!("unexpected sample")
}

/// The fixture lifted onto the IR and taken through the passes an output takes.
fn prepared() -> iridia_ir::Program {
    let mut program = iridia_dxbc::ir::lift(FIXTURE).unwrap_or_else(|e| panic!("{e}"));
    iridia_passes::prepare(&mut program).unwrap_or_else(|e| panic!("{e}"));
    iridia_ir::validate(&program).unwrap_or_else(|p| panic!("{p:?}"));
    program
}

/// `q = n / d`, then `guarded = (isnan(q.xyz) ? 0 : q.xyz, 1)`, `same = (q.x != q.x, q.y == q.y, q.z >= q.z,
/// q.w < q.w)` and `other = (q.x != q.y, q.y == q.z, q.z >= q.w, q.w < q.x)`, each comparison 1 or 0: the values by
/// hand from Direct3D's rule that a comparison with a NaN fails, except `!=`, which holds.
#[test]
fn both_lifts_give_the_comparisons_values_with_a_nan_and_without() {
    let f = |x: [f32; 4]| x.map(f32::to_bits);
    // n, d, then guarded, same and other
    let cases = [
        // q = NaN, 1, NaN, 2 (0 / 0 is NaN)
        (
            ([0., 1., 0., 6.], [0., 1., 0., 3.]),
            (f([0., 1., 0., 1.]), [1, 1, 0, 0], [1, 0, 0, 0]),
        ),
        // q = 1, 2, 3, 2
        (
            ([1., 2., 3., 4.], [1., 1., 1., 2.]),
            (f([1., 2., 3., 1.]), [0, 1, 1, 0], [1, 0, 1, 0]),
        ),
        // q = inf, -inf, 0, 1: an infinity is not a NaN
        (
            ([1., -1., 0., 1.], [0., 0., 1., 1.]),
            (
                f([f32::INFINITY, f32::NEG_INFINITY, 0., 1.]),
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ),
        ),
    ];
    let older = ProgramShader::new(iridia_dxbc::lift(FIXTURE).unwrap_or_else(|e| panic!("{e}"))).unwrap();
    let ir = ProgramShader::from_ir(prepared()).unwrap();
    for ((n, d), (guarded, same, other)) in cases {
        let buffer = [f(n), f(d)];
        for (lift, shader) in [("the older lift", &older), ("the IR lift", &ir)] {
            let out = shader
                .execute(&[], &[(0, &buffer)], &mut no_sample)
                .unwrap()
                .outputs;
            assert_eq!(
                (out[0], out[1], out[2]),
                (guarded, same, other),
                "{lift}: {n:?} / {d:?}"
            );
        }
    }
}

/// The lift holds one test for NaN for each of `ne`, `eq` and `ge` (the lanes against themselves) and one comparison
/// for each instruction's other lanes (`lt` has no lane against itself); both writers write the test on the bits, the
/// magnitude's above an infinity's.
#[test]
fn the_lanes_against_themselves_are_tests_for_nan_in_the_ir_and_its_text() {
    let lifted = iridia_dxbc::ir::lift(FIXTURE).unwrap_or_else(|e| panic!("{e}"));
    let tests = lifted
        .entry
        .values
        .iter()
        .filter(|v| matches!(v.op, Op::Unary(Unary::IsNan, _)))
        .count();
    assert_eq!(tests, 3);
    let mut comparisons: Vec<Cmp> = lifted
        .entry
        .values
        .iter()
        .filter_map(|v| match v.op {
            Op::Binary(Binary::FCmp(c), _, _) => Some(c),
            _ => None,
        })
        .collect();
    comparisons.sort();
    assert_eq!(comparisons, [Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Ge]);

    let program = prepared();
    let texts = [
        (
            "iridia-wgsl",
            iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::NAGA).unwrap_or_else(|e| panic!("{e}")),
        ),
        (
            "iridia-naga",
            iridia_naga::wgsl(&program, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e:?}")),
        ),
        (
            "the older lift",
            iridia::emit::wgsl(&iridia_dxbc::lift(FIXTURE).unwrap()).unwrap_or_else(|e| panic!("{e}")),
        ),
    ];
    for (writer, text) in texts {
        // (bits & 0x7fffffff) > 0x7f800000
        assert!(
            text.contains("2147483647u") && text.contains("2139095040u"),
            "{writer}:\n{text}"
        );
    }
}
