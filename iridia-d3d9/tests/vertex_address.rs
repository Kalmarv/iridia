#![allow(clippy::unwrap_used)]
//! Hand-assembled `vs_3_0` tokens for the address register: `mova`, relative float constants (`c[a0.x + N]`), and the
//! vertex-only `slt`, `sge` and `sincos`. Expected values are worked from the instruction definitions, not from a second
//! decoder, and the rounding of `mova` is Microsoft's reference rasterizer's (a half goes away from zero), a relative
//! read outside the 256 constant registers is zero, and a `def` constant is what the register it names reads as.
use iridia_d3d9::{lift, read};
use iridia_exec::{ProgramShader, Sample};

fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
    [vec![op | (operands.len() as u32) << 24], operands.to_vec()].concat()
}
fn program(body: &[u32]) -> Vec<u8> {
    [vec![0xfffe_0300], body.to_vec(), vec![0x0000_ffff]]
        .concat()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn dcl(usage: u32, index: u32, dst: u32) -> Vec<u32> {
    instruction(31, &[0x8000_0000 | index << 16 | usage, dst])
}
/// `dcl_position v15` and `dcl_position o7`: the position is the program's one input and output.
fn position() -> Vec<u32> {
    [dcl(0, 0, 0x900f_000f), dcl(0, 0, 0xe00f_0007)].concat()
}
const A0_X: u32 = 0xb001_0000;
/// `v15.xxxx`.
const V15_X: u32 = 0x9000_000f;
/// `o7.xyzw`.
const O7: u32 = 0xe00f_0007;

/// `mova a0.x, v15.x` and `mov o7, c[a0.<lane> + n]`.
fn relative_read(n: u32, lane: u32) -> Vec<u32> {
    [
        instruction(46, &[A0_X, V15_X]),
        // the constant source carries the relative flag (bit 13); its address token follows: a0 with the lane replicated
        instruction(1, &[O7, 0xa0e4_2000 | n, 0xb000_0000 | (lane * 0x55) << 16]),
    ]
    .concat()
}

fn run(bytes: &[u8], input: [f32; 4], constants: &[(usize, [f32; 4])]) -> [f32; 4] {
    let mut v = [[0u32; 4]; 16];
    v[15] = input.map(f32::to_bits);
    let mut bank = [[0u32; 4]; 256];
    for &(row, value) in constants {
        bank[row] = value.map(f32::to_bits);
    }
    let (_, clip) = ProgramShader::new(lift(bytes).unwrap())
        .unwrap()
        .execute_vertex(&v, &[(0, &bank)], &mut |_: Sample| Err("no texture".into()))
        .unwrap();
    clip.map(f32::from_bits)
}

fn refusal(body: &[u32]) -> String {
    lift(&program(body)).unwrap_err().to_string()
}

#[test]
fn mova_rounds_a_half_away_from_zero_and_a_relative_constant_reads_that_row() {
    let bytes = program(&[position(), relative_read(2, 0)].concat());
    let rows = [
        (1, [1., 1., 1., 1.]),
        (2, [2., 2., 2., 2.]),
        (3, [3., 3., 3., 3.]),
        (4, [4., 4., 4., 4.]),
        (5, [5., 5., 5., 5.]),
    ];
    // a0.x is the nearest integer, a half away from zero; the program reads c[a0.x + 2]
    for (x, row) in [
        (1.0, 3),
        (1.5, 4),
        (0.5, 3),
        (-0.5, 1),
        (-0.49, 2),
        (-0.6, 1),
        (-1.5, 0),
        (2.49, 4),
        (2.5, 5),
    ] {
        let expected = rows.iter().find(|(n, _)| *n == row).map_or([0.; 4], |r| r.1);
        assert_eq!(
            run(&bytes, [x, 0., 0., 1.], &rows),
            expected,
            "mova of {x} reads c{row}"
        );
    }
}

#[test]
fn a_relative_read_outside_the_constant_file_is_zero() {
    let bytes = program(&[position(), relative_read(2, 0)].concat());
    let all = (0..256).map(|n| (n, [9., 9., 9., 9.])).collect::<Vec<_>>();
    assert_eq!(
        run(&bytes, [253., 0., 0., 1.], &all),
        [9.; 4],
        "c255 is the last row"
    );
    assert_eq!(
        run(&bytes, [254., 0., 0., 1.], &all),
        [0.; 4],
        "c256 does not exist"
    );
    assert_eq!(
        run(&bytes, [-3., 0., 0., 1.], &all),
        [0.; 4],
        "c-1 does not exist"
    );
}

#[test]
fn each_lane_of_a0_can_address() {
    // mova a0.y, v15.x; mov o7, c[a0.y + 0]
    let bytes = program(
        &[
            position(),
            instruction(46, &[0xb002_0000, V15_X]),
            instruction(1, &[O7, 0xa0e4_2000, 0xb055_0000]),
        ]
        .concat(),
    );
    assert_eq!(
        run(&bytes, [7., 0., 0., 1.], &[(7, [1., 2., 3., 4.])]),
        [1., 2., 3., 4.]
    );
}

#[test]
fn a_def_constant_is_what_a_relative_read_of_its_register_returns() {
    let bytes = program(
        &[
            position(),
            instruction(
                81,
                &[
                    0xa00f_0003,
                    11f32.to_bits(),
                    12f32.to_bits(),
                    13f32.to_bits(),
                    14f32.to_bits(),
                ],
            ),
            relative_read(2, 0),
        ]
        .concat(),
    );
    // x = 1 reads c3, which the program defines; x = 2 reads the bank's c4
    assert_eq!(
        run(&bytes, [1., 0., 0., 1.], &[(3, [0.; 4])]),
        [11., 12., 13., 14.]
    );
    assert_eq!(
        run(&bytes, [2., 0., 0., 1.], &[(4, [5., 6., 7., 8.])]),
        [5., 6., 7., 8.]
    );
}

#[test]
fn slt_and_sge_give_one_or_zero_per_lane() {
    // slt r0, v15, c0; sge r1, v15, c0; o7 = r0 + r1 * 2
    let bytes = program(
        &[
            position(),
            instruction(12, &[0x800f_0000, 0x90e4_000f, 0xa0e4_0000]),
            instruction(13, &[0x800f_0001, 0x90e4_000f, 0xa0e4_0000]),
            instruction(4, &[O7, 0x80e4_0001, 0xa0e4_0001, 0x80e4_0000]),
        ]
        .concat(),
    );
    // c1 = 2: o7 = slt + 2 * sge. lanes compare 1 against 2, 2 against 2, 3 against 2, and NaN against 2
    let out = run(
        &bytes,
        [1., 2., 3., f32::NAN],
        &[(0, [2., 2., 2., 2.]), (1, [2., 2., 2., 2.])],
    );
    assert_eq!(out, [1., 2., 2., 0.]);
}

#[test]
fn sincos_gives_the_cosine_in_x_and_the_sine_in_y() {
    // sincos r0.xy, v15.x; mov r0.zw, v15; mov o7, r0
    let bytes = program(
        &[
            position(),
            instruction(37, &[0x8003_0000, V15_X]),
            instruction(1, &[0x800c_0000, 0x90e4_000f]),
            instruction(1, &[O7, 0x80e4_0000]),
        ]
        .concat(),
    );
    for x in [0.0f32, 0.5, -1.25, 2.0] {
        let out = run(&bytes, [x, 5., 6., 7.], &[]);
        assert!(
            (out[0] - x.cos()).abs() < 1e-6 && (out[1] - x.sin()).abs() < 1e-6,
            "{x}: {out:?}"
        );
        assert_eq!(out[2..], [6., 7.]);
    }
}

#[test]
fn what_the_address_register_cannot_do_is_refused_by_name() {
    let body = |tokens: &[u32]| [position(), instruction(46, &[A0_X, V15_X]), tokens.to_vec()].concat();
    // the relative address token must be a0 with one lane replicated
    for (address, why) in [
        (0xb0e4_0000u32, "does not replicate one lane"),
        (0xf000_0000, "aL is not supported"),
        (0xa000_0000, "other than the vertex address register"),
        (0xb100_0000, "with a modifier"),
    ] {
        let tokens = instruction(1, &[O7, 0xa0e4_2000, address]);
        assert!(
            refusal(&body(&tokens)).contains(why),
            "{address:#x}: {}",
            refusal(&body(&tokens))
        );
    }
    // the flag on a register that is not a float constant, and without its address token
    let tokens = instruction(1, &[O7, 0x80e4_2000, 0xb000_0000]);
    assert!(refusal(&body(&tokens)).contains("relative addressing (not supported)"));
    let tokens = instruction(1, &[O7, 0xa0e4_2000]);
    assert!(
        refusal(&body(&tokens)).contains("without its address token")
            || refusal(&body(&tokens)).contains("operands")
    );
    // a relative destination
    let tokens = instruction(1, &[O7 | 0x2000, 0x90e4_000f]);
    assert!(refusal(&body(&tokens)).contains("relative addressing of a destination"));
    // a0 read before mova wrote it
    let no_mova = [position(), instruction(1, &[O7, 0xa0e4_2000, 0xb000_0000])].concat();
    assert!(refusal(&no_mova).contains("before any mova"));
    // mova's destination is a0 only, and only a vertex program has it
    let bad = [position(), instruction(46, &[0x800f_0000, V15_X])].concat();
    assert!(refusal(&bad).contains("Mova destination"));
    let pixel: Vec<u8> = [0xffff_0300u32]
        .into_iter()
        .chain(instruction(46, &[A0_X, 0x9000_0000]))
        .chain([0xffff])
        .flat_map(u32::to_le_bytes)
        .collect();
    assert!(read(&pixel).unwrap_err().to_string().contains("vertex-only"));
    // sincos writes x, y or xy only
    let bad = [position(), instruction(37, &[0x800f_0000, V15_X])].concat();
    assert!(refusal(&bad).contains("SINCOS writes only"));
}
