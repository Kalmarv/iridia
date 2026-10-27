#![allow(clippy::unwrap_used)]
//! Containers a damaged file could hold, made from the committed fixtures by changing one byte (found by mutating every fixture
//! and shrinking each panic of the lifter to the one byte that causes it): each is refused by name, never a panic. The oracle
//! is the format's own rules (a value, a block or a callee the function has; the arguments an operation takes).
use iridia_dxil::lift;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn refusal(bytes: &[u8]) -> String {
    match lift(bytes) {
        Ok(_) => panic!("the program was lifted"),
        Err(e) => e.detail,
    }
}

/// The fixture with the byte at `at`, which is `was`, made `now`.
fn changed(name: &str, at: usize, was: u8, now: u8) -> Vec<u8> {
    let mut bytes = fixture(name);
    assert_eq!(bytes[at], was, "{name} is not the fixture the case was made from");
    bytes[at] = now;
    bytes
}

#[test]
fn an_unchanged_fixture_still_lifts() {
    lift(&fixture("arrays.66.dxil")).unwrap();
}

#[test]
fn a_value_the_function_does_not_have_is_refused() {
    // an operand whose relative id wraps past the function's values
    let bytes = changed("arrays.66.dxil", 4492, 0x06, 0x7a);
    assert!(refusal(&bytes).contains("of 84 values"), "{}", refusal(&bytes));
}

#[test]
fn an_instruction_after_the_last_terminator_is_refused() {
    let bytes = changed("buffer_arrays.66.dxil", 4184, 0x00, 0x01);
    assert!(refusal(&bytes).contains("an instruction after the last terminator"));
}

#[test]
fn a_dx_op_called_with_the_wrong_number_of_arguments_is_refused() {
    // the opcode of a call changed to one that takes more (a binary operation read as ternary), fewer, and a matrix query
    for (name, at, was, now, expected) in [
        (
            "waves.od.dxil",
            3767,
            0x7a,
            0x34,
            "dx.op 40 called with 2 arguments, where it takes 3",
        ),
        (
            "ray_subobjects.closest_hit.dxil",
            3434,
            0x56,
            0x12,
            "dx.op 10 called with 1 arguments, where it takes 2",
        ),
        (
            "ray_query_grid.dxil",
            4304,
            0xe8,
            0xbe,
            "dx.op 188 called with 2 arguments, where it takes 4",
        ),
        (
            "lane_2d.dxil",
            3057,
            0x3d,
            0x1a,
            "dx.op 42 called with 2 arguments, where it takes 3",
        ),
        (
            "image_array_store.od.dxil",
            3093,
            0x0a,
            0x8a,
            "dx.op 98 called with 1 arguments, where it takes 2",
        ),
    ] {
        let bytes = changed(name, at, was, now);
        assert!(refusal(&bytes).contains(expected), "{name}: {}", refusal(&bytes));
    }
}

#[test]
fn a_global_that_is_not_a_global_is_refused() {
    let bytes = changed("ray_tracing.any_hit.od.dxil", 3228, 0x00, 0xf2);
    assert!(refusal(&bytes).contains("a global that is not a global"));
}

#[test]
fn a_graphics_program_without_its_signature_part_is_refused() {
    // the directory entry of the input signature renamed
    let mut bytes = fixture("arrays.66.dxil");
    let at = bytes
        .windows(4)
        .position(|w| w == b"ISG1")
        .expect("the container has an input signature");
    bytes[at] = b'X';
    assert!(refusal(&bytes).contains("without its ISG1 part"));
}

#[test]
fn a_call_that_reads_an_argument_it_does_not_have_is_refused() {
    // an operation's opcode made one the pre-scan reads an argument of, which the call does not have
    let bytes = changed("geometry.66.dxil", 4132, 0x32, 0x33);
    assert!(
        refusal(&bytes).contains("an operation reads argument 1, where the call has 1"),
        "{}",
        refusal(&bytes)
    );
}

#[test]
fn a_call_made_to_return_a_value_it_does_not_define_is_refused() {
    for (name, at, was, now) in [
        ("ray_query_hits.dxil", 4274, 0xbb, 0xbc),
        ("ray_query_hits.od.dxil", 4330, 0xbb, 0xbc),
        ("tessellation.vs.od.dxil", 3848, 0x0b, 0x0c),
        ("tessellation.hs.dxil", 4165, 0x5c, 0x5d),
        ("vertex.66.dxil", 4963, 0x70, 0x60),
    ] {
        let bytes = changed(name, at, was, now);
        assert!(
            refusal(&bytes).contains("an instruction that yields a value but defines none"),
            "{name}: {}",
            refusal(&bytes)
        );
    }
}

#[test]
fn an_integer_constant_of_a_float_type_is_refused() {
    for (name, at, was, now) in [
        ("blend.dxil", 4954, 0x8c, 0x0c),
        ("geometry.od.dxil", 2606, 0x73, 0x33),
        ("half_ops.dxil", 3146, 0x63, 0x43),
        ("half_ops.od.dxil", 3210, 0x63, 0x43),
        ("ray_query_grid.dxil", 2656, 0xe6, 0x66),
        ("tessellation.hs.od.dxil", 4043, 0x46, 0x44),
    ] {
        let bytes = changed(name, at, was, now);
        assert!(
            refusal(&bytes).contains("an integer constant of a type that holds none (Some(Float))"),
            "{name}: {}",
            refusal(&bytes)
        );
    }
}

#[test]
fn a_lane_past_the_four_of_a_register_is_refused() {
    // a signature cell read at a column the element, and the register, do not have (it used to shift a mask by that much)
    for (name, at, was, now, cell) in [
        (
            "tessellation.hs.dxil",
            4213,
            0x65,
            0x66,
            "column 9 of an element that starts at row 0",
        ),
        (
            "tessellation.hs.od.dxil",
            4314,
            0x80,
            0x00,
            "column 25 of an element that starts at row 3",
        ),
    ] {
        let bytes = changed(name, at, was, now);
        let detail = refusal(&bytes);
        assert!(
            detail.contains(cell) && detail.contains("a register has columns 0 to 3"),
            "{name}: {detail}"
        );
    }
}

#[test]
fn a_float_constant_of_an_integer_type_is_refused() {
    let bytes = changed("ray_tracing.callable.od.dxil", 2520, 0x44, 0x04);
    assert!(
        refusal(&bytes).contains("a float constant of a type that holds none (Some(Integer(32)))"),
        "{}",
        refusal(&bytes)
    );
}

#[test]
fn a_handle_annotated_with_itself_does_not_loop() {
    // an annotated handle whose source is its own result: following the chain of annotations never reaches a handle
    let bytes = changed("pixel_ops.66.dxil", 5046, 0xb1, 0xb0);
    assert!(
        refusal(&bytes).contains("read before it is defined"),
        "{}",
        refusal(&bytes)
    );
}
