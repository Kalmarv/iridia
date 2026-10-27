//! What the builder does when a lifter hands it values that cannot do what it is asked: the fault waits for the lifter to
//! ask for it, and the builder is never left to panic on a program's account.
use iridia::build::{Builder, Kind, Width};
use naga::{BinaryOperator, MathFunction};

fn vector(b: &mut Builder, width: usize) -> iridia::build::Val {
    let one = b.lit_f(1.0);
    b.compose(&vec![one; width])
}

#[test]
fn an_operation_on_values_of_other_shapes_is_a_fault_the_lifter_asks_for() {
    let mut b = Builder::new();
    assert!(b.check().is_ok());
    let (two, three) = (vector(&mut b, 2), vector(&mut b, 3));
    b.binary(BinaryOperator::Add, two, three);
    let fault = b.check().unwrap_err();
    assert!(fault.contains("2 lanes where 3 are needed"), "{fault}");
    // asking takes it
    assert!(b.check().is_ok());
    // the first of several is the one kept
    b.lane(two, 2);
    b.swizzle(three, &[0, 3]);
    let fault = b.check().unwrap_err();
    assert!(fault.contains("lane 2 of a value of 2 lanes"), "{fault}");
    assert!(b.check().is_ok());
}

#[test]
fn a_vector_of_no_lanes_or_too_many_and_a_function_of_no_arguments_are_faults() {
    let mut b = Builder::new();
    b.compose(&[]);
    assert!(b.check().unwrap_err().contains("no lanes"));
    let one = b.lit_f(1.0);
    b.compose(&[one; 5]);
    assert!(b.check().unwrap_err().contains("5 parts"));
    let word = b.lit_u(1);
    b.compose(&[one, word]);
    assert!(b.check().unwrap_err().contains("scalars of one kind"));
    b.math(MathFunction::Abs, &[]);
    assert!(b.check().unwrap_err().contains("no arguments"));
    b.swizzle(one, &[]);
    assert!(b.check().is_err());
    assert_eq!(b.ty_of(Kind::F, Width::One), b.ty_of(Kind::F, Width::One));
}

/// A fault nobody checked, or control flow left open, is an error of the finishers, not a panic.
#[test]
fn a_builder_with_a_fault_nobody_asked_for_cannot_be_finished() {
    let mut b = Builder::new();
    let (two, three) = (vector(&mut b, 2), vector(&mut b, 3));
    b.binary(BinaryOperator::Add, two, three);
    let error = b.finish_functions().unwrap_err();
    assert!(
        error.contains("did not check") && error.contains("lanes"),
        "{error}"
    );
    let mut open = Builder::new();
    open.blocks.push(naga::Block::new());
    let error = open.finish_functions().unwrap_err();
    assert!(error.contains("control flow left open"), "{error}");
}

/// A width is one to four lanes and nothing else: every way of naming one agrees, and any other count is refused.
#[test]
fn a_width_is_one_to_four_lanes() {
    for (n, width) in (1u8..).zip(Width::ALL) {
        assert_eq!(Width::new(n), Some(width));
        assert_eq!(Width::of_count(usize::from(n)), Some(width));
        assert_eq!(Width::try_from(u32::from(n)), Ok(width));
        assert_eq!((u8::from(width), width.count()), (n, usize::from(n)));
        assert!(width.lanes().eq(0..n));
        assert_eq!(
            width.vector_size().map(Width::of_vector),
            (n > 1).then_some(width)
        );
    }
    for n in [0u8, 5, u8::MAX] {
        assert_eq!(Width::new(n), None);
        let error = Width::try_from(n).unwrap_err();
        assert!(error.contains(&format!("{n} lanes")), "{error}");
    }
    assert!(Width::try_from(u32::MAX).is_err() && Width::try_from(usize::MAX).is_err());
}
