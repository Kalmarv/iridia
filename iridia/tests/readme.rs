//! The example in the repository README is the example program, byte for byte.

#[test]
fn the_readme_example_is_the_example_program() {
    let readme = include_str!("../../README.md");
    let example = include_str!("../examples/lift.rs");
    assert!(
        readme.contains(example),
        "README.md must contain iridia/examples/lift.rs exactly"
    );
}
