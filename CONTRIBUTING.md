# Contributing

## Build and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Rust 1.99 or newer (`rust-toolchain.toml` pins it). The tests need no game files: the fixtures are small compiled shaders and the HLSL they were built from, in each crate's `tests/fixtures`. Tests that read files this repository cannot ship skip with one line naming the input; the [README](README.md#running-the-tests) says how to point them at your own game files with `GAME_DATA`, and each crate's `tests/inputs/README.md` lists what its tests read.

The examples in the READMEs are checked: each crate's README is run as a doc test, and the repository README must contain `iridia/examples/lift.rs` exactly (`iridia/tests/readme.rs`).

## No I/O in libraries

The library crates read the bytes and text they are given and never touch files, the environment or processes: `clippy.toml` disallows those calls and every `lib.rs` denies the lint. Examples and tests may.

## Changing what a lifter produces

A cache keyed by a program's bytes trusts `LIFTER_VERSION` (each reader) and `PROGRAM_VERSION` (`iridia`) to know that a lifted program is stale. Each reader's `tests/versions.rs` (or `*versions.tsv` in `iridia-d3d9`, `tests/lift.rs` in `iridia-mojo`) hashes what it makes of its fixtures and compares the table with the committed one (`tests/lifted.hashes`). If you change what a lifter makes of an input, the test fails and says which version to bump; after the bump it fails once more until you replace the committed table with the current one, whose path the failure names.

## Reporting a file it cannot read

Open an issue with the "unreadable file" template, for a shader that is refused when it should lift, panics, fails validation, or lifts to something that does not behave as the original. It asks for:

- the version of the crates (or the commit) and your platform;
- which reader you used (`iridia-dxbc`, `iridia-dxil`, `iridia-d3d9`, `iridia-mojo`) and the call (`lift`, `reflection`, `Executable::new`...);
- the exact message: a refusal says the bytecode, the unit (word, bit or line), the offset and the reason; a validation failure lists its problems;
- the shader's target and compiler (`ps_5_0` from `fxc`, `cs_6_6` from `dxc`...) and how you got it: which game or engine, if any, and whether the shader came compiled with it;
- if you can write one, the smallest HLSL (or MojoShader GLSL) that still fails, and the command that compiled it.

Do not attach game files, shader containers extracted from a game, archives or keys: they are not ours to redistribute and the report does not need them. Describe the file instead: its size, its target, the refusal's text, and what Microsoft's disassembler (`fxc /dumpbin`) shows for the instruction at the offset the refusal names. A small HLSL file you wrote and compiled yourself is welcome.

## Changing the code

Keep the public items documented, refusals naming the offset and what was expected, and add a test next to the behaviour you change. Prefer a fixture built from HLSL in `tests/fixtures` (with the compile command in its header) to a binary you cannot reproduce.
