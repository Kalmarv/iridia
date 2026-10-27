# iridia

The shader program that iridia's bytecode readers lift into: a [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module together with the interface, conventions and provenance a module cannot say, plus the validator, the WGSL writers and the typed builder around it.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. The readers are the crates [`iridia-dxbc`](../iridia-dxbc), [`iridia-dxil`](../iridia-dxil), [`iridia-d3d9`](../iridia-d3d9) and [`iridia-mojo`](../iridia-mojo); the vocabulary of the interface is [`iridia-interface`](../iridia-interface), re-exported here; running a program is [`iridia-exec`](../iridia-exec).

## What is in it

- **`Program`**: the `naga::Module`; its `Stage` and entry point; the `Interface` (inputs and outputs by semantic name, each with its D3D register and lanes in provenance; constant buffers with explicit byte layouts; textures, samplers, buffers); `bindings` (which module global each resource is); the `Conventions` it was written for (clip depth, fragment and texture origin; `Conventions::D3D`); `D3dModes` (which of Direct3D's numeric behaviours the body spells out); `Provenance` (bytecode format, target, lifter and version, SHA-256 of the source); and `notes` (what a lifter approximated, each once).
- **`Program::validate`** runs naga's validator with a stated set of capabilities, then checks the interface against the entry point and the resources against the globals (kind, space, binding, layout size). Every problem is reported.
- **`Program::text`** is a stable text form: a header stating all of the above, then the module as WGSL, or, for a program WGSL cannot express, as naga's IR dump under a line saying so. Meant for diffs and logs.
- **`emit::wgsl`** writes WGSL through naga's backend and refuses a program it cannot express. `emit::wgsl_for` writes it as one pipeline binds it: resources at the groups and bindings you choose, a vertex program's outputs at the locations of the pixel program that reads them, clip distances as plain varyings for a pipeline without the feature. `emit::wgsl_functions` writes the functions of a program that has no entry point (hull, domain, geometry, ray tracing).
- **`entry_io`** is the one statement of how the entry point carries the interface: one argument per input varying, one result member per output varying (a vertex stage's clip distances together), named `<NAME><index>`, built-in bindings for system values, locations in interface order, a pixel `Target(n)` always four lanes.
- **`build::Builder`** is the typed builder over naga's IR that the readers share. An operation asked of values that cannot do it (a 2-lane vector added to a 3-lane one, a lane that is not there) does not panic: the builder records the first such fault, returns a value of the shape it was asked for, and `Builder::check` turns the fault into a refusal naming the instruction. `build::behaviour` has Direct3D's numeric results as operations; `build::structure` turns a control flow graph into structured control flow (Ramsey's translation of a reducible graph; an irreducible one is refused by name).
- **`d3d`** maps Direct3D's enum numbers and packing rules (signature elements, resource types, constant buffer layouts) into the interface vocabulary. A value it does not map is an error naming it, never a guess.
- **`Refusal`**, **`gate`**, **`source_hash`**: what every reader returns for bytecode it cannot lift (bytecode, unit, offset, reason, with `group()` to count refusals by cause); the version check readers' tests use (`gate::table` and `gate::check` compare hashes of what a lifter makes of its fixtures against a committed table, so a changed output needs a bumped `LIFTER_VERSION` or `PROGRAM_VERSION`); and the hash of the bytes a program came from.
- **Analyses that read a program without running it**: `flow` (a value graph of what each output lane and each discard is made of, and which inputs and texels may reach them), `Program::constant_reads` (which words of its constant buffers a program reads), `matrix::lines` (whether a constant buffer holds a matrix by rows or by columns, read from the arithmetic), `output_stores` and `output_uses` (what the entry point stores into each output lane and how it is made; what a discard or blend shows an output lane is for), and `surface` (a pixel program's outputs read as a surface, packed for a GPU composite with the WGSL that decodes it).
- **`rt`**: WGSL and naga have no ray tracing pipeline, so a ray tracing program calls functions of its own module that have no body (`trace_ray`, `report_hit`, `call_shader`, `ignore_hit`, `accept_hit_and_end_search`); the call is the operation, and `rt` names them.

## Example

Lift a compiled pixel shader (`fxc /T ps_5_0` output), check it and write WGSL. `iridia-dxbc` is a development dependency of this crate; a program of your own depends on the reader it needs directly:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let container = std::fs::read("../iridia-dxbc/tests/fixtures/modes.dxbc")?;
    let program = iridia_dxbc::lift(&container)?;

    assert_eq!(program.stage, iridia::Stage::Pixel);
    assert_eq!(program.provenance.target, "ps_5_0");
    // the constant buffer `C` of the HLSL source, named by the container's reflection
    assert_eq!(program.interface.resources[0].provenance.name, "C");

    program.validate().expect("a lifted program validates");
    let wgsl = iridia::emit::wgsl(&program)?;
    assert!(wgsl.contains("@fragment"));
    Ok(())
}
```

`cargo run -p iridia --example to_wgsl -- shader.dxbc` and `--example to_spirv -- shader.dxbc out.spv` do the same from the command line; the second writes SPIR-V with naga's own writer.

## Guarantees and limits

- A program that comes out of a lifter validates. A writer that cannot express a program returns an error instead of an approximation.
- Reading is not running: nothing here executes a program (that is `iridia-exec`), and the flow analyses are may-reach over-approximations (a constant that is zero at run time still carries what it multiplies; the answer is never "must reach").
- `flow` follows locals, branches and loops of the lifted module but not a write to a storage buffer or image into a later read of it; a call, a ray query or a cooperative-matrix expression is refused with an error rather than dropped.
- Compute programs are validated against Direct3D's thread group limits (1024 invocations), not WebGPU's guaranteed ones (256); a program beyond those says so in its `notes` (`compute_limit_notes`), and a consumer checks them against its device.
- The vocabulary has no opinion about what a program's inputs or outputs mean: that is for whoever drives it.

## Tests

`cargo test -p iridia` needs no game files. The fixtures it lifts are the small compiled shaders in `tests/fixtures` and the readers' own fixtures (`iridia-dxbc` and `iridia-dxil` are development dependencies, a cycle cargo allows). See the [repository README](../README.md) for the real-data tests.

## Licence

GPL-3.0-or-later.
