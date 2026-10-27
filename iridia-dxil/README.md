# iridia-dxil

Lifts compiled Direct3D Shader Model 6 programs (DXIL, what `dxc` writes) into [`iridia::Program`](../iridia), a [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module with its interface, and reads a container's reflection, signatures and a ray tracing library's subobjects.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. The container, the LLVM bitcode of its `DXIL` part and the reflection and runtime-data records are read by [`dxil-reader`](https://github.com/Kalmarv/dx); this crate maps them.

## What it reads

A DXIL container of Shader Model 6 (its fixtures are `dxc` output for 6.0 and 6.6) of any stage but amplification: vertex, pixel, compute, hull, domain, geometry, mesh and the ray tracing stages (ray generation, intersection, any-hit, closest-hit, miss, callable), including inline ray queries, wave and quad operations, resource arrays and descriptor heaps, unordered-access counters, `half` (computed as a 32-bit float and rounded to a half after each operation, which a note says) and 64-bit integers (carried as a pair of 32-bit words).

A DXIL function is LLVM's: values in single static assignment form over basic blocks. The lifter turns the blocks into structured control flow (an irreducible graph is refused by name; a phi is a local the edges assign) and reads the `dx.op` intrinsics (inputs and outputs, math, bit operations, constant buffers, samples, loads, derivatives, discard, wave, mesh, ray query and ray tracing operations), `alloca`, `getelementptr`, `load` and `store`. The conventions are the same as [`iridia-dxbc`](../iridia-dxbc)'s, so a program lifted from either has the same shape: varyings unpacked from the signature registers (register and lanes in provenance), constant buffers `cb<n>` as uniform `array<vec4<u32>, rows>` in group 0, textures and buffer views in group 1, samplers in group 2, typed texture views in group 3 and the hidden counters of unordered-access views in group 4, binding = register, and Direct3D's numeric results written out (`D3dModes::ALL`).

Reflection without a lift: `interface(container)` reads the stage, the signatures (`ISG1`/`OSG1`/`PSG1`, with each pixel input's interpolation from the entry point's metadata), the resources (`dx.resources`) and the constant buffer layouts (`dx.typeAnnotations`, named from the `STAT` part when the container has one; the program's own metadata carries no names), and `subobjects(rdat)` maps the subobjects of a ray tracing library from its runtime data (`RDAT`).

## Example

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let container = std::fs::read("tests/fixtures/flow.dxil")?; // `dxc -T ps_6_0` output
    let program = iridia_dxil::lift(&container)?;

    assert_eq!(program.stage, iridia::Stage::Pixel);
    assert_eq!(program.provenance.target, "ps_6_0");
    // two colour targets, a float4 and a uint4, as the HLSL declares
    let targets: Vec<_> = program.interface.outputs.iter().map(|v| (v.system, v.value.width)).collect();
    assert_eq!(
        targets,
        [(Some(iridia::SystemValue::Target(0)), 4), (Some(iridia::SystemValue::Target(1)), 4)]
    );

    let wgsl = iridia::emit::wgsl(&program)?;
    assert!(wgsl.contains("@fragment"));

    // a refusal names the bit it is at and why
    let refusal = iridia_dxil::lift(&container[..64]).unwrap_err();
    assert_eq!(refusal.bytecode, "DXIL");
    Ok(())
}
```

## Guarantees

- The interface lists every element of the container's signatures, `used` where the program reads or writes a cell of it (the program's own reading, not the compiler's masks).
- Each `dx.op` is read by position after one check of its argument count (`src/lift/dxop.rs`; the counts are DXC's, from `hctdb.py`, and every one equals how the real-data modules call it). `tests/malformed.rs` makes the damaged cases from committed fixtures; a function body naming something the function does not have is refused.
- Anything not carried is refused by name with the bit it is at, never dropped: a double, 64-bit integer division and conversions between 64-bit integers and floats, an irreducible loop, a wave or quad operation of a kind not lifted, amplification programs (`DispatchMesh`), per-primitive mesh outputs, a mesh payload read, clip or cull distances of a mesh program, an output written at a computed row, and so on. A geometry program's stream other than 0 is refused.
- A ray tracing program has no WGSL form (WebGPU has no ray tracing pipeline): `iridia::emit::wgsl` refuses it, and `iridia-exec`'s `Pipeline` runs the lifted functions.

## Tests

`cargo test -p iridia-dxil` needs no game files:

- `tests/fixtures.rs` lifts our own HLSL (`tests/fixtures/*.hlsl`, each with the `dxc` command that built its `.dxil`): the lifted interface (signature packing of every element, `used`, resource names and slots, buffer field names and offsets), execution on `iridia_exec::ProgramShader` against outputs worked out from the HLSL, every truncation of a fixture refused, and a seeded mutation run that must never panic (the debug key `dxil.fuzz_runs`, default 3000). `tests/semantics.rs`, `tests/mesh.rs`, `tests/pipeline.rs`, `tests/geometry.rs`, `tests/rdat.rs` and the others do the same for their areas; `tests/versions.rs` is the version gate against `tests/lifted.hashes`.
- `tests/exec_warp*.rs` render the fixtures on Microsoft's WARP rasterizer (Direct3D 12) through a separate host program and compare against the executor lane by lane, bit for bit (the float lanes of the fixture that uses `exp2`, `log2` and `rsqrt` within 1e-5 relative). They need `tools.warp` (not part of this repository) and `tools.dxc`.
- `tests/lift_corpus.rs` lifts a folder of real containers (`oracles.corpus.skate/shaders`, `elden/shaders`), validates and writes each and prints the refusals by reason.

The inputs the tests read from `GAME_DATA` are listed in [`tests/inputs/README.md`](tests/inputs/README.md).

## Licence

GPL-3.0-or-later.
