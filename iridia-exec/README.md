# iridia-exec

A bounded CPU reference executor for [`iridia::Program`](../iridia): it runs a lifted shader on explicit inputs, whatever bytecode it was lifted from, and returns exactly what it wrote. It is for checking a translation (against values worked out by hand, or against Microsoft's WARP rasterizer), not for rendering: it is deterministic and slow.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. This crate depends on `iridia`; the readers (`iridia-dxbc`, `iridia-dxil`, `iridia-d3d9`, `iridia-mojo`) do not depend on it.

## What it runs

- **Entry-point programs** (pixel, vertex, compute, mesh). `Executable::new` checks a program and compiles it once into a flat form (typed operations on four-lane word registers, pointers folded to word offsets, dead operations dropped). A `Runner` executes one invocation, a 2 x 2 quad, or several invocations in lockstep (masked where they diverge, each with its own steps and budget) against a `Backend`, which answers what is outside the program: texture samples, loads, queries, unordered-access effects. `ProgramShader` puts a vertex or pixel program behind the `PixelShader` trait, in the register vocabulary of the D3D bytecode it came from: inputs and outputs by signature register, constant buffers and samplers by bind slot.
- **Stages WGSL has no entry point for.** Their programs lift as functions, which `Executable::functions` compiles: `geometry` (the input assembler's topologies and a geometry program's strips), `tessellation` and `tessellator` (hull and domain programs over a port of Microsoft's reference fixed-function tessellator), `mesh` (a thread group's meshlet), `Pipeline` with `Scene` (a ray tracing pipeline: `DispatchRays`, `TraceRay`, hit groups, miss and callable programs, over acceleration structures the caller builds; inline ray queries too).
- **`draw`**: a small software rasterizer for an indexed triangle list (vertex program, clipping, viewport, rasterization, pixel program, depth test, render targets), for re-running a recorded draw on its programs and setting its pixels against the game's.
- **`evaluate`**: a pixel shader over a grid of UVs, to bake a procedural material to textures. The caller states everything the program reads (the constant layout and values, a texture per sampled slot, a value per used input) and which outputs it wants; the evaluator has no policy of its own and refuses what it is not given. `evaluate_periodic` bakes to a texture that tiles.
- **`probe`**: runs one vertex of a vertex program with chosen constants to see what it does with them (what a matrix or a table is).
- **`texture`** and **`views`**: a 2D texture as the executor samples it (levels, colour transfer, the sampler state's addressing and bias, the filtering the caller chooses) and resource views bound by slot.

## Example

Lift a pixel shader and run it once. Its constant buffer holds `f = (0.25, 2, 4, 8)` and `u = (7, 2, 0, 0)`; the HLSL computes `min(f.x, f.y)` and `max(f.x, f.y)` into target 0 and `u.x / u.y` and `u.x % u.y` into target 1:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use iridia_exec::{PixelShader, ProgramShader, Sample};

    let program = iridia_dxbc::lift(&std::fs::read("../iridia-dxbc/tests/fixtures/modes.dxbc")?)?;
    let shader = ProgramShader::new(program)?;

    let bits = |v: [f32; 4]| v.map(f32::to_bits);
    let constants = [bits([0.25, 2.0, 4.0, 8.0]), [7, 2, 0, 0], [0; 4]];
    let inputs = vec![[0u32; 4]; shader.input_count()]; // SV_Position, which the program never reads
    let run = shader.execute(&inputs, &[(0, &constants)], &mut |_: Sample| Err("no textures".into()))?;

    assert_eq!(f32::from_bits(run.outputs[0][0]), 0.25); // min(f.x, f.y)
    assert_eq!(f32::from_bits(run.outputs[0][1]), 2.0); // max(f.x, f.y)
    assert_eq!(&run.outputs[1][..2], &[3, 1]); // 7 / 2, 7 % 2
    Ok(())
}
```

## Guarantees and limits

- What a run does follows the program's `D3dModes`: a program that says its body spells out D3D's integer division, conversions and `min`/`max` runs as written, and one that expects denormals flushed to zero has every 32-bit float operation flush its operands and result (sign kept). Anything the program does that its module cannot make valid (an index past the end of a local array, an integer division by zero in the module) is an error of the run, never a silent value. Errors name their bytecode and location.
- Every run has a step budget (`STEP_LIMIT`).
- Sampling belongs to the caller: a scalar execution cannot infer derivatives, filtering or addressing, so a `Sampler` callback answers texture samples and the caller states the footprint, addressing, filtering and transfer. A quad run supplies the derivatives of its four invocations.
- A construct the executor does not run is refused by `Executable::new` with its name. A ray tracing program is not an entry point; it runs inside a `Pipeline`.
- The tessellator is a port of the reference tessellator Microsoft published with the Direct3D 11.3 functional specification (MIT licence; the notice is kept in `src/tessellator.rs`). Its output is held against that reference's recorded output in `tests/fixtures/tessellator_reference.txt`.

## Tests

`cargo test -p iridia-exec` needs no game files: it runs fixtures through both `iridia-dxbc` and `iridia-dxil` (development dependencies, a cycle cargo allows), holding the interface read from either to the values worked out from the HLSL (`tests/interface.rs`), the evaluator to outputs worked out by hand (`tests/evaluate.rs`), lockstep lanes against one pixel after another with diverging control flow (`tests/lanes.rs`), and the stage drivers to hand-worked meshes, strips, patches and scenes. The comparisons against WARP live in `iridia-dxbc` and `iridia-dxil` (`tests/exec_warp*.rs`).

## Licence

GPL-3.0-or-later, apart from the tessellator port noted above.
