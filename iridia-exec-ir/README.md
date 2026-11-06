# iridia-exec-ir

Runs a program of [iridia's shader IR](../iridia-ir) on the CPU, one invocation or a pixel program's 2 x 2 quad, by the IR's own definition of each operation: the reference the passes and lowerings are checked against.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate does no I/O.

## What it runs

- `run(program, bound)` takes the program and a `Bound`: each input slot's lanes, each constant buffer's rows, each pipeline state resource's value, and a sampler the caller answers each `Sample` with (the texture and sampler ids, the coordinate's lanes, the level of detail). It gives a `Run`: each output slot's lanes as last stored (a lane never stored is zero, a slot never stored `None`) and whether the invocation was killed.
- `run_quad(program, bounds)` runs a pixel program's quad, a `Bound` per lane (lanes 0 and 1 the top row). A derivative is a difference between lanes: the right lane's less the left's along x, the lower's less the upper's along y, of the top row or left column when coarse (and for `Fineness::Any`, as `iridia-exec` takes it), of the lane's own when fine. An implicit-LOD sample is asked with the quad's coarse derivatives of its coordinate (`Lod::Grad`, `Lod::GradBias`). A demoted lane (a discard) goes on as a helper the others still read. The lanes run one after another with no threads, each until it meets a derivative or implicit-LOD sample whose operands are not all known, then again knowing them; lanes that do not meet it at the same operation and loop iteration are refused as divergent, and so is a run of more than `MEETING_LIMIT` meetings. `needs_quad(program)` says whether a program's results depend on its quad.
- Both body forms: every construct of a `Regions` body, and the blocks of a `Cfg` one (block parameters copied in parallel). A run of more than `STEP_LIMIT` loop iterations or block executions is refused.
- The arithmetic is the reference `iridia-exec` uses: IEEE 754 `f32` for the basic operations, the transcendental functions (`exp2`, `log2`, `sin`, `cos`, `sqrt`, `inverseSqrt`, `pow`) in `f64` rounded once, a dot product summed left to right with each step rounded. Denormals are kept. Each operation's enum decides the rest: a zero-wins multiply, a NaN rule of `min` and `max`, ties away from zero.

## Example

```rust
use iridia_ir::{
    Conventions, FloatEnv, Program, Provenance,
    interface::{Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying},
};

let entry = iridia_ir::text::parse(
    "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  %1: vec2<f32> = vec2<f32>(0x7f800000, 3.0)
  %2: vec2<f32> = fmul.zerowins %0, %1
  output 0.xy = %2
  return
}
",
)
.unwrap();
let varying = |name: &str, system| Varying {
    label: Label { name: name.into(), index: 0 },
    system,
    value: Value { scalar: ScalarType::Float, width: 2 },
    interp: None,
    used: true,
    provenance: Default::default(),
};
let program = Program {
    stage: Stage::Pixel,
    entry,
    patch: None,
    workgroup: None,
    interface: Interface {
        stage: Stage::Pixel,
        inputs: vec![varying("TEXCOORD", None)],
        outputs: vec![varying("SV_Target", Some(SystemValue::Target(0)))],
        resources: Vec::new(),
        facts: StageFacts::default(),
    },
    conventions: Conventions::D3D,
    float: FloatEnv::default(),
    provenance: Provenance {
        format: ShaderFormat::D3d9,
        target: "example".into(),
        lifter: "text".into(),
        lifter_version: 1,
        source_hash: None,
        instruction_count: None,
        policies: Vec::new(),
    },
    notes: Vec::new(),
};
iridia_ir::validate(&program).unwrap();

let mut sampler = |_| Err::<[f32; 4], String>("no textures".into());
let inputs = [[0f32, 2.0, 0.0, 0.0].map(f32::to_bits)];
let mut bound = iridia_exec_ir::Bound {
    inputs: &inputs,
    buffers: &[],
    state: &[],
    sampler: &mut sampler,
    denormals: iridia_ir::Denormals::Any,
};
let run = iridia_exec_ir::run(&program, &mut bound).unwrap();
// zero wins: 0 * inf is +0, not NaN; the lanes past the output's width are zero
assert_eq!(run.outputs[0], Some([0, 6f32.to_bits(), 0, 0]));
assert!(!run.killed);
```

## Guarantees and limits

- This subset runs 32-bit scalars and vectors, constant buffer reads, pipeline state, samples, derivatives in a quad, demotion and kill; anything else (memory other than variables and constant buffers, other widths) is refused by name, not approximated.
- A program whose float environment leaves denormals open runs by `Bound::denormals`; one that states a rule runs by it.
- `run` runs one invocation, which has no derivatives: it refuses a derivative, and an implicit level of detail is `Lod::Auto`, for the caller's sampler to decide.
- A quad's sampler may be asked a request again, once per pass that reaches it, and must answer it the same way each time.

## Tests

`cargo test -p iridia-exec-ir` needs no game files: `tests/fixtures.rs` runs programs in IR text on inputs whose results are worked by hand from the operations' definitions (the NaN and rounding rules, zero-wins multiplication, saturating conversion, lane masks, a loop, a kill; a quad's coarse and fine derivatives, a derivative in a loop, a divergent quad refused, an implicit-LOD sample's footprint) and checks the text parser's refusals.

## Licence

GPL-3.0-or-later.
