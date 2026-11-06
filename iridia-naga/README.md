# iridia-naga

Lowers a program of [iridia's shader IR](../iridia-ir) to a stock [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module, validated, for naga's writers (WGSL).

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. Readers lift into the IR, [`iridia-passes`](../iridia-passes) puts a program in the form naga takes, and this crate lowers it. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate is `no_std` with `alloc` and does no I/O; it depends on naga as published, with no fork.

## What it does

- `module(program, plan)` builds the module: the entry point `main` with an argument per interface input and a member of the result structure `Out` per output (`entry_io` states them), and each resource a global where the `BindingPlan` binds it. `BindingPlan::d3d` puts Direct3D's register classes in bind groups 0 to 3 (`cbN`, `tN`, `sN`, `uN`), each register its binding.
- Operations naga has whole map to one expression; the others expand in the lowering, each in one function: a zero-wins multiply, `min` and `max` with their NaN rule, `saturate` of NaN, a float-to-integer conversion that saturates, rounding with ties away from zero, a constant buffer row read outside the buffer.
- A masked input load is a swizzle of the argument; a masked output store is composed with what the output held (zero before the first store).
- `validate(module, profile)` runs naga's validator with the profile's capabilities; `wgsl(program, profile)` (the `wgsl` feature, on by default) preflights the program, lowers it with the D3D plan, validates and writes the WGSL.
- `wgsl_for(program, profile, &Pipeline { plan, feeds })` writes it for one pipeline: the resources where `plan` binds them (two programs of a pipeline share its bind groups), and a vertex program's outputs where the pixel program it `feeds` reads them, matched by semantic as Direct3D links stages. An output read with more lanes is padded with zeros, with fewer its first lanes, as another 32-bit scalar the same bits; each is noted in `Written::notes`, and an input no output carries is refused by name.

## Example

```rust
use iridia_ir::{
    Conventions, FloatEnv, Program, Provenance,
    interface::{Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying},
};

let entry = iridia_ir::text::parse(
    "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: vec4<f32> = compose %0, %0, %0, %0
  %2: vec4<f32> = saturate.zero %1
  output 0.xyzw = %2
  return
}
",
)
.unwrap();
let varying = |name: &str, width, system| Varying {
    label: Label { name: name.into(), index: 0 },
    system,
    value: Value { scalar: ScalarType::Float, width },
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
        inputs: vec![varying("TEXCOORD", 1, None)],
        outputs: vec![varying("SV_Target", 4, Some(SystemValue::Target(0)))],
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
let wgsl = iridia_naga::wgsl(&program, &iridia_naga::Profile::NAGA).unwrap();
assert!(wgsl.contains("@fragment"));
// WGSL leaves what a clamp makes of NaN to the implementation: the NaN-to-zero rule is spelled out
assert!(wgsl.contains("select("), "{wgsl}");
```

## Guarantees and limits

- A program is lowered for a target `Profile` (`Profile::BROWSER` for WebGPU's core WGSL, `Profile::NAGA` for naga's own dialect, either widened with `Profile::with`): `preflight` refuses a program that needs a capability the profile lacks, naming the value, varying or resource that needs it, and naga's validator runs with the profile's capabilities alone. A module naga rejects after the preflight accepted is reported as a lowering bug.
- It lowers what the Direct3D 9 reader makes, and refuses the rest by name.
- naga's `break` and `continue` reach only the innermost construct: run `iridia_passes::break_flags` first. A `Cfg` body must be structured first (`iridia_passes::structurize`).
- Pipeline state is not lowered: `iridia_passes::specialise` it first.
- WGSL text cannot say a varying has no interpolation, and naga's namer adds a `_` to some names; a reader of the WGSL sees those, not the IR's.

## Tests

`cargo test -p iridia-naga` needs no game files: `tests/lower.rs` pins the WGSL of masked loads and stores and checks that unspecialised state is refused. The Direct3D 9 reader's tests lower every Shaun White Skateboarding program they read.

## Licence

GPL-3.0-or-later.
