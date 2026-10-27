# iridia-interface

A shader program's interface as plain data: its inputs and outputs, its resources (constant buffers with explicit layouts, textures, samplers, buffers, storage), Direct3D's fixed-function state, and what iridia's analyses observe of the program's code, independent of any graphics API. It depends on `serde` and nothing else, and every type implements `serde::Serialize`.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. The readers ([`iridia-dxbc`](../iridia-dxbc), [`iridia-dxil`](../iridia-dxil), [`iridia-d3d9`](../iridia-d3d9), [`iridia-mojo`](../iridia-mojo)) map their API's reflection into this once and keep the API's own names and addresses in provenance, so a consumer that only needs meaning (a binding step, a validator, a writer) never sees an API's enum numbers. [`iridia`](../iridia) re-exports it.

## What is in it

- **`Interface { stage, inputs, outputs, resources, facts }`** with one **`Varying`** per label. `label` is the source's own semantic name and index (`TEXCOORD` and 5) and is not interpreted. D3D packs several varyings into one register; here they are separate, and the register and lanes are in `provenance.d3d`. `system` is the `SystemValue` when the pipeline supplies or takes it (`ClipPosition`, `FragCoord`, `FrontFacing`, `VertexIndex`, `InstanceIndex`, `PrimitiveIndex`, `SampleIndex`, `Target(n)`, `Depth`, `Coverage`, the compute, ray tracing and mesh built-ins...). `value` is a `ScalarType` and a width of 1 to 4; `interp` an `Interpolation` of mode (perspective, linear, flat) and sampling (centre, centroid, sample) where the source states it; `used` says the program reads or writes some lane.
- **`Resource { id, kind, count, provenance }`**. `kind` is `ConstantBuffer { layout }`, `Texture`, `Sampler`, `Buffer`, `Storage` or an acceleration structure; `count` is the length of a resource array (`Resource::UNBOUNDED` for one the program does not size); `provenance` holds the source's name and its D3D bind point (class `b`, `t`, `s` or `u`, register, space).
- **Constant layouts.** A `BufferLayout` lists every `Constant` with an explicit byte offset and a `ConstantType` whose matrix lines and array elements have stated strides and matrices their `Majorness`, so D3D packing, std140 and Direct3D 9 registers can all be written in it. `ConstantType::size()` is the extent of the data, with no padding after the last line or element.
- **`StageFacts`**: what a hull, domain, geometry, mesh or ray tracing program states beyond its varyings (tessellation domain, partitioning and control point counts; geometry primitive types and output limits; mesh limits; hit groups).
- **Fixed-function state** (`RenderState` with its blend, depth, stencil, cull, fill and alpha test; `SamplerState` with its filters, addressing and comparison): Direct3D's state, as a source states it, for drawing or sampling a program. The CPU executor (`iridia-exec`) draws and samples with it, and the Direct3D 9 reader reads sampler state.
- **What a program's code shows**, as `iridia`'s analyses read it: `MatrixLines` (whether a matrix constant's lines are rows or columns of the matrix it applies, or why the program does not show which), `AlphaComparison` (what a discard compares a constant word with) and `OutputUse` (which lanes of a render target output a discard tests, and whether the target's blend weighs by its alpha); and `ConstantBufferBinding`, the size a source binds a constant buffer with. These are observations; what an input, a constant or an output stands for (a view-projection matrix, an albedo) is the consumer's to decide, in its own terms.
- **`ShaderFormat`** (how a program's bytes are encoded), **`InstructionCount`** (how large a program is, in its reader's unit).

## Example

An interface is built by a reader, but it is only data; this one is written by hand and serialized:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use iridia_interface::{
        Interface, Label, ScalarType, Stage, StageFacts, SystemValue, Value, Varying, VaryingProvenance,
    };

    let colour = Varying {
        label: Label { name: "SV_Target".into(), index: 0 },
        system: Some(SystemValue::Target(0)),
        value: Value { scalar: ScalarType::Float, width: 4 },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    };
    let interface = Interface {
        stage: Stage::Pixel,
        inputs: Vec::new(),
        outputs: vec![colour],
        resources: Vec::new(),
        facts: StageFacts::None,
    };
    let json = serde_json::to_string(&interface)?;
    assert!(json.contains(r#""stage":"pixel""#));
    Ok(())
}
```

## Guarantees and limits

- Conventions are normalized and the original is kept in provenance: D3D register packing is unpacked into one varying per label, and D3D, std140 and Direct3D 9 constant packings all become explicit byte offsets and strides. Behaviour that differs between APIs is not hidden here at all; that is the program body's business.
- A value a reader cannot map (a `D3D_NAME`, a component type, a resource type with a counter, a `tbuffer`) is an error naming it, never a guess.
- What the readers keep and drop: DXBC states interpolation in the program's declarations, not its signature, so the standalone reflection leaves `interp` empty and `iridia_dxbc::lift` fills it; DXIL states it in the entry point's metadata. A UAV's `access` is `ReadWrite` (neither reflection says more). A structured buffer is its stride; the element layout is not kept. A DXIL `half` and DXBC minimum-precision types are read as 32-bit.
- Every type is `Serialize` only; there is no `Deserialize`.

## Tests

`cargo test -p iridia-interface` needs no game files; the layout arithmetic (`ConstantType::size`) is tested against sizes worked out by hand, and the render state's consistency check has its own unit tests.

## Licence

GPL-3.0-or-later.
