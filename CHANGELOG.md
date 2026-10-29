# Changelog

## Unreleased

- `iridia-dxbc`: the per-opcode operand counts the lifter checked itself (`arity.rs`) are `dxbc-reader`'s operand formats now,
  which `dxbc_reader::decode` checks for every instruction; a program whose instruction has another count is refused by
  the decoder, naming the word, instead of by the lifter. Depends on the `dx` revision with the formats and the container
  writer.
- `iridia-d3d9`: vertex programs lift `slt`, `sge`, `mova` (the address register `a0`, rounded to the nearest integer, a
  half away from zero as Microsoft's reference rasterizer rounds it) and float constants addressed relative to it
  (`c[a0.x + N]`, zero outside the 256 constant registers, a `def` constant returned for the register it names); `sincos`
  lifts in both stages. **Breaking:** `Source` has a new field `relative` and `Register` a new variant `Address`. The
  1,099 vertex and 6,207 pixel programs of Shaun White Skateboarding's three SDB2 bundles carry `vs_3_0` and `ps_3_0`
  only, no Shader Model 1 or 2 program (no corpus we hold carries one), and every one lifts and equals Microsoft's disassembly
  line for line (`iridia-d3d9/tests/corpus.rs`).
- Tests: `iridia-dxbc/tests/vkd3d_oracle.rs` compares the containers, signatures and instruction mnemonics of the games'
  Shader Model 4 and 5 programs with vkd3d's `vkd3d-dxbc` and `vkd3d-compiler` run as programs; `operand_format.rs` checks the
  operand formats against the programs; `container_digest.rs` (both crates) now also writes every container back byte for byte.
- `iridia-interface`: `SamplerState` states a border colour (`border`, for border addressing) and a level range
  (`lod`, the least and greatest mip level read), each `None` where the source states none, and `SamplerState::verify`
  checks a state agrees with itself (finite colour, bias and range, an ordered range, an anisotropy of at least 1).
  Breaking: a struct literal of `SamplerState` needs the two fields (or `..SamplerState::default()`).
- `iridia-exec` samples border addressing: `SamplerUse` and `ViewSampler` carry the border colour, and a sample past a
  border axis reads it (through `primase-foundations`' `Addressing`); a border with no stated colour is still refused.
  `SamplerUse` carries the level range too, and the level of detail is clamped to it after the bias (named as not
  applied under base-level filtering where the range excludes the base level). Breaking: `SamplerUse` and
  `ViewSampler` have new fields, and `ViewSampler` is no longer `Eq`.
- Depends on `primase-foundations` at a revision whose filters read a border colour.

## 0.3.0 (2026-10-28)

- `iridia::emit::wgsl_for` drops the types its pipeline changes leave unused before writing: a sampler heap written as
  one sampler (`Pipeline::single_samplers`) no longer leaves its binding-array type behind, for which the writer declared
  `enable wgpu_binding_array;` though no variable is a binding array, and a WGSL front end that does not know the
  extension refused the whole program. Every global, function and named type keeps its place.
- Depends on `dx` 0.3.0, whose readers these crates' public functions take, so this release is breaking.

## 0.2.0 (2026-10-27)

- `iridia-dxbc` lifts a hull or domain program whose patch-constant signature is a `PSG1` part (fxc writes it when the
  program holds a `min16float`), as it lifts the `PCSG` one: the WGSL of the test programs equals their float twins'.
  Depends on `dx` at a revision that reads `PSG1`, the Shader Model 4 `RDEF` and address-space pointer types.
- Depends on `dx` 0.2.0, whose readers these crates' public functions take (`dxbc-reader` signature elements, `dxil-reader` modules), so this release is breaking too.

## 0.1.0 (2026-10-27)

First release.

- `iridia`: the shader program (a naga module with its interface, conventions and provenance), its validator, the WGSL writers, the typed builder the readers share, and analyses that read a program without running it. A builder value's width is a `Width` of one to four lanes, and finishing a module that cannot be finished is an error, not a panic.
- `iridia-interface`: the plain, serializable description of a program's interface (varyings, resources, constant layouts), Direct3D's fixed-function state, and what the analyses observe of a program's code (matrix lines, discard comparisons, output uses). No dependency on naga. What a consumer decides a program's values stand for is not here.
- `iridia-dxbc`: reads Direct3D Shader Model 4 and 5 (DXBC containers) of every stage, with reflection and signatures.
- `iridia-dxil`: reads Shader Model 6 (DXIL containers) of every stage but amplification, with reflection and a ray tracing library's subobjects.
- `iridia-d3d9`: reads raw Direct3D 9 `ps_3_0` and `vs_3_0` token streams and their CTAB constant table.
- `iridia-mojo`: reads the GLSL that MojoShader writes for Shader Model 2 and 3 programs.
- `iridia-exec`: a bounded CPU reference executor with drivers for the geometry, tessellation, mesh and ray tracing stages, and a small software rasterizer.
- Every public item of every library crate is documented, and each library crate denies `missing_docs`.
