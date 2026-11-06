# Changelog

## Unreleased

- `iridia-naga`: `wgsl_for(program, profile, &Pipeline { plan, feeds })` writes a program for one pipeline: each
  resource where the pipeline's `BindingPlan` binds it, and a vertex program's outputs at the locations, interpolation
  and types of the pixel program it feeds (matched by semantic, as Direct3D links stages; padded, narrowed or bitcast
  where the two differ, each noted). An input no output carries is refused by name.
- `iridia-dxil` refuses a volatile or atomic load or store by name, which it lowered as a plain one before; none
  of the real corpora's programs has one.

## 0.5.0 (2026-11-06)

- `iridia-d3d9-ir` lifts every Direct3D 9 model from `vs_1_1` and `ps_1_1` to `vs_3_0` and `ps_3_0`: the pixel 1.x
  texture operations, co-issue, modifiers and shifts under `Options::ps1x` (modern drivers' policy by default), integer
  and boolean constants, flow control and predication, `aL`-relative inputs and 3.0 outputs, derivatives and explicit
  level and gradient sampling. Its CTAB reading lays out structs, arrays and matrices of every register set as the
  compiler allocates them. A multiply is zero-wins by default (`Options::multiply`, IEEE by choice). Projection leaves
  a cube coordinate as it is, and `texdepth` takes r5.g as 1 past 1, as wined3d. `texkill` demotes the pixel, which
  goes on as a helper its quad's derivatives read. `LIFTER_VERSION` is 4.
- `iridia-ir`: `Node::Demote` discards a pixel program's invocation without ending it (a helper of its quad from
  there); `End::Kill` demotes and ends. `iridia_ir::VERSION` is 2. `iridia-naga` writes `Demote` as WGSL's `discard`
  and `Kill` as `discard` and a return.
- `iridia-exec-ir` runs a pixel program's 2 x 2 quad (`run_quad`): coarse and fine derivatives, implicit-LOD samples
  with the quad's derivatives (`Lod::Grad`, `Lod::GradBias`), demoted lanes as helpers, divergent lanes refused;
  `needs_quad` says whether a program needs it.
- `iridia-d3d9-ir`'s tests: 105 one-feature programs assembled by Microsoft's `D3DAssemble`
  (`tests/assembled`, rebuilt by `tests/oracles/assemble.py`) run as their comments state, and vkd3d's Direct3D 9
  shader tests (`tests/vkd3d`) run and probe, a program with a derivative as a quad; each in the IR, its control flow
  graph, that graph structured again and WGSL. `tests/versions.tsv` gates the lifter version on all of them, 1,209
  programs.
- `iridia-passes`: `flatten` turns structured regions into a control flow graph, the inverse of `structurize`, which no
  longer drops a body that is already structured; `specialise` takes away the textures only a folded branch sampled.
- `iridia-naga` reads a constant at a fixed address at any 32-bit word, and a zero-wins multiply of vectors validates.
- `iridia-dxbc` reads the typed declarations of `dx` at `9b29341` (`dcl_index_range`, `dcl_stream`); the four `dx`
  crates are pinned at `9b29341`.
- `iridia-exec-ir`: `Bound::denormals` is the denormal rule a program that leaves it open runs by (flushed or kept).
- `iridia_ir::gate::check` names at most a dozen fixtures of each kind and counts the rest.
- `iridia-d3d9` reads its tokens through `dx`'s new `d3d9-reader` (pinned by git rev), which reads every Direct3D 9
  version as stored; the crate's own token reader and CTAB reader are gone, and `ConstantTable`, `Constant`, `Type`,
  `TypeMember` and `Comment` are `d3d9-reader`'s, re-exported (`Constant` gains `word`). `read` and `lift` take the
  same `vs_3_0`/`ps_3_0` subset as before and lift every program identically, but a malformed stream is now refused
  with the reader's account of it (an opcode or register of the other stage, an `aL` or address token that is not
  `a0`/`aL`, a relative address without its token), so `LIFTER_VERSION` is 3 and the `*versions.tsv` tables are
  regenerated. `frame` reads the program with `d3d9_reader::frame` and refuses versions other than `vs_3_0`/`ps_3_0`.
  The four `dx` crates are pinned at `5b58dd3`.
- `iridia-dxbc` lifts SM5.1 resource ranges and spaces, including dynamically indexed and unbounded
  CBV/SRV/UAV/sampler arrays and non-uniform descriptor selection. Declaration IDs are distinct from
  registers and reflection must agree with the declarations. `LIFTER_VERSION` is 10.
- `iridia-exec` binds constant-buffer arrays through `Resources::constant_elements` and
  `Executable::bind_resource_arrays`. Buffer arrays and their compute writes use sparse element maps,
  including unbounded arrays. `iridia-dxil` also lifts constant-buffer arrays (`LIFTER_VERSION` 9),
  and program validation admits their binding-array capabilities and structure-wrapped rows.
- Focused fxc SM5.1 fixtures are compared instruction by instruction with fxc disassembly and run on
  the CPU against DXIL builds of the same source, plus SM5.0 where expressible. Their inputs and
  expected outputs are saved beside the fixtures for external comparisons.
- The three `dx` readers were pinned at `292b616`, which reads Shader Model 5.1. `iridia-dxbc`'s `reflection` and
  `resource_reflection` read 5.1 pixel, vertex and compute programs: register spaces, resource arrays (an array of no
  stated size is `Resource::UNBOUNDED`) and the constant buffer variables the program never reads. Two fxc
  10.0.26100.0 fixtures, `reflection_sm51` and `bindings_sm51`, check reflection and declaration resolution.
- `iridia_exec::value`: `Evaluator` evaluates a node of a program's value graph from the words of the leaves it
  reads, with the executor's own operations and word semantics (the operation tables `compile` and `run` use, now
  shared): what a run computes for that lane where it is given the same leaves, bit for bit. The caller answers each
  leaf (an input lane, a supplied value, a constant word, a run-time indexed constant at its index, a texel at its
  coordinate, a query) or leaves it unknown; a node that needs an unknown leaf is `Evaluated::Unknown`, naming the
  leaf, and a select whose condition is unknown is known where both its values are the same bits. A loop's value, a
  merge whose arms differ, an element picked by a run-time index, a derivative, a subgroup operation and a math
  function that mixes lanes (other than a dot product) are refused by name.
- `iridia::flow::Values::reads`: the scalar kind an operation reads its operands in. Two operations alike but for
  that kind are two nodes now: a comparison of the same bits as floats and as integers (a bitcast is no operation in
  the graph) was one node, read as whichever was interned first.
- Tests: `iridia/tests/flow.rs` reads `fixtures/flow_kinds.hlsl`'s comparisons as floats, unsigned and signed
  integers as three nodes; `iridia-exec/tests/value.rs` evaluates every output lane of `fixtures/value.hlsl` (a
  palette row picked by a record number rounded from an input, a saturated blend, comparisons in two kinds) against a
  run of the program, bit for bit, across rounding, conversion saturation and wrap, NaN and out-of-range slots.
- `iridia::blend`: layers a program lays over one another, read from its value graph with no game in it. `Over` is
  a lane made of a chain of interpolations (`weight * (layer - below) + below` in any operand order, the difference
  also as `layer + -below`, or naga's `mix`): its base and each layer's weight and value, bottom first; `Over::under`
  finds the first chain under an output lane (beneath a saturate, a tint). `Scalar` is a value as the float operations
  it is made of (add, subtract, multiply, min, max, saturate, constants), each one of the program's own operations in
  its order, down to leaves the caller names (input lanes, texels, constant words). Direct3D's results the lifters
  spell out are read back as the operation they spell: the NaN-guarded saturate (NaN gives 0) and the guarded `min`
  and `max` (a NaN operand gives the other); an unguarded one is a leaf. `Scalar::evaluate` computes it under those
  rules.
- `iridia::flow::Values::kind`: the scalar kind of an operation's result and of a literal. A float and an integer
  operation of the same operands are two nodes now (they were one), and an integer literal a float operation reads
  through a bitcast is told from a float literal of the same number.
- Tests: `iridia/tests/blend.rs` reads `fixtures/blend.hlsl`'s layers as fxc compiles them, checks them against the
  source by hand, and evaluates what it read against `iridia-exec` running the program, bit for bit (NaN, infinity
  and weights past 0 and 1 included).
- `iridia::specialise`: a word read through an element indexed by a value the fold knows (`Access`, as `iridia-dxbc`
  addresses a constant buffer's rows) folds as one read through a constant index (`AccessIndex`) does. Before, no
  program `iridia-dxbc` lifted folded a stated word: its specialisations returned the program with the words selected
  and nothing decided. Signed integers fold too (their bitcasts, arithmetic and comparisons, and unsigned comparisons),
  so a signed word compared with a number decides its select.
- `iridia::specialise::specialise` refuses a result that leaves a branch or select the selected words decide untaken
  (a read the fold cannot follow, or a value it cannot evaluate), naming the words, instead of returning it.
  `feature_words` no longer selects a word for a select whose arms are the same value.
- Tests: `iridia/tests/specialise.rs` specialises `fixtures/specialise.hlsl`'s entry points as fxc and dxc compile them.
- The tests depend on `primase-testdata` 0.1.8 from crates.io instead of a git revision.
- Depends on dx 0.5.0 (`d3d9-reader`, `dxbc-reader`, `dxcontainer`, `dxil-reader`) and primase-naga 30.0.1 from crates.io
  instead of git revisions (the same sources).
- First releases of `iridia-ir`, `iridia-passes`, `iridia-naga`, `iridia-exec-ir` and `iridia-d3d9-ir`.

## 0.4.1 (2026-10-29)

- `iridia::specialise`: specialises a program for one material's stated uniform words. `feature_words` finds the
  words a program's branches read among those known, and `specialise` folds them as constants into a copy of the
  program (`Specialised`: the program, the words used, the resources no longer read, and how many values were folded).
  Only the words given become constants; the original program is unchanged, and control flow the pass does not handle
  is refused, naming it, rather than guessed. `iridia/tests/specialise.rs` runs each original and specialised program
  through `iridia-exec` on the same inputs and requires equal outputs.
- `iridia-exec` compiles a named constant (a vector constant expression) as an initial value.
- Depends on `primase-testdata` 0.1.3.

## 0.4.0 (2026-10-29)

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
- Depends on `primase-foundations` at a revision whose filters read a border colour, and on `dx` at the revision
  with `texture_reads`, so that a crate reading programs with that `dx` passes them to these crates.

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
