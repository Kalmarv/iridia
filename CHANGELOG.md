# Changelog

## Unreleased

- `iridia-coverage`: the table names the dx readers' versions as the build takes them (from `Cargo.lock`, where it
  named 0.5.0 after 0.11.0 came in), and every row has a format: a vkd3d run with no program (the compiler refused
  every shader) its runner's, an offload test with no program DXIL's (the suite is read for DirectX with DXC).
- `iridia-ir`: the ray tracing pipeline's programs. `Space::RayPayload` (the payload of the ray a hit or miss program
  runs for, or a callable program's parameter, read and written) and `Space::HitAttributes` (the hit's attributes, read
  alone), each laid out as Direct3D lays out the structure and as many words as `RayTracingFacts::payload` and
  `RayTracingFacts::attributes` state (at most `MAX_RAY_PAYLOAD` and `MAX_HIT_ATTRIBUTES`, 32 bytes);
  `Node::TraceRay` (the acceleration structure, a `Ray` of the call's nine operands and the payload's variable),
  `Node::CallShader`, `Op::ReportHit` (whether the hit was accepted), and `End::IgnoreHit` and
  `End::AcceptHitAndEndSearch` (and `Terminator`'s), an any-hit program's ends. The text form is
  `trace ray r0 flags %f ... payload name`, `call shader %i parameter name`, `reporthit %t kind %k attributes name`,
  `ignorehit` and `accepthit`; the validator holds each operation to the stages Direct3D allows it in. Breaking:
  `RayTracingFacts` has the fields `payload` and `attributes`, and `Space`, `Node`, `End`, `Terminator` and `Op` have
  new variants. `iridia-exec` runs them on the IR (a function's pointer arguments fill the variables of those spaces;
  `Executable::pointer_words` gives a pointer argument's words) and `Pipeline::from_ir` builds a pipeline of IR
  programs. `iridia-naga` refuses them by name.
- `iridia-dxil::ir` lifts the programs of a ray tracing library of one export: the payload and attributes parameters,
  the ray's system values, `TraceRay`, `ReportHit`, `CallShader`, `IgnoreHit` and `AcceptHitAndEndSearch`, a library's
  resources (`CreateHandleForLib` of a resource global), and vectors in memory laid out in bytes. Both lifts of
  `ray_tracing.hlsl`'s pipeline draw what WARP draws, bit for bit alike, and the Elden Ring corpus's 4,281 ray tracing
  programs, which the IR refused before, lift and agree with the naga lift.

## 0.6.0 (2026-11-07)

Breaking changes, for code that uses 0.5.0 (each is in its entry below):

- The dx readers are 0.11.0: the `d3d9-reader` types `iridia-d3d9` and `iridia-d3d9-ir` re-export (`Comment`,
  `Constant`, `ConstantTable`, `Type`, `TypeMember`, `Comparison`, `Opcode`, `Version`) are 0.11.0's, so code that
  builds them with `d3d9-reader` 0.5 needs 0.11.
- `iridia-exec`: `MeshShader::interface` and `GeometryShader::interface` replace `MeshShader::program` and
  `GeometryShader::program`; `DrawShader::draw` (any topology, from `Primitive::assemble`) replaces
  `DrawShader::draw_indexed` (a triangle list); `Rasterizer::viewports` replaces `Rasterizer::viewport`, and
  `Rasterizer` is no longer `Copy`; `DrawStats::primitives` replaces `DrawStats::triangles`; `DepthState` is no longer
  `Eq` (its `bounds` are floats). `Rasterizer`, `DrawStats`, `DepthState`, `Surface`, `Output`, `Execution` and
  `ImageAtomic` have new public fields, and `ImageAtomicOp` the variant `CompareExchange`.
- `iridia-ir`: `Op::LoadBuffer` takes an `address` (`Address`) in place of `index`, `stride` and `offset`;
  `OnZero::ByOne` replaces `OnZero::Dividend` (it applies to `SDiv` and `SRem` too); `Op::Sample` has `compare`,
  `offset` and `clamp`; `Program` has `patch` and `workgroup`, `Var` has `len`, `space` and `init`, `MeshFacts` has
  `payload`. `Op`, `Node`, `Unary`, `Binary` and `Terminator` have new variants, which an exhaustive `match` must take.
  `Stage::Amplification` is before `Stage::RayGeneration`, so the ray tracing stages' discriminants (`as u8`) are one
  more.
- `iridia::flow::Reach` has the field `words`; `iridia-naga`'s `Profile` has `bindings` and `directives`, and
  `BindingPlan` has `counters`: a struct literal of these needs the new fields.

- Contributing: `scripts/check.sh` runs the full check (fmt, clippy, every test, the doc tests, the docs), the tests
  under cargo-nextest where it is installed (`.config/nextest.toml`): about 9 minutes warm on the gate configuration,
  where one `cargo test` took 38.
- The dx readers are 0.11.0 (`d3d9-reader`, `dxbc-reader`, `dxil-reader`, `dxcontainer`; before, 0.5.0). The
  `d3d9-reader` types `iridia-d3d9` and `iridia-d3d9-ir` re-export (`Comment`, `Constant`, `ConstantTable`, `Type`,
  `TypeMember`, `Comparison`, `Opcode`, `Version`) are 0.11.0's. A DXIL resource whose lower bound or tag is -1 (an
  unbounded range) is read, so 10 slim programs go further and are refused later by name. A `_feedback` sample is read,
  and refused by its opcode's name where the reader refused its operands. The DXBC lifter ignores the debug layer's
  messages (`printf`) and refuses by name a declaration a later reader reads that it does not know.
- `iridia-d3d9` and `iridia-d3d9-ir` refuse an instruction of another version (`texkill` in a vertex program, `defb` in
  `ps_2_0`), and `iridia-d3d9-ir` co-issue and a result shift outside pixel 1.x, which the reader refused before
  0.9.0 and now reads, as Microsoft's disassembler lists them.
- Tests: the corpus tests read their programs through primase-testdata 0.2.6's `oracle_tree` (`Tree::names`,
  `Tree::read`), so a corpus may be a folder or an archive: iridia-dxil's `lift_corpus`, `metadata_census` and
  `container_digest`, iridia-dxbc's census helpers (`tests/common`: `lift_corpus`, `container_digest`,
  `control_bits`, `operand_format`, `exec_warp`, `vkd3d_oracle`, `reflection`, `disassembly`, `ir_gate`), iridia's
  slim gate and the d3d9 crates' sdb2 `corpus`, `gate` and `disassembly` tests. A folder is read by one thread at a
  time (SMB shares refuse many parallel reads, Windows error 1450) while the programs are checked in parallel; the
  programs come in the order the folder walk gave, so every count is the same. The slim set, the Elden Ring and skate.
  corpora and Shaun White's sdb2 are archives in 0.2.6, read on every thread, with the same counts.
- `iridia-ir`: amplification programs. `Stage::Amplification` with `StageFacts::Amplification` (`AmplificationFacts`:
  the task payload's bytes; `MeshFacts::payload` the mesh program's, at most `MAX_TASK_PAYLOAD`, 16384);
  `Node::DispatchMesh` (the mesh thread groups, a `vec3<u32>`, and the variable whose words are the payload: a barrier
  every invocation reaches once, the first invocation's values counting), in the text form as
  `dispatch mesh %g, name`; `Space::Payload`, a mesh program's task payload, read alone (`var name: [u32; n] payload`).
  `Stage::has_thread_groups` names the compute, mesh and amplification stages. `iridia-exec` runs one
  (`amplification::AmplificationShader`: `launches` gives each thread group's `Launch`, its mesh thread groups and
  payload words; `draw` runs each launch's mesh thread groups too), and `MeshShader::draw_launch` runs a launch;
  `Executable::dispatch_amplification_elements` runs one on given memory. Breaking: `dispatch_mesh_elements` takes the
  payload the mesh program reads (`&[]` for none), and `Space` and `Node` have new variants (a match of every
  variant needs the new arms).
  `iridia-naga` refuses both by name.
- `iridia-dxil::ir` lifts amplification programs and a mesh program's payload: `DispatchMesh`, `GetMeshPayload`, and
  the payload's memory laid out in bytes as the module's datalayout lays out its structure (each field at its natural
  alignment, a bool four bytes), so the two programs agree on its bytes whatever structures they declare; a halfword
  store into a groupshared payload keeps the other half of its word. The coverage runners draw an amplification
  pipeline (offload's mesh pipelines, vkd3d's `dispatch-mesh`) with the IR's lift of its amplification and mesh
  programs.
- `iridia-passes::promote` renumbers every variable a node or operation names among those it keeps: a mesh launch's
  payload was left at its old number, so a private variable promoted ahead of it left the launch naming another.
  `Node::vars_mut`, `Op::var` and `Op::var_mut` name them.
- `iridia-dxbc` and `iridia-dxil` refuse by name a gather of a depth texture's channel other than red
  (`GatherCmpGreen` of a shadow texture: "a gather of lane 1 of a depth texture", the words the IR's lowering refuses
  it in): they made a module naga refuses, since WGSL gathers a depth texture's one lane.
- `iridia-ir`'s `f32tof16` gives the quiet NaN 0x7e00 for every NaN (`iridia-exec-ir`'s definition, the WGSL
  `iridia-naga` writes for it, and `pack2x16float` in `iridia-exec`): the sign and payload of an operation's NaN are
  open in IEEE 754, and two lifts or a pass that orders an operation's operands otherwise made different ones, which
  the conversion carried into the half. On the Elden Ring corpus 96 programs differed between the lifts this way; none do now.
- `iridia-ir`: a mesh program's meshlet, `Node::SetMeshCounts`, `Node::StoreVertex` (lanes of a vertex's output, a
  vertex past the most the facts state not written) and `Node::StoreIndices` (a primitive's vertex indices), in the text
  form as `mesh counts %v, %p`, `store vertex %i 0.xy = %x` and `store indices %i = %x`, valid in a mesh program alone.
  `iridia-exec` runs a mesh program on the IR (`MeshShader::from_ir`) with its meshlet in the words the naga-path
  executor writes, so both draw the same meshlets; `Executable::dispatch_mesh_elements` runs one on given memory and
  hands each group's meshlet over. `MeshShader::interface` replaces `MeshShader::program`. `iridia-naga` refuses a mesh
  program by name.
- `iridia-dxil::ir` lifts mesh programs (`SetMeshOutputCounts`, `EmitIndices`, `StoreVertexOutput`), with the thread
  group size the entry states; a task payload, per-primitive outputs and `DispatchMesh` are refused by name. The gate
  runs a mesh program as a compute one and compares each group's meshlet too: the DXIL fixtures' 3 mesh builds agree,
  and on the slim set A runs 1,619 (3 mesh), B and P agree on 1,353, C on 1,123, none differ.
- `iridia-naga`: `Profile` states which `enable` directives `wgsl` and `wgsl_for` write (`Profile::directives`,
  `Directives::{Naga, Standard}`). `BROWSER` writes the standard's: `enable subgroups;` before a program that uses a
  subgroup operation or built-in, which a browser requires and naga's writer never states. `NAGA` writes naga's, without
  it: naga 30's WGSL reader refuses the directive.
- `iridia-naga`: `BindingPlan::webgpu` binds inside WebGPU's default limits (four bind groups, bindings below 1000):
  a register of space 0 at its number, the registers of other spaces after the largest binding of their group, and the
  hidden counters of structured views in group 3 after its views (`BindingPlan::d3d` put them in group 4, and other
  spaces at `(space << 16) | register`, both past the limits); a binding it would place at 1000 or past is refused by
  name. `Profile` states how `wgsl` binds (`Profile::bindings`, `Bindings::{D3d, WebGpu}`): `BROWSER` with the WebGPU
  plan, `NAGA` with the D3D one; `BindingPlan::for_profile` gives a profile's plan.
- `iridia-ir`: wave and quad operations, `Op::Wave` (`Wave`: a ballot, a read of the first active invocation's
  value, of another invocation's or of another of the quad's, and a reduction or scan of a value over the active
  invocations, `WaveOp` and `WaveScan`), in the text form as `wave.ballot`, `wave.first`, `wave.lane`,
  `wave.quadlane`, `wave.quadswap.x` and `wave.add.exclusive`. `iridia-exec` runs them as it runs naga's subgroup
  operations; `iridia-naga` lowers them to naga's (the `NAGA` profile allows `SUBGROUP`, and the preflight names the
  operation or lane input that needs it) and refuses a scan other than a sum or a product, and a quad read of a
  computed lane, which naga has no form of. The lane number and lane count inputs lower to naga's subgroup built-ins.
- `iridia-dxil::ir` lifts the wave and quad operations the older lift does (`WaveIsFirstLane` to `QuadOp`,
  `WaveAllBitCount`, `WavePrefixBitCount`), as it reads them: a read of a lane at a number is a shuffle, a prefix an
  exclusive scan, a flag read across the wave travels as 1 or 0, and a 64-bit value keeps its width; a 64-bit atomic
  operation and a derivative outside a pixel program are refused by name. The lane number of a thread group laid out in
  more than one dimension is the invocation's rank among its wave's when the program starts, as the older lift makes it.
  On the slim set's 4,059 DXIL programs, the older lift runs 1,616 and the IR agrees on 1,350, 1,123 again through
  `iridia-naga`'s WGSL; none differ. The IR refuses 51 programs the older lift takes (mesh and ray tracing stages, ray
  queries, compute derivatives) and lifts 1,006 it refuses.
- `iridia-dxil::ir`: a lift of DXIL onto the IR, its body a control flow graph (block parameters for LLVM's phis,
  blocks in reverse post-order) that `iridia_passes::structurize` makes structured. It lifts vertex (one that writes no
  position too), pixel, compute, hull, domain and geometry programs (a hull program's patch-constant function is the
  program's `patch`; a point's input, the output control points, the patch constants, the domain location, the
  primitive, control point and instance ids, and a geometry program's vertices and strip cuts on stream 0 as the older
  lift reads them), with the older lift's resource semantics carried over: constant buffer
  rows (a minimum-precision row of 16-bit lanes reads each lane's low half), samples, gathers, texel loads and stores,
  typed, raw and structured buffers, sizes and sample counts, sample positions, atomics, hidden counters (one per
  element of an array of views), the interface edits the older lift makes (a texture sampled with a comparison is a
  depth texture, a typed buffer four lanes an element, a view's access what the program does with it), 32-bit
  denormals flushed unless the entry's `fp32-denorm-mode` is `preserve`, and `MakeDouble`, `SplitDouble` and
  `Coverage`. A 16-bit lane of a raw or structured buffer is half of a word: read with a shift and a mask, written with
  an atomic and and or, so another invocation's half of the word is never overwritten; `tests/ir_warp.rs` holds such a
  program (`fixtures/buffer_lanes16`) to the memory Direct3D 12 WARP leaves, word for word. Mesh, amplification and
  ray tracing programs and ray queries are refused by name for now. `iridia-dxbc/tests/ir_gate.rs` takes DXIL
  containers (the gate's numbers are in the wave operations' entry above). The slim pre-push gate has the `IR` and
  `IR WGSL` columns for DXIL.
- `iridia-dxil`: a structured buffer whose element holds native 16-bit members (`-enable-16bit-types`) is no longer
  refused for a layout the interface reads as 32-bit lanes: its layout, which the interface's types cannot state, is
  left out.
- `iridia-naga` refuses a 1D array texture or storage image by name (WGSL has none; it wrote `texture_1d_array`, which
  naga does not read back), folds a conversion of an integer literal (naga's front end does not evaluate one of a 64-bit
  literal), and returns at the end of a function whose last block cannot be reached (naga wants the return).
- `iridia-naga` lowers compute programs and the memory of every stage: the thread group size and the invocation ids
  as built-ins; workgroup variables; barriers (WGSL's are execution barriers of the workgroup, so a memory barrier
  without execution and a device-wide one are lowered as the workgroup's); raw, structured and typed buffers, read and
  written, bounds-checked as the IR states (outside, a read gives zero and a write does nothing); every atomic of the
  IR on buffers, typed buffers and workgroup memory (a compare-exchange is a loop of naga's weak one, and a minimum or
  maximum of the other signedness than the memory's words a compare-exchange loop); the hidden counters of
  structured views; storage images (`rgba32*`, or `r32*` where an atomic reaches them) loaded and stored; texel loads
  of textures, checked against the size of their level; size, level, layer and sample queries; gathers; depth
  textures and comparison samples; fixed sample offsets; and a vertex program's clip distances as one
  `clip_distances` array (`Profile::NAGA` now allows `CLIP_DISTANCES`). A sample's array layer is rounded and taken at
  least 0, as Direct3D does (it was passed as a coordinate lane, which naga rejected). A front-facing flag read as an
  integer is all ones. A bitcast between 32-bit lanes and f64 goes through 64-bit integers (naga's bitcast keeps the
  lane count), an infinite or NaN literal is the bits of one, bitcast (naga holds none), and a lane or swizzle of a
  scalar is the scalar. What it still refuses, by name: hull, domain and geometry programs, a vertex program that feeds
  a tessellator, an element of an array of resources, an atomic of a storage image, a multisampled array texture, a
  level clamp, a gather at a computed offset, a sample of a 1D texture at an explicit level, bias or derivatives, and a
  biased sample of a depth texture or one at a computed level (WGSL has none of these). A module naga rejects is
  reported as a lowering bug naming the expression it rejects. Breaking: `BindingPlan` gains `counters` (group 4, at the
  view's binding), and `BindingPlan::d3d` binds a register of another space than 0 at `(space << 16) | register`
  (`uN_spaceS`) where it refused it.
- `iridia-exec`: a lane or swizzle of a value loaded from memory is the value as it was loaded. It was read from memory
  again where it was taken, so a store between the load and the read reached it: a naga module whose loads and their
  uses are apart (WGSL read back, `let s = l; l = ...; s.z`) ran wrong. The lifts' own modules, whose reads follow
  their loads, were not affected.
- `iridia-dxbc/tests/ir_gate.rs` has a column C: the IR after the passes, lowered by `iridia-naga`, written as WGSL,
  read back by naga and run in the executor A runs in, held to A on every vector. Sekiro: 432 of 432 agree; Riders
  Republic: 2,414 of 2,443 (25 hull, domain and geometry programs, 1 vertex program that feeds a tessellator and 3
  image atomics not lowered); Unity: 4,847 of 4,854 (7 multisampled array textures not lowered). The gate's texture
  backend answers a texel outside the size it reports with zero and drops a store there, as Direct3D's views do.
  `iridia-naga/tests/compute.rs` holds generated compute programs (buffer and workgroup stores, loads and atomics,
  inside and past their ends, with barriers) to leave the memory the IR leaves (proptest, 1,000 cases). The slim set's
  DXBC programs whose IR lowers and whose WGSL naga reads back: 3,026 of 3,670, from 951.
- The pre-push gate: `iridia/tests/slim.rs` takes the slim shader set (`oracles.corpus.general/slim`, primase-testdata
  0.2.2: 8,287 DXBC, DXIL and Direct3D 9 programs that reach every lifter region the full pool of 108,394 reaches)
  through every lift of each family: the older lift (validated, WGSL written and read back by naga), the IR lift
  (after the cut-over's passes, validated, its text printed and parsed back) and the IR's WGSL (lowered to stock naga,
  read back). What each program gives is held to `iridia/tests/slim/<family>.tsv`; a panic or a program that goes
  further or less far than the table fails. `CONTRIBUTING.md` names the one command to run before pushing. First
  run: DXBC 3,670 programs, older lift 3,292 ok, IR 3,424, IR WGSL 951; DXIL 4,059, older lift 2,239; Direct3D 9
  557, older lift 273, IR 428, IR WGSL 423; 17 s on four threads.
- `iridia-dxbc/tests/ir_gate.rs` checks its programs on every thread the machine runs, with the report a serial run
  gives, and runs A once for both B and P: the Unity corpus took 57 minutes on one thread and takes about four on 32
  (Riders Republic 418 s to 40 s). `IR_GATE_TIMES=1` prints where the time goes.
- `iridia-dxbc`: reflection reads the minimum-precision constant types fxc writes (`min16float`, `min16int`,
  `min16uint` and the narrower ones, types 52 to 57) as their 32-bit types, as DXBC stores them; Shader Model 4
  programs that declare them lifted no more once reflection read every Shader Model 4 program. Two constant buffers of
  one name (fxc keeps a `cbuffer C` declared twice in a namespace as two) are no longer refused: fxc lists the bindings
  by register and the buffers in the order declared, so each binding takes the layout that holds the rows its
  declaration states, and a program whose declarations leave that open (and whose layouts differ) is refused naming
  them. Fixtures: `tests/fixtures/reflection_sm4.hlsl` in three variants.
- `iridia-coverage` (`tools/coverage`, not published): the coverage table, per construct of the suites' programs, of
  what dx decodes and lists as Microsoft's tools do, what iridia lifts and writes as WGSL naga reads back, and what runs
  in `iridia-exec` to the result the suite expects, with the runs that prove it. Its runners run vkd3d's HLSL shader
  tests (d3d9, d3d11, d3d12 and d3d12-dxc, shader models 2.0 to 6.6) and LLVM's offload-test-suite (lit's DirectX
  configuration with DXC, its FileCheck and imgdiff checks and the golden images) on a device that holds Direct3D's
  resources and pipeline state and draws and dispatches with `iridia-exec`; each run that does not pass says why, its
  status that of the probes vkd3d's runner runs at that model (the extended ones, at models and in sections vkd3d's own
  runs leave out, counted apart); a Direct3D 9 run, whose programs leave denormals open, that does not hold as run is
  run again with them flushed, and passes when that holds and breaks nothing that held. A Direct3D 9 draw samples a
  pixel at its whole coordinates, as Direct3D 9 does (the viewport half a pixel on, as wined3d's), and runs again from
  the lifted IR in `iridia-exec` (`ProgramShader::from_ir`, in the register view of the WGSL), which fails the run where
  it writes otherwise than the WGSL's draw. A sample reduces its coordinate to its address mode's range before scaling
  it (D3D11.3 7.18.6; a NaN left is 0), a draw steps through a structured buffer by the stride the program declares
  whatever the view's, a vector probe steps 16 bytes as vkd3d's readback does, and a dispatch's typed buffer store
  reaches every lane of its element. A Direct3D 9 program that reads its texture stage's state (a `ps_1_x` `tex`:
  the projection flag, the bound texture's dimension), which WGSL cannot read, is specialised by each draw's state
  (`iridia_passes::specialise`) and made again for it; the table shows it at Direct3D 9's defaults. The device
  holds multisampled resources (a texel's samples together), loads a multisampled texture's sample, and a probe
  reads a multisampled target resolved, as vkd3d's runners do. A mesh dispatch runs the mesh program's thread groups
  (`iridia_exec::mesh`) and draws the primitives they emit with the pixel program; an amplification program and a
  pixel program reading primitive outputs are refused by name. The offload-test-suite's mesh pipelines run as the
  offloader's D3D12 mesh pipeline state has them (its default rasterizer: back faces culled, clockwise front faces),
  and its raster pipelines draw on multisampled targets, resolved before they are read back. Every atomic operation
  of an unordered-access view reaches the device (and, or, xor, minimum and maximum, exchange and compare-exchange,
  of a view in an array too), where it took adds alone. The vkd3d case files (format 3) keep the 377 probes vkd3d's
  tests mark `winbug`, with the qualifier: one naming devices (`amd`, `nvidia`, `warp=1.0.16`) is a driver's bug and a
  probe like any other (mesh.shader_test's per-primitive outputs and payload layouts among them), one that does not is
  counted as `winbug` and decides no run. CI makes the table again (`docs/coverage`) and fails when it
  differs from the one committed.
- `iridia-exec`: `draw` draws points and lines as well as triangles (every list and strip topology), each primitive
  numbered for `SV_PrimitiveID`; an aliased line by the diamond exit rule (3.4.3), which an NVIDIA RTX 5070 Ti draws
  exactly on every pixel of 10,640 recorded lines and WARP differs from in four classes the tests name and count;
  and `SV_ClipDistance` and `SV_CullDistance`, a primitive culled when every vertex is below 0 on one cull
  distance, clipped to each clip distance's plane, a point drawn when each of its clip distances is at least 0.
  `ProgramShader::system_input` gives the register the primitive's number, the coverage and the sample's index are
  read from when the signature gives them none, and a vertex program's clip and cull distances and target and
  viewport indices are in its register view.
- `iridia-exec`: `draw` takes Direct3D 9's fixed-function state: the alpha test (`Rasterizer::alpha_test`), user
  clip planes (`clip_planes`), the point size from the state or the vertex program's `PSIZE` (`point_size`), and
  point sprites (`point_sprite`), and the flat shade mode (`flat_colours`: the colour inputs are the provoking
  vertex's). A primitive's leading vertex picks its viewport from `Rasterizer::viewports`
  (which replaces `viewport`; an index of 16 or more is 0) and its target slice (`Surface::layers`; one past them is
  0); a pixel program's `SV_RenderTargetArrayIndex` and `SV_ViewportArrayIndex` are the values written, its
  `SV_SampleIndex` 0 and its `SV_Coverage` the pixel's one bit ANDed with `Rasterizer::sample_mask`, which also
  leaves out the write (`DrawStats::masked`). At shader model 2 a vertex program's colours are clamped to [0, 1]
  between the stages (`Rasterizer::clamp_colours`), and `DrawShader::with_linkage` takes Direct3D 9's linkage
  (`Linkage::Direct3D9`), where a pixel input no output writes reads 0 as a texture coordinate and, before shader
  model 3, white as a colour; `DrawShader::new` keeps the strict linkage, which refuses it. A pixel input lane reads
  the output lane of its component, where an input whose mask starts past x read the output's lanes from x.
  `Rasterizer::fog` takes Direct3D 9's fog as vkd3d-shader applies it (`Fog`): a coordinate from the vertex
  program's `FOG` output (or its specular alpha) or the position's z or w, interpolated perspective-correct, gives a
  factor (`FogFactor`: the coordinate, linear, exp, exp2), and a pixel that passes the alpha test writes its first
  target as `(colour - fog colour) * saturate(factor) + fog colour`.
  `DepthState::bounds` is Direct3D 12's depth bounds test, on the depth buffer's value, ends included; a pixel
  program that forces early tests (`[earlydepthstencil]`, naga's `early_depth_test`) runs only where the depth test
  and bounds pass, and writes its depth before it runs, where the tests ran after every program.
  `draw` draws on multisampled targets (`Surface::samples`: 2, 4, 8 or 16 at Direct3D's standard positions, a
  pixel's samples together): a triangle covers each sample it holds, the depth test and write are each sample's, a
  centroid input is read at the first covered sample where the triangle does not cover them all, `SV_Coverage` is
  the covered samples the sample mask keeps, and a program that reads `SV_SampleIndex` or a `sample` input runs once
  a sample. A multisampled line or point is refused.
- `iridia-dxil`: a boolean converts to a float as 1 or 0 (`uitofp`) and -1 or 0 (`sitofp`), where it converted its
  all-ones word; and extended to a 16-bit integer (`zext`, `sext`), it is cut to the 16 bits. An array of read-write
  views keeps a counter per element, the element's word at its index (`counter_of`, documented in `iridia-ir`). A
  program compiled with `-denorm preserve` (the entry's `fp32-denorm-mode`) keeps 32-bit denormals
  (`D3dModes::flush_denormals` false), where every program flushed them. `dx.op.coverage` (91) reads the pixel's
  `SV_Coverage`, where it was refused. `Texture2DMS::GetDimensions` gives the sample count, where it gave 1; a
  constant row of minimum-precision lanes (`min16float`, `min16int`) reads each lane as the low 16 bits of its word,
  where it reinterpreted the word; a 16-bit lane of a raw or structured buffer, which shares its word with another, is
  refused by name, where it was read and written as a word of its own. `LIFTER_VERSION` is 10.
- `iridia-dxbc`: `ld` with an immediate texel offset (`ld_aoffimmi`) reads the texel the offset moves to, where it
  dropped the offset; a hull program without a control-point phase passes its input control points through to its
  outputs, where they were left unwritten. `LIFTER_VERSION` is 12.
- Tests: property tests (proptest, a dev-dependency of `iridia-exec` and `iridia-coverage`) on generated inputs
  against reference formulas: the draw's fog factors and blend, the depth test within its bounds, the standard sample
  positions and the input lane mapping, and the coverage sampler's address-mode range reduction.
- Tests: the oracle tools (vkd3d's programs, WARP, dxc) run through primase-testdata 0.1.14's `program(Tool::X)` with a
  time limit (`output_within`): a tool that hangs is killed and the test fails naming it, instead of hanging. A minute
  per vkd3d call, two per dxc compile, ten per WARP batch. vkd3d's tools are its Linux programs, so the vkd3d
  comparison runs on Linux and skips on Windows (no wrapper that starts them through WSL).
- `iridia-dxbc`: `iridia_dxbc::ir::lift` (and `lift_with`) lifts a Shader Model 4 or 5 vertex, pixel or compute program
  onto the IR (`iridia-ir`), with the interface `lift` reads: registers as variables of raw lanes, Direct3D's own rules
  as the IR's operations, flow as its constructs. Memory is the IR's: raw, structured and typed loads and stores of
  buffers and views, texel loads and stores of texture views, every atomic and its immediate form (compare-exchange
  takes the comparand first), `imm_atomic_alloc` and `imm_atomic_consume` as counters, thread-group memory as a workgroup
  variable, `sync` as a barrier; Shader Model 5.1 resource ranges as an element of an array resource (`nonuniform`
  kept). Hull, domain and geometry programs lift too (below).
- `tests/ir_gate.rs`'s planted-bug tests are gone. Property tests (proptest) hold the gate's comparison to generated
  runs: a written lane that changes, a memory word, a write more or a texture request A does not make is flagged, in
  one invocation and in any lane of a quad, and in a compute program's memory; what no target takes (a lane not
  written, a discarded invocation, one NaN for another) is not. Another holds each coherent quad's steps to differ in
  size along x and y, neither zero, from any first-lane value the gate seeds, in coarse and fine derivatives.
- `iridia-dxbc/tests/ir_gate.rs`, the gate of the cut-over: every program of the Sekiro, Riders Republic and Unity
  corpora and every fixture lifts both ways and runs in `iridia-exec`, the older lift (A), the IR lift (B) and the IR
  lift after `promote`, `simplify` and `break_flags` (P), on seeded inputs, constants and textures; a pixel program with
  a derivative or an implicit-level sample runs as a 2 x 2 quad, a compute program as two thread groups whose read-write
  buffers, texels and counters are compared after. Every written lane must agree bit for bit (a NaN with any NaN); any
  other refusal of the IR lift fails it. No difference anywhere: Sekiro, 432 programs, B and P agree on all 432; Riders
  Republic, 2,443 programs (736 compute, 26 made of functions), B and P agree on all 2,443; Unity, 4,854 programs, B and
  P agree on all 4,854; every fixture A runs. A vector A cannot evaluate is skipped and counted (Riders: 238 of its
  vectors, a loop past the executor's step budget on seeded constants in 235, a quad diverging at an implicit-level
  sample in 3).
- `iridia-dxbc`: the IR lift takes hull, domain and geometry programs (and a vertex program feeding a tessellator) as
  the functions `iridia::stage_functions` states: a hull program's control-point phase as `main` and its fork and join
  phases, each a loop over its instances, as `patch_constant`; a geometry program's `emit` and `cut` as the IR's `Emit`
  and `Cut` (a stream other than 0 is refused by name). `iridia-exec`: `Executable::function_from_ir` runs one of an IR
  program's functions; `GeometryShader::from_ir` and `TessellationShader::from_ir` run the stages from the IR.
- `iridia::stage_functions`: a hull program's `main` takes `SV_PrimitiveID` after `SV_OutputControlPointID`, since its
  control-point phase may read `vPrim`; `tests/exec_warp_tess.rs` writes it into each control point, against WARP.
  `iridia::PROGRAM_VERSION` is 4 and the lifters' version tables are rewritten.
- `tests/ir_gate.rs` runs each function of a program made of functions, A, B and P on the same seeded arguments: the
  points of each array, the varyings, and pipeline values within the program's facts (an output point it has, a domain
  location inside the patch, an instance it runs). `tests/exec_warp_tess.rs` and `tests/geometry.rs` run the IR beside
  the older lift.
- A `switch` case that falls into the next one after code of its own is refused by the IR lift, in every model, as a
  program Direct3D 11 does not allow (the Direct3D 11.3 specification, 22.7.18; Direct3D 10 allowed it, HLSL never
  wrote it): WARP runs such a case as if it ended with `break`, in a `ps_4_0` program and a `ps_5_0` one alike.
  `switches_agree_with_warp_and_a_falling_case_is_refused` (in `tests/exec_warp.rs`) runs two switches against WARP
  through both lifts and makes their cases fall by changing `break`s of `fxc`'s output to `nop`.
- `Texture1DArray` (resource dimension 7) is taken by both DXBC lifts; `resinfo_agrees_with_warp` asks its size.
- Reflection of every stage and of Shader Model 4: `resource_reflection` reads the RDEF of a hull, domain or geometry
  program and of a Shader Model 4.0 or 4.1 program, and returns the resources alone (`Option<Vec<Resource>>`, was an
  `Interface`); `reflection` takes Shader Model 4 pixel, vertex and compute programs too. The lifts name those
  programs' resources and lay out their constant buffers from it. `every_stage_and_model_reflects_as_fxc_lists_it`
  (in `tests/reflection.rs`) holds Sekiro's 87 hull, domain and geometry programs and the Unity corpus's 16 `ps_4_0`
  programs to Microsoft's listing of each (`fxc /dumpbin`).
- Shader Model 5 doubles on the IR: the IR lift takes every double-precision instruction (`dadd` to `utod`, the
  comparisons, `dmovc`), a register's lane pair as a double, the low word first; the IR runners run 64-bit values
  (`iridia-exec-ir`'s `wide`). `doubles_agree_with_warp` holds a program of all of them to WARP, through the IR lift.
  The older lift refuses a program with the doubles flag by name; its `LIFTER_VERSION` is 11.
- The older DXBC lift refuses, by name, a signed and an unsigned minimum or maximum on the same memory (naga's atomics
  take the memory's one type, so one of them was lowered as the other); the IR lift takes both.
  `a_buffer_takes_signed_and_unsigned_minimums` (in `tests/exec_warp.rs`) holds the IR lift to the four results
  `InterlockedMin` and `InterlockedMax` give, each of which a signed and unsigned mix-up changes.
- The gate's vectors: five (`CASES`), the unstated constants as floats (0, 1), small integers (2, 3) and zero (4, where
  memory, texel and structured loads are 0 too, so a branch on a cleared value is taken). A quad whose lanes diverge at
  a derivative runs again with coherent inputs (each interpolated input the first lane's plus steps of its own along x
  and y), then with the first lane's in every lane. Of the programs that ran coherent quads, those whose coherent
  vectors take no derivative (`Execution::derivatives`) and those that discard every lane taking one are counted: what
  those vectors leave untested (Unity: 5 of its 46 take none and 26 discard every lane that takes one; Riders Republic:
  84 of 91 take none). A vector A cannot evaluate (a loop past the step budget on seeded constants, a request its
  sampler does not answer) is skipped and counted by reason, in every stage; a program is A's failure only where A
  evaluates none of its vectors. The programs neither lift reads and those only the IR lifts are counted by reason.
- The gate compares a pixel or vertex program's memory: every store and atomic (by view, element or texel, word), each
  word's values in the order it is given them; and each texture request B makes (sample, texel load, query) must be one
  A makes, so a texel the program asks for differently is caught even where a select then drops it.
- `shader.outside`, a debug key (`GAME_DATA_DEBUG`): a folder of DXBC containers outside the configured corpora, which
  the IR gate (`outside_programs`) and the WARP comparison (`outside_programs_agree_with_warp`) run.
- `iridia-exec`: a discarded pixel changes no memory after its `discard` (the Direct3D 11.3 specification, 16.13): its
  stores, texel stores and atomics are dropped, an atomic gives 0. Every buffer atomic reaches the sampler
  (`Sampler::atomic`, `Atomic`), not only adds. A typed buffer's store and atomic address its element (`u[1]` is element
  1, the lanes in it), as `Store` states. A vertex program's results are its entry's result items (`iridia::entry_io`),
  each clip or cull distance its own member at its lane. A front-facing flag an IR program reads as an integer is
  Direct3D's all ones for a front face (it was 1). `Saturate` gives +0 below zero, -0 among it, and a `mov` with a
  source modifier flushes a denormal (both as WARP does); the older executor's negation and absolute value flush too.
  `Execution::derivatives` counts the screen derivatives an evaluation took part in.
- `tests/exec_warp.rs`: a discarded pixel's memory, outputs after several clip distances, a typed buffer's element, a
  view at its register and not its id (Shader Model 5.0 and 5.1), a front-facing flag, denormals through moves; each
  against WARP or the specification.
- `iridia-ir`: the vocabulary the DXBC and DXIL lifts need: buffers (`Res`, `Address`), texel loads, gathers and
  resource queries, stores to buffers and images, atomics on buffers, images (a typed storage buffer's element too) and
  workgroup variables (`Memory`), counters, barriers, switches, arrays and workgroup variables, patch constants,
  `Emit` and `Cut`, ternary and bit-field operations; `Program::workgroup`. `FRem` is C's `fmod`, exact. A cube's size
  query has two lanes, its faces' width and height. A barrier outside a compute or mesh program may order storage
  only; a barrier of nothing is refused.
- `iridia-exec`: `Executable::from_ir` and `ProgramShader::from_ir` run an IR program in the flat executor. Buffer
  atomics (`Backend::atomic_buffer`, `BufferAtomic`) and counters (`Backend::counter`) reach the backend where the run
  does not hold the memory; image atomics gain compare-exchange (`ImageAtomic::compare`). A discarded or helper
  invocation's atomics and an atomic outside the memory change nothing and give 0. The naga path gains WGSL integer
  division (by zero, the lowest `i32` by -1), `firstLeadingBit`, `extractBits` and `insertBits`.
- `resinfo`, in both DXBC lifters, as the Direct3D 11.3 specification states it (22.4.14): `_rcpFloat` takes the
  reciprocals of the size's lanes only, not of the array size, the level count or a lane a smaller texture fills with
  zero; at a level the texture does not have, every lane but the level count is zero (the array size was given); the
  older lift's cube size is two lanes. The IR lift takes `resinfo` of a texture view (one level). No compiler writes
  `_rcpFloat` and none of the corpora asks past a texture's levels. `tests/exec_warp.rs` checks all three return types
  against WARP (`resinfo_agrees_with_warp`, the `_rcpFloat` program made by changing the float one's instructions) and
  runs the IR lift beside the older one on every program it renders.
- `iridia-d3d9-ir`'s gate runs a fourth column, D, the IR in `iridia-exec`, and takes a NaN as any NaN (IEEE 754 leaves
  a result's NaN sign and payload open; debug and release builds differ in them).
- `iridia-naga`: `tests/ops.rs` checks each operation's WGSL against its IR definition; `FRem`, `f32tof16` and
  `f16tof32` are lowered in integers (exact, without the f16 capability); a naga validation error keeps its causes.
- `iridia-passes`' tests run the flat form and the WGSL in the `break_flags` form; a remainder by zero is 0.
- `iridia::flow`: `Reach` gains `words`, the constant-buffer words (`GlobalWord`: stage, resource, word) that may
  reach an output lane or the discard, each with its `Via`, as inputs and texels have it. A constant that only decides
  a branch or a discard is `Control`, one that scales a texture coordinate `Coordinate`; `WordsRead` says which words
  are read and nothing of how. `link` carries the vertex program's words through, as its texels.
- `iridia-naga`: `wgsl_for(program, profile, &Pipeline { plan, feeds })` writes a program for one pipeline: each
  resource where the pipeline's `BindingPlan` binds it, and a vertex program's outputs at the locations, interpolation
  and types of the pixel program it feeds (matched by semantic, as Direct3D links stages; padded, narrowed or bitcast
  where the two differ, each noted). An input no output carries is refused by name.
- `iridia-naga`: the largest finite `f32`, either sign, is written as its bits bitcast (`bitcast<f32>(2139095039u)`):
  its shortest decimal, 3.4028235e38, lies a little above it, and Tint (Chrome's WGSL compiler) refused the module as
  out of range. Direct3D 9's `rsq` and `nrm` of zero give it, so a vertex program that normalises a vector failed to
  compile in Chrome.
- Oracle scripts run on Linux under Wine (primase-oracles `tools/wine`), each checked there byte for byte against what it
  made on Windows: `iridia-d3d9/tests/oracles/disassemble.py` takes the path of `d3dcompiler_47.dll` and refuses a bare
  name, which under Wine loaded Wine's own D3DCompiler and wrote a listing that is not Microsoft's (it now writes all 3
  `general/sm3` and all 7,306 `shaun-white/sdb2` listings of the oracle home as stored); `iridia-dxbc/tests/fixtures/sm51/generate.py`
  builds the SM5.1 fixtures from their headers' commands and writes the `.51.asm` dumps (all 31 files as committed, with
  fxc 10.0.26100.4654 and the Windows SDK's dxc.exe; the Linux dxc 1.7.2308 release writes other containers);
  `assemble.py` (all 105 assembled fixtures) and `vkd3d_cases.py compile` (`vkd3d/cases.json` as committed) say which
  DLL and how to run them under Wine.
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
