# Changelog

## Unreleased

- `iridia-dxbc` (IR lift, `LIFTER_VERSION` 5, and the older lift, `LIFTER_VERSION` 13): a float compared with itself is
  lifted as the test for NaN on its bits. `ne x, x` is what fxc writes for `isnan(x)` and for `x != x`, and both lifts
  wrote it as `x != x`: a compiler that assumes no NaN (DXC without `-Gis`, and WGSL allows it of any implementation)
  writes `false` for it, so a program's guard against a NaN was gone from what it ran. A lane of `ne`, `eq` or `ge`
  whose two sources are one value (the same component of the same register at the same indices, the same modifier and
  minimum precision; the components a swizzle names for lanes the destination does not write do not count) is `IsNan`
  for `ne` and its negation for `eq` and `ge`; fxc packs such a lane beside lanes that compare two values, so it is
  decided lane by lane, and the other lanes are compared as before. `dne`, `deq` and `dge` are the same for a double
  (IR lift). `lt x, x` never holds and stays a comparison, as does every comparison of two values, which still relies
  on the compiler keeping Direct3D's rule for a NaN. Fixture: `self_compare` (our own HLSL); no other fixture's lift
  changes.
- `iridia-exec` (breaking): the executor names the compute invocation of every write and atomic it hands a backend.
  `ComputeInvocation` (a thread group and the linear index in it, scoped to one dispatch) is new; `Backend::store_buffer`
  and `Backend::atomic_add` take it as their first argument, and `Store`, `AtomicAdd`, `Atomic`, `ImageStore`, `ImageAtomic`
  and `BufferAtomic` carry it as `invocation` (`None` outside a compute dispatch). The pipeline, pixel and slot adapters pass
  it through. A `Sampler` implementation only receives the new field.
- `iridia-dxil` (IR lift, `LIFTER_VERSION` 8): `dx.op.isSpecialFloat.f16` `IsNormal` (dxc writes it for `isnormal` of a half
  from shader model 6.9; before, it writes the exponent's bit tests, which were already right) took the smallest normal
  float, 2^-126, as the bound, so a subnormal half (a normal float once widened) was normal; the bound is the half's, 2^-14.
  The float-to-16-bit-integer conversion (`fptosi float to i16`) is unchanged: it saturates at the 32-bit range and keeps
  the low 16 bits, which is right in range (tested at -3.7, 100.9, -32768 and 32767); LLVM leaves an out-of-range
  `fptosi` poison and DXIL.rst states nothing for it, so no value outside the 16-bit range has a defined result to hold it to.
- `iridia-dxil` (IR lift, `LIFTER_VERSION` 7): a 16-bit integer is carried in the low bits of a 32-bit word, and the lift
  had ordered and counted it as that word: a signed `icmp` on an `i16` compared the zero extension (so `sign(int16_t)` of a
  negative gave 0), `dx.op` `IMax` and `IMin` did the same (`abs` of a negative `int16_t` stayed negative, `max`, `min` and `clamp`
  took the wrong bound), `Bfrev` reversed 32 bits of it, and `FirstbitHi` and `FirstbitSHi` counted the 16 bits
  of the extension (dxc subtracts that count from 15, so `firstbithigh` gave 4294967283 where it should give 3). A signed
  compare and `IMax`/`IMin` sign-extend an `i16`, `Bfrev` shifts the reversed word down by 16 and `FirstbitHi`/`FirstbitSHi`
  take the 16 off; no fixture's lift changes, the version is bumped for the programs that did.
- `iridia-dxil` (the older lift, `iridia_dxil::lift`): the older lift had the same bug as the IR lift's `LIFTER_VERSION` 6: a bitcast between a half and a
  16-bit integer took the widened float's 32 bits as the integer (the exponent and mantissa tests dxc makes of a half in
  `isnan`, `isinf`, `isfinite` and `isnormal`, `asint16`, `asuint16`, a 16-bit store of a half), so those programs ran and gave wrong
  values. A half's bits are made with `pack2x16float` and a 16-bit integer's half with `unpack2x16float`; the gate's column A (the
  older lift) and the IR lift now agree on the 32 slim-set programs they differed on (4059 programs, 0 differ).
- `iridia-dxil` (IR lift, `LIFTER_VERSION` 6): a bitcast between a half and a 16-bit integer (`asint16`, `asuint16`, and the
  exponent and mantissa tests dxc makes of a half in `isnan`, `isinf` and `isfinite`) gives the half's bits. A half is carried
  widened to 32 bits, and the lift had taken the widened float's bits as the integer, so those programs ran and gave wrong
  values; no fixture's lift changes, the version is bumped for the programs that did. A 16-bit integer bitcast to a half
  widens it.
- `iridia-analysis`: `blend`, the recogniser of `iridia::blend` on the IR: `Scalar`, `Term`, `Over` and `Layer` read from `flow::Values`, with `Over::at`
  and `Over::under`. A term is spelled for the IEEE operations only (`FAdd`, `FSub`, `FMul(MulZero::Ieee)`, `FNeg`, `Saturate`, `FMin`,
  `FMax` by `MinNum`); Direct3D 9's zero-wins multiply is a leaf, as it is not the operation a term evaluates. The older tests run with the
  same expectations (`tests/blend.rs`); `tests/blend_oracle.rs` holds every pixel program of the fixtures, the slim set and Elden Ring's
  corpus to the older recogniser's answer, by text and by value under the same named leaves (`tests/expected/blend-*.tsv`, judged in
  `tests/expected/README.md`).
- `iridia` tests: the slim gate has no column for the older lifts (`iridia_dxbc::lift`, `iridia_dxil::lift`, `iridia_d3d9::lift`);
  each program of the set goes through the IR lift and the IR's WGSL, and `tests/slim/<family>.tsv` lists those two columns.
- `iridia-d3d9-ir` tests: the gate no longer runs the older lift (`iridia-d3d9`) and the WGSL read back into the naga
  interpreter (`iridia-exec`) beside the IR; it holds the IR's run to the IR compiled into `iridia-exec` and to the WGSL
  of both writers on WARP, and its assembled and vkd3d tests run the IR, its graph and the graph structured again. The
  older-lift half of the projection gate is gone with it; the crate's dev-dependencies on `iridia`, `iridia-d3d9` and the
  naga fork are gone.
- `iridia-ir`: a ray tracing program's facts state the scalars of the structures it declares its payload and hit attributes
  as (`RayTracingFacts::payload_leaves` and `attribute_leaves`, each a `Leaf` of a byte offset and a `LeafKind`: bool, 16, 32
  or 64-bit integer, 16, 32 or 64-bit float; integers carry no sign), and `Interface::verify` holds them in ascending order
  within the bytes. `iridia-dxil` reads them from the entry function's parameter types laid out by the module's datalayout and
  refuses a program whose structure is not the size the entry point states (`LIFTER_VERSION` 5: the eleven ray tracing fixtures of a
  hit, miss or callable program change; a program with no leaves serializes as before). The datalayout code the byte-laid
  payloads use moved to its own module.
- `wgsl-run` (the WARP harness): workgroup memory is zeroed in the naga module, by loops strided over `local_invocation_index`
  and a workgroup barrier, with wgpu's own zero-initialisation turned off. wgpu 30.0.1 zeroes it with one assignment of the
  whole array by the first invocation, which WARP takes 50 to 120 s to compile for arrays of about 3,000 words (1,024 words:
  3.4 s, 4,096: 115 s); gfx-rs/wgpu PR 10314 fixes that on trunk and is not in a release. A job's `workgroup_init` field
  (`"wgpu"`) selects the old behaviour; the output of a program is the same either way (a test compares them).
- `iridia-dxbc` tests: the compute comparison of the WGSL writers against the IR executor runs programs with typed views,
  storage images, samplers and arrays of them; a word's float or integer origin (what decides whether it is compared within
  tolerance or exactly) now follows variables, array elements and control-flow joins and takes a typed view or image's declared
  lane type; a word two or more of the executor's invocations left different final values is compared by whether WARP's
  value in every repeat is one of those finals, with no table row (a word one invocation wrote, however many times, is
  compared as any word is). `WARP_UNSTABLE` lists the rest by name and source hash with a cause the
  test re-checks on every run. Test-side only; no library source changes.
- `iridia-d3d9-ir`: `log` and `logp` of a zero source give -FLT_MAX and `texdepth` with `r5.g` zero gives 1.0, as Microsoft's
  instruction reference states (`LIFTER_VERSION` 5, twelve fixtures change); the crate documentation tables what the lift
  does at every operation's documented special inputs and why. `iridia-d3d9`, the older lift, gives `log` of zero the same
  value. The WGSL of both writers is run on WARP in the crate's gate, vkd3d and assembled tests.
- `iridia-d3d9-ir`: `frame` and `sampler_states`, the two functions of `iridia-d3d9` that read bytes and states rather than
  lift them, with the same behaviour and tests (`frame` returns `iridia_ir::interface::Stage`; errors are this crate's `Error`).
  `iridia-d3d9` keeps its own until it is removed.
- `iridia-dxbc`, `iridia-dxil`: the readers the IR lift and the naga-path lift share (`setup`, and for DXIL the small helpers
  and the `dx.op` argument counts) are modules of their own, not parts of the naga-path lift. Nothing they produce changes.
- Route nonlocal structured exits through typed result payloads only across the boundaries they cross. Keep ordinary
  SSA values and loop parameters in place, replacing the whole-function demote/route/promote sequence that multiplied
  transfer copies in nested control flow. Mutually exclusive payloads share a typed field pool selected by the route
  tag. The IR form and version are unchanged.

- `iridia-naga`: check unsupported operations, resource and variable shapes, and structured exits before allocating
  the lowering arena. Named limitations return without expanding the preceding operations; successful output is
  unchanged. Unused value-table entries are not treated as placed operations.
- `iridia-passes`: remove unused construct results, loop parameters and their incoming arguments by a dependency
  worklist, including dead transfer cycles. Exit routing demotes only live slots and prunes again after promotion,
  avoiding large networks of temporary copies in both WGSL writers.
- `iridia-wgsl` writes ray queries, for wgpu's `wgpu_ray_query` WGSL extension (`Profile::NAGA`; `Profile::BROWSER` has
  none and refuses them by name): `enable wgpu_ray_query;`, an `acceleration_structure` resource, a function-level
  `var q: ray_query` per query, `rayQueryInitialize` with the query's declared flags joined to the ray's, `rayQueryProceed`,
  the terminate, confirm and generate-intersection calls, and every query field as the committed or candidate
  `RayIntersection`'s (a candidate's kind is 1 for a triangle and 3 for a procedural primitive in naga, 0 and 1 in the IR;
  a ray in object space is the hit's world-to-object matrix applied to the ray the program started the query with, kept
  in `let`s only when read). Refused by name: whether a candidate procedural primitive is opaque (the intersection does
  not say), an array of acceleration structures, and a query that is started again. naga 30 cannot restart a query (a
  function-level one keeps its finished bit across `rayQueryInitialize`, so the second search never proceeds; a
  loop-local one panics its HLSL writer), so a program that starts one query at more than one site, or at one site under a
  loop that goes round again, is refused with the probe evidence in `rayquery::restarted`'s doc comment. `iridia-naga`'s
  refusal now says iridia-wgsl writes them. `exec_warp_rq.rs` also compares the grid, procedural and object-space
  programs through three roads on WARP (the fixture's DXIL, the naga module through stock naga's HLSL writer, the WGSL text
  read by stock naga and written as HLSL), by the same two-of-three consensus: no ray differs.
- `iridia-naga`: the software remainder helper's quiet NaN is a runtime bitcast of its exact word, like an IR
  float literal, rather than a WGSL constant expression that Tint cannot represent.
- `iridia-naga`: construct each emitted expression range directly from its arena suffix, so lowering many
  subgroup operations takes linear work instead of repeatedly scanning the growing expression arena.
- `iridia-naga` standard WGSL filters derivative-uniformity diagnostics, and subgroup-uniformity diagnostics
  when the program uses subgroups and the consumer reports that WGSL language feature. This preserves the IR's
  divergent operations when creating a browser shader module.
- Breaking: `iridia_naga::Profile` now carries a separate `WgslLanguageFeatures` set; built-in profiles default
  to no optional language features. Consumers can set it with `with_language_features` or `with_subgroup_uniformity`.
- The single-container IR gate diagnostic is `cargo run --release -p iridia-dxbc --example ir_gate`, with
  `IR_GATE_ONE` naming its input. It uses the gate's existing comparisons and is no longer an ignored test.
- `iridia-naga`: a filtered sample of an integer texture (a sample, a biased or a level one, or one at gradients, of a texture
  of unsigned or signed integers) is refused by name, "a filtered sample of an integer texture has no WGSL form", as
  `iridia-wgsl` already did; naga's writer made text its own validator refused ("Unable to operate on image class
  Sampled { kind: Uint }"). A load or a gather of an integer texture is written as before.
- `iridia-d3d9-ir`: `project`, `reencode` and `project_vertex_output` (each with a `_with` taking the lift's `Options`) and
  `Projection`, the derived programs `iridia-d3d9` makes, on the IR: a ps_3_0 program cut after an instruction (the cut's
  lanes as RGB, alpha one), the whole program with the cut's lanes replaced by a new input `DERIVED_RGB0`, and the
  backward slice of a vs_3_0 output run as a pixel invocation. A vertex program with flow control is refused by name
  (the older slice dropped the flow control and kept the writes inside it).
- `iridia-passes`: `flatten_sampler_heaps`, each array of samplers (a Shader Model 6 sampler heap, indexed at run time)
  made one sampler, for a pipeline that binds one: the program's notes say every index reads the same sampler. It is
  the older writer's `single_samplers` option as a pass over the IR.
- `iridia-passes`: `clip_distances_as_varyings` and `discard_clipped`, the clip distances of a vertex program carried as
  plain varyings for a pipeline whose clipper cannot take them (WebGPU without `clip-distances`), and the pixel
  program's discard on them. Each `SV_ClipDistance` output becomes an ordinary varying of the same label and width, code
  untouched (the program no longer needs the `ClipDistances` capability, so the browser profile writes it); the pixel
  program gets a perspective-interpolated input for each and, before each return, a demote where any lane is below
  zero. The two programs meet by label, so `Pipeline::feeds` links them. It is the older writer's `clip_varyings` and
  `clip_discard` pipeline options as passes over the IR.
- `iridia-ir`: `BindingPlan::d3d` and `BindingPlan::webgpu` place a resource that has no Direct3D register (a descriptor
  heap's, the sampler a flattened sampler heap leaves) in space 0 after the highest register of its class, in the order of
  resource ids, where they refused it; a hidden counter is still refused. Of the slim set's Shader Model 6 programs, 38
  that were refused for this reason are not: 35 are written as WGSL (the IR WGSL count of dxil 2604 -> 2639 of 4059), two
  are refused as ray generation programs and one as a lowering bug naga's validator names.
- `iridia-ir`: `BoundSizes`, the size each constant buffer is bound with (taken by `lift_with`), is `iridia_ir::interface::BoundSizes`,
  beside `ConstantBufferBinding`; `iridia_interface` and `iridia` re-export it, so no path breaks. The IR lifts of
  `iridia-dxbc` and `iridia-dxil` take it from there.
- `iridia-ir`: `StageFacts::EarlyDepthStencil` carries a pixel program's forced early depth/stencil tests. DXBC and
  DXIL IR lifts preserve the source flag; the draw executor suppresses external side effects on depth-rejected
  pixels while helper invocations still evaluate for derivatives. WGSL writers retain a note that WGSL has no early
  depth/stencil guarantee and continue writing the shader. The IR version is 3 (was 2); DXBC and DXIL IR lifter
  versions are 4, invalidating cached lifts made without the stage fact.
- `iridia-dxil::ir` distinguishes typed scalar constants from raw constant-array data words, and refuses scalar
  constants of the wrong kind and volatile or atomic loads and stores. The damaged-input regression tests now use
  the IR lift and its diagnostics.
- `iridia-ir`: ray queries (DXR 1.1's inline ray tracing, Direct3D's `RayQuery<flags>`). A function declares its
  queries (`Function::queries`, a `QueryDecl` of the template's flags each, `QueryId` their index: an object of the
  function, not a variable, which no pass promotes); `Node::RayQueryInit` (the acceleration structure and a `RayDesc`
  of `TraceRayInline`'s operands), `Node::RayQueryAbort`, `Node::RayQueryCommitTriangle`,
  `Node::RayQueryCommitProcedural`, `Op::RayQueryProceed` and `Op::RayQueryState` (a `QueryField`: the committed
  status, the candidate's type and opacity, the ray's flags, origin, direction and least distance, and a `HitField` of
  the candidate or the committed hit). A search's flags are the ray's and the declared ones together. The text form is
  `query q0 flags 0x10` declarations, `rayquery init q0 r0 flags %f ...`, `rayquery abort q0`, `rayquery commit q0`,
  `rayquery commit q0 t %t`, `rayquery.proceed q0` and `rayquery.state q0 committed.status`; the validator holds each
  operation to a query the function declares, a field to the lanes, rows and columns it has and the type it gives, and
  a search to an acceleration structure. Breaking: `Function` has the field `queries`, and `Node`, `Op` and the
  re-exported names (`QueryDecl`, `RayDesc`, `QueryId`, `QueryField`, `HitField`) are new or have new variants.
  `iridia_ir::target::Capability::RayQuery` is the shared requirement for queries and acceleration structures;
  `iridia-wgsl` refuses them by name, including when the profile allows that capability.
- `iridia-dxil::ir` lifts ray queries (dx.op 178 to 215): every `ray_query_*` fixture lifts on the IR, and
  `ray_query_object.hlsl` (new: object-space rays, matrices, candidate opacity, the flags a search runs with), which
  the naga lift refuses, too. `iridia-exec` runs them on the IR, and `exec_warp_rq.rs` compares its every committed hit
  and candidate with WARP's on the IR lift (768 rays of each of three programs, none differ). `iridia-exec-ir` refuses
  them by name. Of the slim set's 28 DXIL programs that stopped at `allocateRayQuery`, 24 lift on the IR (the 22 the
  naga lift alone took among them); the other 4 the naga lift refuses too. `iridia-naga` lowers them into the naga
  module, which naga's validator accepts with the ray query capability (`Profile::NAGA` allows it;
  `Profile::BROWSER` refuses the program before lowering): a ray in object space is the hit's world-to-object
  transform of the ray, and whether a candidate procedural primitive is opaque, which naga's intersection does not
  say, is refused by name. `iridia-naga`'s WGSL output refuses them by name (naga's writer has none); `iridia-wgsl` writes them.
- `iridia-dxil::ir` refuses an operation that yields a value called as a function that returns none (a damaged
  callee's type) by name, "an instruction that yields a value but defines none", as the naga lift does: it lifted the
  program and dropped the value.
- `iridia_dxil::ir::LIFTER_VERSION` is 3 (was 2), invalidating cached lifts for the new ray-query representation.
  The fixture table includes seven newly lifted ray-query fixtures and the new object-space query fixture; existing
  successful fixture hashes are unchanged.
- Material feature selection follows uniform lanes through structured block results when every exit carries the
  same source lane. Other lanes remain dynamic, and enclosing-label breaks are distinguished from loop exits.

- Material feature folding uses the IR reference evaluator, including integer bit operations, conversions and float
  arithmetic. Folded decisions now agree with reference execution by construction, including its portable math and
  denormal rules. Unsupported reference operations remain typed refusals.
- Reference float rounding preserves an input NaN's sign and payload and sets its quiet bit for all five modes.
  Nearest-away rounding deliberately now quiets signaling NaNs, consistent with the other modes; its old result
  inconsistently left them signaling. Finite rounding results are unchanged.

- The reference evaluator's pure operations live in `iridia_ir::eval`, shared by execution and material folding.
  `iridia-exec-ir` re-exports its existing evaluation API. Reference transcendentals now come from `libm` on every
  target, instead of the platform math library; there is one backend for folding and execution.
- `iridia-exec`: raster helper invocations evaluate for derivatives and private variables but perform no external
  writes, image or buffer atomics, or append/consume counter operations. A fullscreen triangle writes each covered
  pixel once; helper lanes previously wrote UAVs too.
- `iridia-naga`: exact half conversions, floating remainder, high-word multiplication and bit-field operations use
  cached typed helper functions instead of repeating their software expansions at every use. Tests execute the
  emitted calls on WARP and retain the native differential oracle through a temporary test-only inliner.
- `iridia-naga`: long single-use expression chains are bound to intermediate WGSL `let` values, keeping parser
  nesting bounded. The reduced deep-chain fixture validates in both profiles; generated chains agree with IR execution.
- `iridia-naga`: IR literals and zero values have runtime `let` bindings in written WGSL. Specialising a constant
  into `log2`, inverse square root or division no longer asks WGSL constant evaluation to represent a non-finite
  result. Non-finite float literals likewise reconstruct their bits from a runtime integer binding. Reduced IR
  fixtures validate in both profiles and run against the IR interpreter.
- `iridia-ir::target` owns `FnItem`, `FunctionSig`, `ItemList` and `stage_functions`, the function signatures of
  stages without graphics entry points. `iridia` re-exports the same definitions for existing callers.
- `iridia-dxbc`, `iridia-dxil`: the IR lifts have version gates, as the older lifts and `iridia-d3d9-ir` do: what
  `ir::lift` makes of each committed fixture is hashed with `iridia_ir::gate` under the IR lifter's `LIFTER_VERSION` in
  `tests/ir-lifted.hashes`, so a change to what the IR lift makes of a bytecode fails its tests until the version is
  bumped. The IR lifts had changed since their versions were set (ray tracing, the switch arguments above, and more),
  so the versions go up: `iridia_dxil::ir::LIFTER_VERSION` 1 -> 2, `iridia_dxbc::ir::LIFTER_VERSION` 2 -> 3. A cache
  keyed by those versions drops what the earlier lifts made.
- `iridia-wgsl` (new, `no_std` with `alloc`, no unsafe code, builds for `wasm32-unknown-unknown`): a WGSL writer from the IR with no
  dependency but `iridia-ir`, as `iridia-naga`'s `wgsl` is without naga. `wgsl(program, &Profile)` writes a vertex, pixel or
  compute program (the same refusals by name as `iridia-naga`, and a program a profile's capabilities cannot carry refused before
  writing), `wgsl_for` the same for one pipeline (a vertex program's outputs where the pixel program it feeds reads them).
  `Profile::BROWSER` writes standard WebGPU WGSL (a `diagnostic(off, derivative_uniformity);` where the program samples or
  differentiates, since the source program takes those derivatives where it takes them); `Profile::NAGA` binds Direct3D's registers
  and writes naga's dialect. Held to upstream naga's reader and validator, to Tint (`tools.tint`) and, run on WARP, to the IR
  executor, by `iridia-dxbc`'s `exec_wgsl` (debug keys `shader.writer`, `shader.tint`, `shader.tint_keep`, `shader.only`,
  `shader.worst`, `shader.dump`). A vector that a loop or an `if` carries is kept a lane at a time, and a lane the construct leaves
  the value it was before is that value, not a variable: Tint's uniformity analysis takes a variable assigned under a divergent
  condition, or a vector with one thread-dependent lane, to differ between threads, and a barrier after it is refused. The
  barrier programs of the Elden Ring and Riders Republic corpora are accepted by Tint, apart from four Elden Ring light
  organisers whose counter starts at the thread's index and subtracts it again: uniform by arithmetic, which WGSL's analysis
  cannot follow. As `iridia-naga`'s `wgsl` does, `Profile::NAGA` writes an array of resources as one binding array (`enable wgpu_binding_array;`,
  a structure with one member `data` for each array of buffers, an element's counter word n of the resource's packed counter buffer)
  and a texel atomic whose old value nothing reads as `textureAtomic*` of an `atomic` storage image; Direct3D's standard sample positions
  are a private table. `Profile::BROWSER` refuses the first two by name, as WebGPU's core has neither.
- Removed the D3D9 IR gate's remaining planted-change test and its design-document claims. The gate compares the
  unmodified programs across executors; input fuzzing and property tests remain.
- `iridia-dxil`: a switch several of whose cases (or a case and the default, or both ways of a branch) go to one
  block gives that block one set of arguments. dxc writes the block's phi with an entry per edge, the same value on
  each (`phi [7.0, %0]` three times); the IR lift made each edge's arguments afresh, so a constant became three
  values and `iridia_passes::structurize` refused the program ("block N's switch gives block M two sets of
  arguments").
- The dx readers are 0.12.0 (`d3d9-reader`, `dxbc-reader`, `dxil-reader`, `dxcontainer`; before, 0.11.0). Nothing a
  lift reads changes; the `d3d9-reader` types `iridia-d3d9` and `iridia-d3d9-ir` re-export are 0.12.0's.
- `iridia-exec`: `VertexProbe::from_ir(iridia_ir::Program)`, the probe of a vertex program on the IR, with the checks
  `VertexProbe::new` makes (a vertex program that writes a clip position), and `VertexProbe::ir` and
  `VertexProbe::interface`. Breaking: `VertexProbe::program` gives an `Option<&Program>` (`None` for a probe of an IR
  program).
- `iridia-exec`: a scalar evaluation of an IR program takes the derivative of an input it reads through bitcasts and
  composes (as a DXBC lift reads registers as bits) from the derivatives it is given, as it does a plain read; it was
  refused as "a derivative of a computed value".
- The executor's and the D3D9 lifter's tests run their programs on the IR: each one written as IR text or lifted by the
  IR lifts, run by the `::from_ir` constructors and, where it can, by `iridia-exec-ir` too, the two held to agree.
- `iridia-ir::target`: what a module written from a program carries of it, for every writer and executor alike, with
  no writer's types. `BindingPlan` (with `Bind` and `Bindings`; `d3d`, `webgpu` and `new(interface, Bindings)`) says
  where each resource binds; `entry_io` (with `EntryIo`, `IoItem`, `IoType`, `Binding`, an iridia-owned `BuiltIn`,
  `carried` and `varying_name`) how the entry point carries the interface, locations, interpolation and clip
  distances included; `requirements` (with `Capability` and `Requirement`) which capabilities of a target a program
  needs. `iridia-naga` and `iridia` each map them to their naga, so the two lowerings bind and carry a program by one
  statement of it. `iridia-naga` re-exports `BindingPlan`, `Bind` and `Bindings`; its `entry_io`, `EntryIo`, `Item`,
  `requirements` and `Requirement` are unchanged, and so are `iridia`'s `entry_io`, `EntryIo`, `IoItem` and `IoType`.
  Breaking: `iridia_naga::BindingPlan::for_profile(interface, &profile)` is now `BindingPlan::new(interface,
  profile.bindings)`, and `BindingPlan::d3d` and `webgpu` refuse with a `String` (it was `iridia_naga::Error`, whose
  text it was). Behaviour: a pixel program's lane index and lane count are carried as `u32` (the built-ins' type) by
  `iridia-naga` too, and `iridia`'s refusal of a used clip or cull distance outside a vertex program's outputs reads
  "a clip or cull distance as a Pixel input has no built-in", as `iridia-naga`'s did.
- `iridia-naga` lowers the sample position query (`Query::SamplePosition`, HLSL's `GetSamplePosition`): a private
  table of Direct3D's 31 standard positions (`sample_positions`, from `iridia_ir::standard_sample_position`), read at
  the texture's sample count's rows; (0, 0) for an index past the texture's samples or a count with no standard
  pattern. It is core WGSL. It was refused ("query SamplePosition(ValueId(N)) of a D2 image"); a query of something
  an image does not have is refused in words ("a query of the layers of a D2 texture, which has none"). Run lowered
  in `iridia-exec` for counts 1, 2, 4, 8 and 16, every index and past it, and counts 0, 3, 6 and 32, each position is
  the IR's.
- `iridia-naga` lowers an atomic of a storage image's texel whose old value nothing reads: naga's image atomic
  (`textureAtomicAdd` and the others) on an `r32uint` or `r32sint` image of WGSL's `atomic` access, where the texel
  is inside the image. A subtraction is the addition of the negation; a minimum or maximum is of the texels'
  signedness. It was refused ("an atomic of an image is not lowered yet"). What naga 30's image atomic cannot carry is
  refused by name: an atomic whose old value is read, a compare-exchange, an exchange (naga's WGSL reads
  `textureAtomic*` as statements, of no value and no exchange), and a minimum or maximum of the other signedness.
  `Profile::NAGA` has `TEXTURE_ATOMIC`; `Profile::BROWSER` refuses such a program by name (WGSL has no texel
  atomics): the requirement is `iridia_ir::target::Capability::TextureAtomic` (breaking: a new variant). The DXBC and DXIL `image_atomics` fixtures lowered and run on naga's executor agree with their naga lift.
  `iridia`'s program validator takes a storage image of WGSL's `atomic` access (atomic, load and store) for one an
  atomic reaches, whatever else the program does with it.
- `iridia-naga` lowers arrays of resources: an array of textures, samplers, storage images, constant buffers or
  buffers (a `Resource` whose `count` is not 1, sized or unbounded) is a naga binding array (`binding_array<T, N>`, or
  `binding_array<T>` unbounded; a buffer's element is a structure of its one member), and an element of one (`Res`
  with an `index`) an index of it, unclamped as in Direct3D. It was refused ("an element of an array of resources is
  not lowered yet", "a sample through an element of an array of samplers ..."). `Profile::NAGA` has naga's binding
  array capabilities, so its WGSL is naga's dialect (`enable wgpu_binding_array;`); `Profile::BROWSER` refuses such a
  program by name ("resource t is an array of textures": WGSL has no arrays of resources). The requirements name each
  array and each one indexed per invocation: `iridia_ir::target::Capability::BindingArray` and
  `BindingArrayNonUniform`, of an `ArrayClass` (textures and samplers, constant buffers, buffers, storage images),
  which `iridia-naga` maps to naga's `*_BINDING_ARRAY` and `*_NON_UNIFORM_INDEXING` capabilities. Breaking:
  `Capability` has the two new variants. The DXBC 5.1 and DXIL fixtures'
  arrays (sized, unbounded, indexed per invocation, in spaces) lowered and run on naga's executor give, lane for lane,
  what the IR's executor gives.

- `iridia-mojo::ir::lift` lifts MojoShader's GLSL onto an `iridia_ir::Program` (`LIFTER`, `LIFTER_VERSION`), with the
  older lift's interface, resources and notes, and reads more of the dialect: `samplerCube` and `sampler3D`
  (`textureCube`, `texture3D`), `break`, the fog varying (`varying float vFogFragCoord`), `bvec4` registers and booleans
  converted to numbers (`float(a >= b)`, `vec4(greaterThanEqual(a, b))`). Where GLSL leaves a result undefined the
  program does as Direct3D 9 does and states it (`glsl.float_to_int=saturate`, `glsl.uniform_oob=zero`). Its oracle is
  the Direct3D 9 program the GLSL was made from: `tests/oracles/mojoshader.py` runs MonoGame's own `libmojoshader_64.dll`
  (dotnet-mgfxc 3.8.1.303) as MonoGame's effect compiler does, and `tests/ir.rs` (fixtures compiled by fxc) and
  `tests/oracle.rs` (real programs) run both lifted, `iridia-d3d9-ir`'s and this one's, in both executors. MojoShader
  prints a `def` constant to six decimals, which the oracle allows for by a second Direct3D 9 program holding the
  printed numbers, and a pair whose GLSL is not its program (`NRM` of four lanes, `TEXLDL` without its level, a `CMP`
  reading a lane it just wrote, MojoShader's one-float fog) is counted by that departure. Breaking: the varyings are
  named as MojoShader names them: `vFrontSecondaryColor` is colour 1 in both lifts, and `vFogFragCoord` the fog on the IR (a `float`, which the older lift does not read); `vBackColor` and
  `vFogFactor`, which MojoShader never writes, are refused.

- `iridia-passes`: `prepare(&mut Program)`, the passes every output and executor takes a lifted program through, on
  each of its functions: `structurize`, `promote`, `simplify` and `break_flags`. What it gives is what `iridia-exec`'s
  `::from_ir` constructors and `iridia-naga` take, so a caller no longer copies the sequence (the slim gate and the IR
  gate's column P call it).
- `iridia-dxbc::ir` lifts a hull program whose patch-constant function leaves patch constants unwritten: the registers
  the patch-constant signature places them in need no declaration in any fork or join phase, and an unwritten one gives
  zero, as the naga lift does (it refused "output register oN is not declared": the patch function's registers were
  the control points'). `hull_unwritten.hlsl` (fxc) tessellates on the IR as the HLSL says, as the naga lift does, and
  as WARP does.
- Breaking: `iridia_passes::specialise` is now material feature specialisation, taking a program and known constant-buffer words and returning `Result<Specialised, SpecialiseError>`. `Specialised` is `#[must_use]`. The previous pipeline-state API is `specialise_state`.

- New crate `iridia-analysis` (`no_std` with `alloc`, no `unsafe`): the `iridia` crate's analyses on the IR.
  - `output_uses`, `discard_compares` and `alpha_tests` read output/discard/blend evidence from IR graphs, including the alpha-expression helpers.
  - `matrix::lines` and `matrix::lines_at` recover matrix row/column use, including indexed palettes; graph refusals use `GraphError`.
  - `flow::Values` is the value graph, lane by lane.
  - `flow::Flow` is what may reach each output lane and the discard, and via what (`Via`).
  - `flow::VertexDemand` is a vertex and pixel pair's vertex input lanes.
  - `reads::WordsRead` is the constant words a program's results are made of, and which buffers it reads at a computed
    index.
  - Each takes an `iridia_ir::Program` as a lifter gives it; a control flow graph is structured on a copy.
  - tests/oracle.rs holds them to the `iridia` crate's analyses on the fixtures, the slim set and Elden Ring's corpus,
    and records the IR's answers in tests/expected.
  - Property tests run generated programs in `iridia-exec-ir`: changing a source that `Flow` leaves out of a lane never
    changes that lane, and never changes whether the program discards.
  - Code after a way out of a construct (a return, a discard, an exit or continue of an enclosing one) runs only where
    that way was not taken, so `Flow` puts those conditions on what follows.
  - Nested loops take one walk of each body, with header values installed before the walk and unchanged headers
    removed afterwards. The oracle bounds the legacy analysis in a subprocess and records an out-of-memory refusal.
- `iridia-exec-ir::value` ports `iridia_exec::value` onto `iridia_analysis::flow::Values`, with the same shape:
  - `Evaluator::new(values, flush_denormals, leaf)` and `evaluate(id)`, which gives an `Evaluated::Word` or the
    `Evaluated::Unknown` leaf it needs;
  - each operation is evaluated by `iridia_exec_ir::eval`, the interpreter's own definition;
  - a derivative, a wave operation, an atomic's result, an undefined value, a loop's value, a merge whose arms
    differ, a run-time indexed element and a value that is no bool or 32-bit scalar are refused, never guessed.

  `iridia-exec-ir` now depends on `iridia-analysis`. `iridia-analysis`'s tests check each output lane the evaluator
  gives against a run of the program, on generated programs and on every program of the oracle sets.
- `iridia-ir`: the operation enums are `PartialOrd` and `Ord`, so a graph can key on them: `Unary`, `Binary`,
  `Ternary`, `Cmp`, `AtomicOp`, `WaveOp`, `WaveScan`, `QuadSwap`, `Axis`, `Fineness`, `MinMaxNan`, `SaturateNan`,
  `MulZero`, `OnZero`, `FloatToInt` and `Rounding`.
- `iridia-coverage` grades the IR path: DXBC, DXIL and Direct3D 9 programs go through each bytecode's IR lift,
  `iridia_passes::prepare` and the IR's validation, and run in `iridia-exec` on the IR (`from_ir`); the WGSL column is `iridia-naga`'s lowering for browsers,
  read back by naga, and decides no run. It graded the naga-path lift of DXBC and DXIL before, and Direct3D 9's WGSL.
  Each case file runs again through the lift the IR replaces, each row's `reference_status`, until that lift is
  removed. The table's schema is version 2: `graded` names what it grades; `statuses` and `groups` name each status
  and its group (gap, runner-limit, not-applicable, expected, unexpected-pass, pass); each row has its `group`; each
  suite its `reasons`, what is left by status and class of reason with an example, and its `reference_totals`.
  offload's `unsupported` is split: `not-applicable` where lit runs the test on no Direct3D 12 device in its DirectX
  configuration with DXC (a `REQUIRES: Vulkan` or `Clang`, an `UNSUPPORTED: DirectX`), `not-modelled` where a Direct3D
  12 device can run it and iridia's lacks the feature (acceleration structures, ray tracing pipelines, wave sizes but
  32...). vkd3d's `failed` (a program iridia took and a pass or the executor stopped on) is `error`, not `fail`. The
  table is made of parts, a case file through one path each, or one shard of it (`--shard <index>/<count>`):
  `iridia-coverage part` runs one, `table --parts` makes the table of them, and CI builds the tool once and runs the
  parts side by side. The `vkd3d` and `offload` commands take `--reference` and name
  their slowest runs.
- `iridia-exec-ir`: `Prepared`, an operation prepared once on its operand types (`Prepared::new`), and `Prepared::run`
  for one lane's operands or `Prepared::run_lanes` for every active lane of a register file in one loop, which
  `iridia-exec` runs its IR operations by; `eval` is `Prepared::new(..).run(..)`. Results are what they were,
  and a loop of float arithmetic takes about half the CPU time it did. IEEE 754 leaves the payload and sign of
  the NaN an operation on NaNs gives open, and which operand's a multiplication keeps depends on how the compiler
  orders its operands; the crate documents it.
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
- `iridia-dxil` names a program's target as dxc does by dxil-reader's `Part::target`: an amplification program's
  `provenance.target` is `as_6_5` (it was `cs_6_5`: every shader kind past those named fell to `cs`), and a shader kind
  no target names is refused by name.
- Both DXIL lifters refuse `RenderTargetGetSampleCount` by name, as the DXBC lifters refuse `sampleinfo` of the
  rasterizer: the render target's sample count is the pipeline's, not the program's ("the sample count of the
  rasterizer is the pipeline's, not the program's"). A program that reads it (`sample_count.hlsl`) stays refused by
  every lifter, onto naga and onto the IR: a refusal a caller's tests can rely on.
- `Stage::with_article` names a stage after the article it takes (`an Amplification`, `a Pixel`), and every message
  that names a stage uses it: "an Amplification program is not lowered yet", not "a Amplification".

## 0.6.0 (2026-11-07)

Breaking changes, for code that uses 0.5.0 (each is in its entry below):

- The dx readers are 0.11.0: the `d3d9-reader` types `iridia-d3d9` and `iridia-d3d9-ir` re-export (`Comment`,
  `Constant`, `ConstantTable`, `Type`, `TypeMember`, `Comparison`, `Opcode`, `Version`) are 0.11.0's, so code that
  builds them with `d3d9-reader` 0.5 needs 0.11.
- `iridia-exec`: `MeshShader::interface` and `GeometryShader::interface` replace `MeshShader::program` and
  `GeometryShader::program`; `DrawShader::draw` (any topology, from `Primitive::assemble`) replaces
  `DrawShader::draw_indexed` (a triangle list); `Rasterizer::viewports` replaces `Rasterizer::viewport`, and
  `Rasterizer` is no longer `Copy`; `DrawStats::primitives` replaces `DrawStats::triangles`; `DepthState` is no longer
  `Eq` (its `bounds` are floats); `ProgramShader::program` gives an `Option` (a program, where a naga-based lifter
  made it; `None` for one run from the IR); `TessellationShader::hull` and `TessellationShader::domain` give the
  stage's `Interface`, where they gave its program. `Rasterizer`, `DrawStats`, `DepthState`, `Surface`, `Output`,
  `Execution` and `ImageAtomic` have new public fields, and `ImageAtomicOp` the variant `CompareExchange`.
- `iridia-dxbc`: `reflection::resource_reflection` gives the reflected resources (`Vec<Resource>`), where it gave an
  `Interface`.
- `iridia-ir`: `Op::LoadBuffer` takes an `address` (`Address`) in place of `index`, `stride` and `offset`, and it and
  `Op::Sample` name their resources by `Res` (a resource and an array element), where they took a `ResourceId`;
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
