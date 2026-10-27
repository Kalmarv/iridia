# Changelog

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
