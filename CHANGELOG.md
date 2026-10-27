# Changelog

## 0.1.0 (2026-10-27)

First release.

- `iridia`: the shader program (a naga module with its interface, conventions and provenance), its validator, the WGSL writers, the typed builder the readers share, and analyses that read a program without running it.
- `iridia-interface`: the plain, serializable description of a program's interface (varyings, resources, constant layouts, binding roles, render state). No dependency on naga.
- `iridia-dxbc`: reads Direct3D Shader Model 4 and 5 (DXBC containers) of every stage, with reflection and signatures.
- `iridia-dxil`: reads Shader Model 6 (DXIL containers) of every stage but amplification, with reflection and a ray tracing library's subobjects.
- `iridia-d3d9`: reads raw Direct3D 9 `ps_3_0` and `vs_3_0` token streams and their CTAB constant table.
- `iridia-mojo`: reads the GLSL that MojoShader writes for Shader Model 2 and 3 programs.
- `iridia-exec`: a bounded CPU reference executor with drivers for the geometry, tessellation, mesh and ray tracing stages, and a small software rasterizer.
- Every public item of every library crate is documented, and each library crate denies `missing_docs`.
