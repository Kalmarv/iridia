# iridia

iridia lifts compiled Direct3D shaders (DXBC, DXIL, Direct3D 9 tokens and MojoShader's GLSL) into [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) modules, so that a shader you only have as bytecode can be validated, inspected and written back out as WGSL, SPIR-V or whatever else naga writes. If you are looking for a DXBC to WGSL translator, an HLSL bytecode to SPIR-V converter or a shader decompiler in Rust, this is one.

It reads vertex, pixel, hull, domain, geometry, compute, mesh and ray tracing programs from Shader Model 2 to 6, and keeps what a naga module cannot say (semantic names, D3D register packing, constant buffer layouts, the clip-space and numeric conventions the program was written for) in a typed interface beside it. It was written against the shaders of shipped games (Sekiro, Elden Ring, Riders Republic, skate., Shaun White Skateboarding, a Unity game and an XNA/MonoGame game), whose programs are what its real-data tests lift. It also has a CPU reference executor that runs a lifted program on explicit inputs, to check a translation against Microsoft's WARP rasterizer or against values worked out by hand. Used by [primase](https://github.com/Kalmarv/primase), a game-asset conversion library.

## What is in it

| crate | what it does |
| --- | --- |
| [`iridia`](iridia) | The shader program every reader lifts into: a `naga::Module` with its interface, validation, WGSL writers, the typed builder the readers share, and analyses that read a program without running it. |
| [`iridia-dxbc`](iridia-dxbc) | Reads Direct3D Shader Model 4 and 5 bytecode (DXBC containers, `fxc` output) of any stage, with its reflection and signatures. |
| [`iridia-dxil`](iridia-dxil) | Reads Direct3D Shader Model 6 bytecode (DXIL containers, `dxc` output) of every stage but amplification, with its reflection and a ray tracing library's subobjects. |
| [`iridia-d3d9`](iridia-d3d9) | Reads raw Direct3D 9 `ps_3_0` and `vs_3_0` token streams and their CTAB constant table. |
| [`iridia-mojo`](iridia-mojo) | Reads the GLSL dialect that MojoShader writes when it translates Shader Model 2 and 3 programs for OpenGL. |
| [`iridia-interface`](iridia-interface) | The plain, serializable data that describes a program's interface: varyings, resources, constant layouts, binding roles and render state. No dependency on naga. |
| [`iridia-exec`](iridia-exec) | A bounded CPU executor for lifted programs, with drivers for the stages WGSL has no entry point for (geometry, tessellation, mesh, ray tracing) and a small software rasterizer. |

## Example

Lift a Shader Model 5 pixel shader, look at what it reads and write it as WGSL (`cargo run -p iridia --example lift -- shader.dxbc`; this is `iridia/examples/lift.rs`):

```rust,no_run
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("usage: lift <shader.dxbc>")?;
    let container = std::fs::read(path)?;

    // Shader Model 4 or 5; `iridia_dxil::lift` takes Shader Model 6 the same way
    let program = iridia_dxbc::lift(&container)?;
    println!(
        "{:?} program, target {}",
        program.stage, program.provenance.target
    );
    for input in program.interface.inputs.iter().filter(|input| input.used) {
        println!("reads {}{}", input.label.name, input.label.index);
    }

    // the same program as WGSL; `program.module` is a `naga::Module` for any other naga backend
    print!("{}", iridia::emit::wgsl(&program)?);
    Ok(())
}
```

Two more examples take a container and write the result: `to_wgsl` (DXBC or DXIL, WGSL on standard output, the lifter's notes on standard error) and `to_spirv` (DXBC or DXIL to a SPIR-V file through naga's SPIR-V writer). The crate READMEs have a smaller example each.

Add the crates you need from crates.io: `cargo add iridia iridia-dxbc` for the example above. Rust 1.99 or newer (edition 2024).

## What it guarantees

- **A reader returns an error for bytes it cannot read; it is written not to panic on them.** Every reader returns a `Result`. A refusal ([`iridia::Refusal`](iridia/src/refusal.rs)) says which bytecode, where (a 32-bit word of a DXBC or D3D9 token stream, a bit of a DXIL module, a line of MojoShader's GLSL) and what it expected or could not carry. Sizes a declaration asks for are checked against Direct3D's limits before anything is allocated. This is tested, not proven: the DXIL tests truncate a compiled shader at every length and run seeded mutations of it, and the DXBC tests damage declarations and operand counts one case at a time.
- **Nothing is approximated silently.** A construct the lifter does not carry is refused by name; a construct it carries approximately (min-precision operands computed at 32 bits, say) is a note on the program (`Program::notes`).
- **Direct3D's numeric results are written out.** Where Direct3D differs from naga or WGSL (integer division by zero, float to integer conversion of NaN and overflow, `min`/`max` and `saturate` with a NaN), the body spells D3D's answer out as plain operations and `Program::modes` says it did. The denormal flush is recorded as the source's expectation, because WGSL leaves it to the implementation.
- **A lifted program validates.** `Program::validate` runs naga's validator and then checks the interface against the entry point and the resources against the module's globals. A writer that cannot express a program refuses it; it does not approximate.
- **A cache can tell when it is stale.** Each lifter and the program representation have a version number, and each lifter's tests compare a table of hashes of what it makes of its fixtures against a committed one, so a change of output without a bump of the version fails (`iridia::gate`). A lifted program carries the SHA-256 of the bytes it came from.
- **The executor never invents a value.** A run follows the program's stated Direct3D behaviours, has a step budget, and answers an index past the end of a local array or a texture nobody supplied with an error that names it. Sampling is the caller's, because a scalar execution cannot infer derivatives, filtering or addressing.
- **The library crates do no I/O and contain no `unsafe`.** They read the bytes and text they are given; `clippy.toml` forbids file, environment and process access in them, and the workspace denies `unsafe_code`.

## What it does not do

- It does not write bytecode, HLSL or GLSL itself. Output is whatever naga's backends write; this repository's tests exercise the WGSL writer and the SPIR-V example, and a program naga's writer cannot express (a texture atomic that returns the old value; the ray tracing pipeline) is refused or kept as naga's IR dump in `Program::text`.
- It does not cover all of every Shader Model. Refused by name, among others: doubles (and, in DXIL, 64-bit integer division and conversions between 64-bit integers and floats), Shader Model 5.1 resource declarations, append and consume buffer counters, geometry output to a stream other than 0, amplification programs and per-primitive mesh outputs, and, in raw Direct3D 9 token programs, relative addressing, control flow and vertex texture fetch. Each lifter's README lists what is carried and what is not.
- It does not apply a pipeline's conventions. A program keeps Direct3D's clip depth and texture origin (`Program::conventions`); a pipeline that wants Vulkan's applies them where it meets the program.
- A hull, domain or geometry program, a vertex program that feeds a tessellator and every ray tracing program is a set of functions, not an entry point, because WGSL has no such stage. It lifts, validates and runs on the executor, and `iridia::emit::wgsl_functions` writes the functions as text, but there is no WGSL pipeline for it.
- The executor is a reference, not a renderer: it is slow and deterministic, and its sampling, addressing and filtering are what the caller supplies. Its agreement with Microsoft's WARP rasterizer is checked only by tests that need a WARP host program this repository does not contain.
- The container's digest is not verified: nothing here authenticates a shader.

## Why it depends on primase-naga

The lifters build `naga::Module`s that use three things naga 30.0.1 lacks, so the workspace depends on [primase-naga](https://github.com/Kalmarv/primase-naga), a fork of naga 30.0.1 with exactly those additions: the WGSL writer writes ray queries; `Statement::ImageAtomic` has an optional result (the texel's previous value); and `AtomicFunction::Exchange` is accepted on 32-bit images. The library is still named `naga` (the dependency is renamed with `package = "primase-naga"`), so the types are naga's. A program of your own that uses `Program::module` must depend on the same fork, or the two `naga::Module` types are different types; see the workspace `Cargo.toml` for the pinned revision. When upstream naga has these features the fork goes away. `iridia-exec` also uses [primase-foundations](https://github.com/Kalmarv/primase-foundations) for texture addressing and sampling and a small thread helper, and the tests use [primase-testdata](https://github.com/Kalmarv/primase-testdata) to find real-data inputs.

## Running the tests

```sh
cargo test --workspace
```

needs nothing but the repository: the fixtures are small compiled shaders and the HLSL they were compiled from (`fxc` for Shader Model 5, `dxc` for 6), kept in each crate's `tests/fixtures`. Tests that need files this repository cannot ship (a game's own shaders, Microsoft's compilers, reference output made from them) skip, printing one line that names the missing input.

To run those against your own game files, write one TOML file outside the repository and point the `GAME_DATA` environment variable at it. The layout belongs to [primase-testdata](https://github.com/Kalmarv/primase-testdata), which has a documented sample; the keys these tests read are:

```toml
[tools]
fxc = 'C:/path/to/fxc.exe'        # Microsoft's shader compiler, Shader Model 5
dxc = 'C:/path/to/dxc.exe'        # DirectX Shader Compiler, Shader Model 6
warp = 'C:/path/to/warp-host.exe' # a program that runs a shader on WARP; not part of this repository, so those tests will skip for you

[oracles]
# <root>/<set>/<game>/<folder>: oracles.corpus.sekiro/shaders is <root>/corpus/sekiro/shaders
root = 'D:/oracles'

[scratch]
root = 'D:/scratch'               # where tests write their outputs (the temporary folder when unset)

[run]
require = 'gate'                  # or 'all': an unset input fails instead of skipping
```

The real-data tests lift whole folders of shader containers you extract from your own installs (`corpus/<game>/shaders`: Sekiro, Elden Ring, Riders Republic, skate. and a Unity player's), validate and write each, and print the refusals counted by reason; a program that lifts but fails validation fails the test. Others compare the readers with Microsoft's disassembler (`fxc /dumpbin` text, made with `iridia-dxbc/tests/oracles/dxbc_corpus.py`, or `iridia-d3d9/tests/oracles/disassemble.py` for Direct3D 9 tokens). `GAME_DATA_DEBUG='key=value;key=value'` sets the debug switches a crate declares, such as `dxil.fuzz_runs`. Every input, what it is and where it comes from is in the generated tables of
[`iridia-dxbc/tests/inputs/README.md`](iridia-dxbc/tests/inputs/README.md), [`iridia-dxil/tests/inputs/README.md`](iridia-dxil/tests/inputs/README.md), [`iridia-d3d9/tests/inputs/README.md`](iridia-d3d9/tests/inputs/README.md) and [`iridia-mojo/tests/inputs/README.md`](iridia-mojo/tests/inputs/README.md). Game files stay on your machine; nothing in this repository downloads or extracts them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), which also says how to report a shader these crates cannot read.

## Licence

GPL-3.0-or-later, see [LICENSE](LICENSE). The fixed-function tessellator in `iridia-exec` is a port of Microsoft's reference tessellator, published under the MIT licence, whose notice is kept in `iridia-exec/src/tessellator.rs`. The primase-naga fork keeps naga's own licence (MIT OR Apache-2.0).
