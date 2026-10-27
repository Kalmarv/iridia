# Real-data inputs

What this crate's real-data tests read, as tests/inputs/mod.rs declares them: the keys of the configuration file
`GAME_DATA` names (its layout is described by [primase-testdata](https://github.com/Kalmarv/primase-testdata)), the oracle sets and the files of each, and the debug keys
`GAME_DATA_DEBUG` sets. A test whose input is not configured skips with a line naming its key.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `tools.dxc` | yes | Microsoft's `dxc` shader compiler executable | third-party: Microsoft's dxc |
| `tools.warp` | yes | the WARP oracle program that runs a shader on Microsoft's WARP rasterizer | third-party: Microsoft's WARP rasterizer, through a thin host program that is not part of this repository |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

#### Oracle sets (`<oracles.root>/<set>/<game>/`)

| set | about | source |
| --- | --- | --- |
| `oracles.corpus` | the games' own files cut from their installs (shader containers, EBX documents, Havok tagfiles, DCX files, CAS frames) | the user's own installs: files extracted as stored |

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.corpus.elden/shaders` | live | - | a folder (searched recursively) of the DXIL containers of Elden Ring's Shader Model 6 shader bundles (one file per container), lifted, validated and written as WGSL, the refusals counted by reason with an example program; iridia-dxil/tests/lift_corpus.rs | the user's own install: the containers of its `/shader/*.shaderbnd.dcx` binders and `/shader/*.shaderbdlebnd.dcx` bundles (ray-tracing `[rt]` bundles included); primase's [corpus_elden_shaders example](https://github.com/Kalmarv/primase/tree/main/worlds/fromsoft/examples/corpus_elden_shaders.rs) extracts them |
| `oracles.corpus.riders/shaders` | live | - | a folder (searched recursively) of the DXBC containers of Riders Republic's shader pack (ShaderDataDX11.bin, split at its containers), lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs | the user's own install: the programs split from its shader pack |
| `oracles.corpus.sekiro/shaders` | live | - | a folder (searched recursively) of the DXBC containers of Sekiro's extracted pixel and vertex programs (`.ppo`, `.vpo`), lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs; the same programs are used in primase's [detail_derive test](https://github.com/Kalmarv/primase/tree/main/worlds/fromsoft/tests/detail_derive.rs) | the user's own install: the programs extracted from its shader archives |
| `oracles.corpus.skate/shaders` | live | - | a folder (searched recursively) of the DXIL containers of skate.'s CompiledBytecode resources (one file per container), lifted, validated and written as WGSL, the refusals counted by reason with an example program; iridia-dxil/tests/lift_corpus.rs | the user's own install: the containers of its superbundles' CompiledBytecode resources; primase's [corpus_skate_shaders example](https://github.com/Kalmarv/primase/tree/main/worlds/reskate/examples/corpus_skate_shaders.rs) extracts them |
| `oracles.corpus.sledding/shaders` | live | - | a folder (searched recursively) of the DXBC containers of a Unity player's compiled pixel programs, read out of its shader assets, lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs | the user's own install: the programs read from its shader assets |

#### Debug keys (`GAME_DATA_DEBUG`)

| key | about |
| --- | --- |
| `dxil.fuzz_runs` | how many mutated fixtures the DXIL fuzz test runs (default 3000) |

<!-- end of real-data inputs -->
