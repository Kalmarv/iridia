#![allow(clippy::unwrap_used)]
//! The IR lifters against the lifters they replace: every DXBC or DXIL container of a corpus is lifted both ways (by
//! `iridia_dxbc` or `iridia_dxil`, as the container holds Shader Model 4 and 5 code or a `DXIL` part), and each
//! vertex or pixel program both lift is run in the executor on the same deterministic inputs, constant buffers and
//! textures. Column A is the old lift, B the IR lift as lifted, P the IR lift after `promote`, `simplify` and
//! `break_flags`, and C that program lowered to stock naga (`iridia_naga`), written as WGSL and read back by naga's
//! front end, run in the executor A runs in; a program the lowering refuses is counted by its reason, not compared in
//! C. Every written output lane, the masks and the discard must agree bit for bit (NaNs alike); a
//! program the IR lifter refuses is counted by reason, and a program the old lifter lifts but the IR lifter does not
//! is a lifted-count regression.
//! Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or `sledding/shaders`. `IR_GATE_TRACE=1` names
//! each program as it starts.
mod inputs;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use inputs::Game;
use iridia::{
    BufferKind, FunctionSig, IoType, ItemList, ResourceId, ResourceKind, Stage, StageFacts, StorageTarget,
};
use iridia_exec::{
    Atomic, Backend, BufferAtomic, CallShader, Dim, Element, Executable, ImageAtomic, ImageLoad, ImageLod,
    ImageQuery, ImageSample, ImageStore, Load, PixelShader, ProgramShader, QueryKind, ReportHit, Resources,
    Runner, Sample, Sampler, Store, TraceRay,
};

mod common;

/// Whether a container holds a part of this kind (`DXIL`): its part table read as `dxcontainer` lays it out (the
/// count at byte 28, then each part's offset; at each, the part's kind).
fn has_part(bytes: &[u8], kind: &[u8; 4]) -> bool {
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at..at.checked_add(4)?)?;
        usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
    };
    let Some(count) = word(28) else {
        return false;
    };
    (0..count.min(64)).any(|k| {
        word(32 + 4 * k)
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
            .is_some_and(|k| k == kind)
    })
}

/// A container lifted both ways: by the older lift (A) and the IR lift (B) of its family, DXIL's for a container with
/// a `DXIL` part and DXBC's otherwise.
fn lift_both(
    bytes: &[u8],
) -> (
    Result<iridia::Program, iridia::Refusal>,
    Result<iridia_ir::Program, iridia::Refusal>,
) {
    if has_part(bytes, b"DXIL") {
        (iridia_dxil::lift(bytes), iridia_dxil::ir::lift(bytes))
    } else {
        (iridia_dxbc::lift(bytes), iridia_dxbc::ir::lift(bytes))
    }
}

/// The vectors each program is run on: the first two seed the constants no reflection names as floats, the next two
/// as small integers, and the last ([`ZEROED`]) is every constant and every unordered-access word 0.
const CASES: u32 = 5;

/// The vector whose constants, unordered-access memory, texel loads and buffer reads are all 0, as a cleared buffer
/// and an unset parameter are (its samples keep their hash): it reaches the code behind a guard against 0 or against
/// a constant (a debug pixel at the origin, a first use of a counter, a texel compared with an id).
const ZEROED: u32 = 4;

/// How vector `c` seeds the constants no reflection names (all of them, for [`ZEROED`]).
fn unstated(c: u32) -> common::Unstated {
    match c {
        0 | 1 => common::Unstated::Floats,
        ZEROED => common::Unstated::Zeros,
        _ => common::Unstated::Ints,
    }
}

/// Answers every request from a hash of everything it states, so two programs that ask the same agree and two that
/// ask differently (another coordinate, offset, level, component) almost surely do not.
#[derive(Default)]
struct Hashed {
    asked: Vec<String>,
    /// Every read request (a sample, a texel load, a query, a structured read) and atomic operation, in order, as the
    /// words its answer is hashed by: compared between the lifts, since a texel asked for differently is the
    /// program's own behaviour even where what it computes from it is dropped (a select that does not take it).
    reads: Vec<Vec<u32>>,
    /// The words the run's stores and atomic operations left, a texel never written read as its hash.
    written: Written,
    /// The element stride of each structured unordered-access view, by bind slot: a store's words are addressed by
    /// element and byte offset there, by element and lane in a typed buffer ([`Hashed::typed`]), by word in a raw one.
    strides: BTreeMap<u32, u32>,
    /// The bind slots of the typed unordered-access buffers.
    typed: std::collections::BTreeSet<u32>,
    /// A texel load answers small integers, not floats' bits: the vectors that seed the constants so (a count of
    /// lights read from a buffer of integers, which a float's bits make a loop of millions).
    ints: bool,
    /// Texel loads, buffer reads and an unordered-access word never written read 0 ([`ZEROED`]), not their hash.
    zeroed: bool,
}

impl Hashed {
    /// The sampler of test vector `c`.
    fn for_case(c: u32) -> Self {
        Hashed {
            ints: c >= 2,
            zeroed: c == ZEROED,
            ..Hashed::default()
        }
    }

    /// The sampler of test vector `c` for `shader`'s views.
    fn for_shader(shader: &ProgramShader, c: u32) -> Self {
        let resources = &shader.interface().resources;
        let strides = resources
            .iter()
            .filter_map(|r| match r.kind {
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Structured { stride }),
                    ..
                } => Some((r.provenance.d3d?.register, stride)),
                _ => None,
            })
            .collect();
        let typed = resources
            .iter()
            .filter_map(|r| match r.kind {
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(_)),
                    ..
                } => Some(r.provenance.d3d?.register),
                _ => None,
            })
            .collect();
        Hashed {
            strides,
            typed,
            ..Hashed::for_case(c)
        }
    }

    /// The word of buffer view `view` a store or an atomic operation at `address` reaches `k` words in.
    fn buffer_word(&self, view: u32, address: [u32; 3], k: u32) -> Word {
        match self.strides.get(&view) {
            Some(&stride) => Word::Element {
                view,
                element: address[0],
                word: (address[1] / 4).wrapping_add(k),
                stride,
            },
            // a typed buffer's lanes are its element's, four words to an element
            None if self.typed.contains(&view) => Word::Buffer {
                view,
                word: address[0].wrapping_mul(4).wrapping_add(k),
            },
            None => Word::Buffer {
                view,
                word: address[0].wrapping_add(k),
            },
        }
    }
}

/// A 32-bit word of an unordered-access view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Word {
    /// Of a raw or typed buffer, by word.
    Buffer { view: u32, word: u32 },
    /// Of a structured buffer: an element's word.
    Element {
        view: u32,
        element: u32,
        word: u32,
        stride: u32,
    },
    /// A lane of a texel.
    Texel {
        view: u32,
        dim: u32,
        texel: [u32; 3],
        layer: Option<u32>,
        lane: u32,
    },
}

/// What a pixel or vertex run's stores and atomic operations do: each word written, with the values written to it in
/// the order they were (the last is what it holds). Each word's sequence is compared, not the order of writes to
/// different words: Direct3D orders an invocation's writes to a word, and leaves the order between the lanes of a quad
/// open; a record a program rewrites in place leaves the same last words with a write left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Written {
    writes: BTreeMap<Word, Vec<u32>>,
}

impl Written {
    /// Word `at` takes `v`.
    fn write(&mut self, at: Word, v: u32) {
        self.writes.entry(at).or_default().push(v);
    }

    /// What word `at` holds, where it was written.
    fn last(&self, at: &Word) -> Option<u32> {
        self.writes.get(at).and_then(|v| v.last()).copied()
    }

    /// An atomic operation's change of word `at` (`fresh` where never written): the word as it was.
    fn atomic(&mut self, at: Word, fresh: u32, change: impl FnOnce(u32) -> u32) -> u32 {
        let old = self.last(&at).unwrap_or(fresh);
        self.write(at, change(old));
        old
    }

    /// The words with every NaN as one pattern, as outputs are compared.
    fn canonical(mut self) -> Self {
        for w in self.writes.values_mut().flatten() {
            *w = canonical(*w);
        }
        self
    }
}

fn hash_words(words: &[u32]) -> u64 {
    words
        .iter()
        .fold(0x1234_5678u64, |h, &w| common::mix64(h ^ u64::from(w)))
}

/// The mip levels, samples per texel, and size of each level of every texture the gate's sampler answers for.
const LEVELS: u32 = 9;
const SAMPLES: u32 = 4;
fn texture_size(level: u32) -> [u32; 3] {
    let shift = level.min(8);
    [(256 >> shift).max(1), (128 >> shift).max(1), 4]
}

/// Whether a texel address is inside the texture this sampler answers for: within the size of its level, a
/// level, layer and sample it has (as its queries answer). Outside, a load reads zero and a store writes nothing,
/// as Direct3D's views answer.
fn texel_inside(
    texture: u32,
    dim: Dim,
    texel: [u32; 3],
    layer: Option<u32>,
    sample: Option<u32>,
    mip: Option<u32>,
) -> bool {
    let mip = mip.unwrap_or(0);
    let lanes = match dim {
        Dim::D1 => 1,
        Dim::D2 | Dim::Cube => 2,
        Dim::D3 => 3,
    };
    let size = texture_size(mip);
    mip < LEVELS
        && texel.iter().zip(size).take(lanes).all(|(t, s)| *t < s)
        && layer.is_none_or(|l| l < texture_layers(texture, dim))
        && sample.is_none_or(|s| s < SAMPLES)
}

/// The array layers of texture `texture` of shape `dim` (1 to 6, by its hash).
fn texture_layers(texture: u32, dim: Dim) -> u32 {
    common::mix(hash_words(&[3, texture, dim_word(dim), QueryKind::Layers as u32])) % 6 + 1
}

fn dim_word(d: Dim) -> u32 {
    match d {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 => 3,
        Dim::Cube => 4,
    }
}

/// A float a request states, as the words it is hashed by: every NaN one pattern, since a sampler's answer does not
/// depend on a NaN's sign or payload, which IEEE 754 leaves open and two lifts compute differently.
fn bits(f: f32) -> u32 {
    canonical(f.to_bits())
}

fn texel_floats(h: u64) -> [f32; 4] {
    std::array::from_fn(|k| common::unit(common::mix(h ^ k as u64)) * 2.0 - 0.5)
}

impl Sampler for Hashed {
    fn sample(&mut self, r: Sample) -> Result<[f32; 4], String> {
        self.sample_image(ImageSample {
            texture: r.texture,
            sampler: r.sampler,
            dim: Dim::D2,
            coord: [r.uv[0], r.uv[1], 0.0],
            layer: r.layer,
            lod: ImageLod::from(r.lod),
            offset: [0; 3],
            gather: None,
            depth_ref: None,
            depth: false,
        })
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        let mut w = vec![1, r.texture, r.sampler, dim_word(r.dim)];
        w.extend(r.coord.map(bits));
        w.push(r.layer.map_or(u32::MAX, bits));
        match r.lod {
            ImageLod::Implicit => w.push(10),
            ImageLod::Bias(b) => w.extend([11, bits(b)]),
            ImageLod::Level(l) => w.extend([12, bits(l)]),
            ImageLod::Gradients(g) => {
                w.push(13);
                w.extend(g.iter().flatten().copied().map(bits));
            }
            ImageLod::GradientsBias(g, b) => {
                w.extend([14, bits(b)]);
                w.extend(g.iter().flatten().copied().map(bits));
            }
        }
        w.extend(r.offset.map(i32::cast_unsigned));
        w.push(r.gather.map_or(u32::MAX, u32::from));
        w.push(r.depth_ref.map_or(u32::MAX, bits));
        w.push(u32::from(r.depth));
        self.asked.push(format!("{r:?}"));
        let mut t = texel_floats(hash_words(&w));
        self.reads.push(w);
        if r.depth || r.depth_ref.is_some() {
            // a depth texture's one lane; the rest as a depth view reads them
            t = [t[0].clamp(0.0, 1.0), 0.0, 0.0, 1.0];
        }
        Ok(t)
    }
    fn load(&mut self, r: Load) -> Result<[u32; 4], String> {
        self.load_image(ImageLoad {
            texture: r.texture,
            dim: Dim::D2,
            storage: false,
            texel: r.texel,
            layer: None,
            sample: None,
            mip: Some(r.mip),
        })
    }
    fn load_image(&mut self, r: ImageLoad<u32>) -> Result<[u32; 4], String> {
        let mut w = vec![2, r.texture, dim_word(r.dim), u32::from(r.storage)];
        w.extend(r.texel.map(i32::cast_unsigned));
        w.push(r.layer.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.sample.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.mip.map_or(u32::MAX, i32::cast_unsigned));
        self.asked.push(format!("{r:?}"));
        let h = hash_words(&w);
        self.reads.push(w);
        let texel = r.texel.map(i32::cast_unsigned);
        let (layer, sample, mip) = (
            r.layer.map(i32::cast_unsigned),
            r.sample.map(i32::cast_unsigned),
            r.mip.map(i32::cast_unsigned),
        );
        if !texel_inside(r.texture, r.dim, texel, layer, sample, mip) {
            return Ok([0; 4]);
        }
        let mut t = if self.ints {
            std::array::from_fn(|k| common::mix(h ^ k as u64) % 4)
        } else {
            texel_floats(h).map(f32::to_bits)
        };
        if self.zeroed {
            // a cleared texture: a guard comparing a texel with a constant (an id lookup) holds
            t = [0; 4];
        }
        if r.storage {
            // what this run stored there
            for (lane, t) in (0..).zip(t.iter_mut()) {
                let at = Word::Texel {
                    view: r.texture,
                    dim: dim_word(r.dim),
                    texel: r.texel.map(i32::cast_unsigned),
                    layer: r.layer.map(i32::cast_unsigned),
                    lane,
                };
                if let Some(w) = self.written.last(&at) {
                    *t = w;
                }
            }
        }
        Ok(t)
    }
    fn query_image(&mut self, r: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let w = [
            3,
            r.texture,
            dim_word(r.dim),
            r.kind as u32,
            r.level.unwrap_or(u32::MAX),
        ];
        self.asked.push(format!("{r:?}"));
        self.reads.push(w.to_vec());
        Ok(match r.kind {
            QueryKind::Size => {
                let [x, y, z] = texture_size(r.level.unwrap_or(0));
                [x, y, z, 0]
            }
            QueryKind::Levels => [LEVELS, 0, 0, 0],
            QueryKind::Layers => [texture_layers(r.texture, r.dim), 0, 0, 0],
            QueryKind::Samples => [SAMPLES, 0, 0, 0],
        })
    }
    fn store(&mut self, r: Store) -> Result<(), String> {
        self.asked.push(format!("{r:?}"));
        for k in (0..4).filter(|k| r.mask >> k & 1 != 0) {
            let at = self.buffer_word(r.view, r.address, k);
            self.written.write(at, r.value[k as usize]);
        }
        Ok(())
    }
    fn store_image(&mut self, r: ImageStore<u32>) -> Result<(), String> {
        self.asked.push(format!("{r:?}"));
        if !texel_inside(r.view, r.dim, r.texel, r.layer, None, None) {
            return Ok(());
        }
        for (lane, &v) in (0..u32::from(r.lanes)).zip(&r.value) {
            let at = Word::Texel {
                view: r.view,
                dim: dim_word(r.dim),
                texel: r.texel,
                layer: r.layer,
                lane,
            };
            self.written.write(at, v);
        }
        Ok(())
    }
    fn atomic(&mut self, r: Atomic) -> Result<u32, String> {
        self.asked.push(format!("{r:?}"));
        let [x, y, z] = r.address;
        self.reads.push(vec![
            8,
            r.view,
            x,
            y,
            z,
            r.op as u32,
            r.value,
            r.compare,
            u32::from(r.signed),
        ]);
        let at = self.buffer_word(r.view, r.address, 0);
        let fresh = if self.zeroed {
            0
        } else {
            common::mix(hash_words(&[8, r.view, r.address[0], r.address[1], r.address[2]]))
        };
        Ok(self
            .written
            .atomic(at, fresh, |old| r.op.apply(old, r.value, r.compare, r.signed)))
    }
    fn atomic_image(&mut self, r: ImageAtomic<u32>) -> Result<u32, String> {
        self.asked.push(format!("{r:?}"));
        let [x, y, z] = r.texel;
        let layer = r.layer.unwrap_or(u32::MAX);
        self.reads.push(vec![
            5,
            r.view,
            x,
            y,
            z,
            layer,
            r.op as u32,
            r.value,
            r.compare,
            u32::from(r.signed),
        ]);
        let at = Word::Texel {
            view: r.view,
            dim: dim_word(r.dim),
            texel: r.texel,
            layer: r.layer,
            lane: 0,
        };
        let fresh = if self.zeroed {
            0
        } else {
            common::mix(hash_words(&[5, r.view, r.texel[0], r.texel[1], r.texel[2]]))
        };
        Ok(self
            .written
            .atomic(at, fresh, |old| r.op.apply(old, r.value, r.compare, r.signed)))
    }
    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        self.asked.push(format!("structured t{slot}[{element}]+{offset}"));
        self.reads.push(vec![4, slot, element, offset]);
        if self.zeroed {
            return Ok(0);
        }
        Ok(common::mix(hash_words(&[4, slot, element, offset])) % 1024)
    }
    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        let _ = slot;
        Some(64)
    }
}

/// What a run gives: each invocation's output lanes, masks and discard, the words its stores and atomic operations
/// left, and its read requests ([`Hashed::reads`]); or the error.
type Run = Result<(Vec<Invocation>, Written, Vec<Vec<u32>>), String>;

/// One invocation's written output lanes (unwritten lanes zero), its masks and whether it was discarded; a discarded
/// invocation's outputs are not compared, since nothing it writes reaches a target.
type Invocation = (Vec<[u32; 4]>, Vec<u8>, bool);

/// A run of test vector `c` (its invocation, or its quad's four).
fn run(shader: &ProgramShader, c: u32, cases: &[common::Case]) -> Run {
    run_asking(shader, cases, &mut Hashed::for_shader(shader, c))
}

/// A run of one invocation, or of a quad's four (lanes 0 and 1 the top row) whose derivatives are the quad's own.
fn run_asking(shader: &ProgramShader, cases: &[common::Case], sampler: &mut Hashed) -> Run {
    let first = cases.first().ok_or("no test vector")?;
    let buffers = common::refs(first, shader);
    let executions = match cases {
        [one] => vec![shader.execute(&one.inputs, &buffers, sampler)],
        [a, b, c, d] => {
            match shader.execute_quad([&a.inputs, &b.inputs, &c.inputs, &d.inputs], &buffers, sampler) {
                Ok(q) => q.into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            }
        }
        _ => {
            return Err(format!(
                "{} test vectors, where one or a quad's four are run",
                cases.len()
            ));
        }
    };
    let mut out = Vec::with_capacity(executions.len());
    for e in executions {
        out.push(invocation(&e.map_err(|e| e.to_string())?));
    }
    Ok((
        out,
        std::mem::take(&mut sampler.written).canonical(),
        std::mem::take(&mut sampler.reads),
    ))
}

/// What the gate compares of an execution: its written output lanes, each NaN as one pattern (a lane it did not write,
/// and every lane of a discarded execution, zero), its masks and whether it was discarded.
fn invocation(e: &iridia_exec::Execution) -> Invocation {
    let outputs = e
        .outputs
        .iter()
        .zip(&e.output_masks)
        .map(|(o, &m)| {
            std::array::from_fn(|k| {
                if m >> k & 1 != 0 && !e.discarded {
                    canonical(o[k])
                } else {
                    0
                }
            })
        })
        .collect();
    (outputs, e.output_masks.clone(), e.discarded)
}

/// Whether B's run `y` gives what A's run `x` gave: the same invocations and memory and, where `textures`, no texture
/// request A does not make.
fn same_run(x: &Run, y: &Run, textures: bool) -> bool {
    match (x, y) {
        (Ok((ix, wx, rx)), Ok((iy, wy, ry))) => {
            ix == iy && wx == wy && (!textures || texture_requests(ry).is_subset(&texture_requests(rx)))
        }
        _ => x == y,
    }
}

/// A NaN as one bit pattern: two lifts may produce different NaNs.
fn canonical(w: u32) -> u32 {
    if f32::from_bits(w).is_nan() {
        0x7fc0_0000
    } else {
        w
    }
}

/// The IR lift's program after the passes the cut-over runs, on each of its functions.
/// A program whose functions are graphs made regions (`structurize`, then `break_flags`), as the executor takes them; a
/// program of regions as it is.
fn structured(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, String> {
    for f in std::iter::once(&mut p.entry).chain(p.patch.as_mut()) {
        if matches!(f.body, iridia_ir::Body::Cfg(_)) {
            iridia_passes::structurize(f)?;
            iridia_passes::break_flags(f)?;
        }
    }
    Ok(p)
}

fn passed(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, String> {
    for f in std::iter::once(&mut p.entry).chain(p.patch.as_mut()) {
        iridia_passes::promote(f)?;
        iridia_passes::simplify(f);
        iridia_passes::break_flags(f)?;
    }
    iridia_ir::validate(&p).map_err(|e| format!("{e:?}"))?;
    Ok(p)
}

/// One program's comparison over `cases` vectors.
enum Verdict {
    /// Every case ran in A and gave what B gave.
    Agree,
    /// A case A itself cannot evaluate (its sampler request, a step limit): nothing to compare.
    AFails(String),
    /// The first case where B gives something else.
    Differs(String),
}

/// The inputs of a quad's lanes, in the order the gate tries them: a quad whose lanes take different paths to a
/// derivative or an implicit-level sample cannot be evaluated, and runs again with the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum QuadInputs {
    /// Each lane its own.
    Own,
    /// As a rasterizer interpolates them across a quad: each interpolated float input the first lane's plus a step
    /// of its own along x and along y (so its derivatives are those steps); the flat and integer ones the first
    /// lane's.
    Coherent,
    /// Every lane the first lane's (but its position): the derivatives of the inputs are 0.
    First,
}

/// Test vector `c`: the constants seeded as [`unstated`] says (a loop bound read from a stripped buffer needs small
/// integers), as [`Hashed`] then answers texel loads. A program that takes
/// derivatives of what it computes (`quad`) runs as a quad of four vectors: the lanes' inputs as `inputs` says, the
/// constants, the facing of lane 0 and positions a pixel apart.
fn gate_case(a: &ProgramShader, c: u32, quad: bool, inputs: QuadInputs) -> Vec<common::Case> {
    let unstated = unstated(c);
    if !quad {
        return vec![common::case_with(a, c, unstated)];
    }
    let own = |k: u32| if inputs == QuadInputs::Own { k } else { 0 };
    let mut lanes: Vec<common::Case> = (0..4)
        .map(|k| common::case_with(a, c * 4 + own(k), unstated))
        .collect();
    if inputs == QuadInputs::Coherent {
        for v in &a.interface().inputs {
            let interpolated = v.system.is_none()
                && v.value.scalar == iridia::ScalarType::Float
                && v.interp.is_none_or(|i| i.mode != iridia::InterpMode::Flat);
            let Some(sig) = v.provenance.d3d.filter(|_| interpolated) else {
                continue;
            };
            let r = sig.register as usize;
            if r >= lanes[0].inputs.len() {
                continue;
            }
            for lane in (0..4usize).filter(|l| sig.mask >> l & 1 != 0) {
                let quad = coherent_quad(f32::from_bits(lanes[0].inputs[r][lane]), c, sig.register, lane);
                for (l, x) in lanes.iter_mut().zip(quad) {
                    l.inputs[r][lane] = x.to_bits();
                }
            }
        }
    }
    let first = lanes[0].buffers.clone();
    let facing = a
        .interface()
        .inputs
        .iter()
        .filter(|v| v.system == Some(iridia::SystemValue::FrontFacing))
        .filter_map(|v| v.provenance.d3d.map(|d| d.register as usize))
        .collect::<Vec<_>>();
    let position = a
        .interface()
        .inputs
        .iter()
        .find(|v| v.system == Some(iridia::SystemValue::FragCoord))
        .and_then(|v| v.provenance.d3d.map(|d| d.register as usize));
    // an input the program does not read may lie past the registers the case seeds
    let seeded = lanes[0].inputs.len();
    let facing: Vec<usize> = facing.into_iter().filter(|&r| r < seeded).collect();
    let position = position.filter(|&r| r < seeded);
    let front: Vec<[u32; 4]> = facing.iter().map(|&r| lanes[0].inputs[r]).collect();
    for (k, lane) in lanes.iter_mut().enumerate() {
        lane.buffers = first.clone();
        for (&r, f) in facing.iter().zip(&front) {
            lane.inputs[r] = *f;
        }
        if let Some(r) = position {
            lane.inputs[r][0] = (0.5 + (k & 1) as f32).to_bits();
            lane.inputs[r][1] = (0.5 + (k >> 1) as f32).to_bits();
        }
    }
    lanes
}

/// Lane `lane` of input register `register` across a coherent quad of vector `c` whose first lane holds `base`: the
/// first lane's plus a step of its own along x and one along y (lanes 0 and 1 the top row).
fn coherent_quad(base: f32, c: u32, register: u32, lane: usize) -> [f32; 4] {
    let h = common::mix(hash_words(&[11, c, register, lane as u32]));
    let dx = (common::unit(h) - 0.5) / 16.0;
    let dy = (common::unit(common::mix(u64::from(h) ^ 1)) - 0.5) / 16.0;
    std::array::from_fn(|k| base + dx * (k & 1) as f32 + dy * (k >> 1) as f32)
}

/// A's run of each of the gate's vectors with the inputs it ran on, what B and P are both held to: `quad` whether each
/// vector is a quad, `furthest` the furthest of the [`QuadInputs`] a quad needed.
struct Older {
    quad: bool,
    runs: Vec<(Vec<common::Case>, Run)>,
    furthest: QuadInputs,
}

/// A on every vector. A quad whose lanes diverge at a derivative or an implicit-level sample runs again with coherent
/// inputs, then with every lane the first lane's.
fn older_runs(a: &ProgramShader, cases: u32, quad: bool) -> Older {
    let mut furthest = QuadInputs::Own;
    let runs = (0..cases)
        .map(|c| {
            let mut case = gate_case(a, c, quad, QuadInputs::Own);
            let mut x = run(a, c, &case);
            for next in [QuadInputs::Coherent, QuadInputs::First] {
                if !(quad && x.as_ref().is_err_and(|e| e.contains("diverge"))) {
                    break;
                }
                case = gate_case(a, c, quad, next);
                x = run(a, c, &case);
                furthest = furthest.max(next);
            }
            (case, x)
        })
        .collect();
    Older { quad, runs, furthest }
}

/// B against A's runs, on the inputs each ran on. A vector A cannot evaluate (a loop past the executor's budget on
/// these inputs, a request the sampler does not answer) is not compared, and B is not run on it: the second result is
/// why, for each; a program none of whose vectors A evaluates is `AFails`.
/// Where `textures`, every texture request (sample, texel load, query) B makes must be one A makes too: a texel the
/// program asks for differently is its own behaviour even where a select then drops it. A request A makes and B does
/// not is one for a value nothing uses, which B's compiler leaves out (buffer reads have no effect; a pass removes a
/// sample nothing reads, so P is held to its results only).
fn compare(a: &Older, b: &ProgramShader, textures: bool) -> (Verdict, Vec<String>) {
    let mut skipped = Vec::new();
    let mut compared = 0;
    for (c, (case, x)) in (0..).zip(&a.runs) {
        if let Err(e) = x {
            skipped.push(e.clone());
            continue;
        }
        let y = run(b, c, case);
        if !same_run(x, &y, textures) {
            return (
                Verdict::Differs(format!("case {c}: A {x:x?}\n  but {y:x?}")),
                skipped,
            );
        }
        compared += 1;
    }
    let verdict = match skipped.first() {
        Some(e) if compared == 0 => Verdict::AFails(e.clone()),
        _ => Verdict::Agree,
    };
    (verdict, skipped)
}

/// The distinct texture requests of a run's read requests ([`Hashed::reads`]): its samples, texel loads and queries.
fn texture_requests(reads: &[Vec<u32>]) -> std::collections::BTreeSet<&[u32]> {
    reads
        .iter()
        .filter(|r| matches!(r.first(), Some(1..=3)))
        .map(Vec::as_slice)
        .collect()
}

#[derive(Default)]
struct Gate {
    programs: usize,
    /// Old-lift vertex and pixel programs the executor runs.
    a: usize,
    agree_b: usize,
    agree_p: usize,
    /// Programs the IR lifter refuses that A lifted, by reason, with the first.
    refused: BTreeMap<String, (usize, String)>,
    /// Programs B lifted the executor (or a pass) refuses, by reason.
    unrun: BTreeMap<String, (usize, String)>,
    /// Programs A cannot evaluate on any vector, by reason.
    a_fails: BTreeMap<String, (usize, String)>,
    /// The vectors A cannot evaluate of programs B agrees with on the others, by reason: not compared.
    skipped: BTreeMap<String, (usize, String)>,
    /// The vectors of compute programs a column ran past the executor's step limit on where A finished, by column:
    /// not compared. The limit is of the operations a run executes, which a program's other form takes more or fewer
    /// of for the same result.
    over_budget: BTreeMap<String, (usize, String)>,
    /// Disagreements: program and the first difference.
    differ: Vec<String>,
    /// Programs the IR lifter lifts and A does not, by A's reason.
    only_ir: BTreeMap<String, (usize, String)>,
    /// Containers neither lifter lifts (DXIL among them), by the IR lifter's reason.
    neither: BTreeMap<String, (usize, String)>,
    /// Compute programs among those A runs.
    compute: usize,
    /// Mesh programs among those A runs.
    mesh: usize,
    /// Programs made of functions (hull, domain, geometry, a vertex program feeding a tessellator) among those A runs.
    functions: usize,
    /// Programs a quad of whose ran with coherent inputs at most, and with the first lane's in every lane.
    coherent: usize,
    identical: usize,
    /// Of the programs that ran coherent quads and agree, those that take no derivative on any vector with coherent
    /// inputs (their quads diverged at an implicit-level sample only), and those in which every lane taking one is
    /// discarded (a derivative reaches no output there): what the coherent vectors leave untested.
    coherent_no_derivative: usize,
    coherent_derivatives_discarded: usize,
    /// Programs C agrees with A on (every vector A evaluates).
    agree_c: usize,
    /// Programs the lowering refuses, by reason: not compared in C.
    c_unlowered: BTreeMap<String, (usize, String)>,
    /// Programs the lowering takes whose WGSL C cannot build or run, by reason: failures.
    c_unbuilt: BTreeMap<String, (usize, String)>,
}

fn tally(map: &mut BTreeMap<String, (usize, String)>, why: &str, at: &str) {
    map.entry(common::reason(why)).or_insert((0, at.to_string())).0 += 1;
}

impl Gate {
    /// Another program's tallies added to these: its reasons' first examples kept where these have none.
    fn absorb(&mut self, other: Gate) {
        let Gate {
            programs,
            a,
            agree_b,
            agree_p,
            refused,
            unrun,
            a_fails,
            skipped,
            over_budget,
            differ,
            only_ir,
            neither,
            compute,
            mesh,
            functions,
            coherent,
            identical,
            coherent_no_derivative,
            coherent_derivatives_discarded,
            agree_c,
            c_unlowered,
            c_unbuilt,
        } = other;
        self.programs += programs;
        self.a += a;
        self.agree_b += agree_b;
        self.agree_p += agree_p;
        self.compute += compute;
        self.mesh += mesh;
        self.functions += functions;
        self.coherent += coherent;
        self.identical += identical;
        self.coherent_no_derivative += coherent_no_derivative;
        self.coherent_derivatives_discarded += coherent_derivatives_discarded;
        self.agree_c += agree_c;
        self.differ.extend(differ);
        for (mine, theirs) in [
            (&mut self.refused, refused),
            (&mut self.unrun, unrun),
            (&mut self.a_fails, a_fails),
            (&mut self.skipped, skipped),
            (&mut self.over_budget, over_budget),
            (&mut self.only_ir, only_ir),
            (&mut self.neither, neither),
            (&mut self.c_unlowered, c_unlowered),
            (&mut self.c_unbuilt, c_unbuilt),
        ] {
            for (why, (n, e)) in theirs {
                mine.entry(why).or_insert((0, e)).0 += n;
            }
        }
    }

    /// A vector `column` ran past the executor's step limit on where A finished.
    fn past_budget(&mut self, column: &str, at: &str) {
        tally(
            &mut self.over_budget,
            &format!("{column} ran past the executor's step limit on, where A finished: not compared"),
            at,
        );
    }

    /// C's program of `p` ([`lower_c`]), or `None` with why tallied.
    fn lowered(
        &mut self,
        old: &iridia::Program,
        p: &iridia_ir::Program,
        at: &str,
    ) -> Option<iridia::Program> {
        match lower_c(old, p) {
            Ok(c) => Some(c),
            Err(Unlowered::Refused(e)) => {
                tally(&mut self.c_unlowered, &e, at);
                None
            }
            Err(Unlowered::Unbuilt(e)) => {
                tally(&mut self.c_unbuilt, &e, at);
                None
            }
        }
    }

    /// Settle a column of a program whose compared vectors agree: `compared` the vectors A evaluated, `skipped` why it
    /// could not evaluate the others. With none compared A fails the program (`false`); otherwise the column agrees,
    /// and B's skipped vectors are counted.
    fn settle(&mut self, column: &str, compared: usize, skipped: &[String], at: &str) -> bool {
        if compared == 0 {
            tally(
                &mut self.a_fails,
                skipped.first().map_or("no vector", String::as_str),
                at,
            );
            return false;
        }
        if column == "B" {
            self.agree_b += 1;
            for e in skipped {
                tally(&mut self.skipped, e, at);
            }
        } else {
            self.agree_p += 1;
        }
        true
    }
}

fn gate(game: Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    gate_over(&corpus, &[]);
}

/// What checking one program took: lifting both ways and building A and B, A's runs, B's runs and their comparison,
/// and P's (the passes, building P, its runs and their comparison); a program not compared all in `lift`.
#[derive(Clone, Copy, Debug, Default)]
struct Took {
    lift: Duration,
    a: Duration,
    b: Duration,
    p: Duration,
    c: Duration,
}

impl Took {
    fn total(&self) -> Duration {
        self.lift + self.a + self.b + self.p + self.c
    }
}

/// The time since `*mark`, and `*mark` moved to now: the phases of [`Took`] one after another.
fn lap(mark: &mut Instant) -> Duration {
    let now = Instant::now();
    let took = now - *mark;
    *mark = now;
    took
}

/// Where the gate's time went (`IR_GATE_TIMES`): the phases summed over the programs, against the wall time the
/// `workers` threads had, and the slowest programs.
fn report_times(took: &mut [(Took, String)], workers: usize) {
    let sum = |f: fn(&Took) -> Duration| took.iter().map(|(t, _)| f(t)).sum::<Duration>();
    println!(
        "  times over {} programs on {workers} threads: lifting and building {:.1} s, A's runs {:.1} s, B's {:.1} s, \
         P's {:.1} s, C's {:.1} s",
        took.len(),
        sum(|t| t.lift).as_secs_f64(),
        sum(|t| t.a).as_secs_f64(),
        sum(|t| t.b).as_secs_f64(),
        sum(|t| t.p).as_secs_f64(),
        sum(|t| t.c).as_secs_f64()
    );
    took.sort_by_key(|(t, _)| std::cmp::Reverse(t.total()));
    for (t, at) in took.iter().take(20) {
        println!(
            "  {:7.2} s ({:.2} + {:.2} + {:.2} + {:.2} + {:.2})  {at}",
            t.total().as_secs_f64(),
            t.lift.as_secs_f64(),
            t.a.as_secs_f64(),
            t.b.as_secs_f64(),
            t.p.as_secs_f64(),
            t.c.as_secs_f64()
        );
    }
}

/// One program of a corpus through the gate, tallied in `g`.
fn check(g: &mut Gate, bytes: &[u8], at: &str, took: &mut Took) {
    let mut mark = Instant::now();
    g.programs += 1;
    let (old, p) = match lift_both(bytes) {
        (Ok(old), Ok(p)) => (old, p),
        (Ok(_), Err(e)) => {
            tally(&mut g.refused, &e.to_string(), at);
            return;
        }
        (Err(e), Ok(_)) => {
            tally(&mut g.only_ir, &e.to_string(), at);
            return;
        }
        (Err(_), Err(e)) => {
            tally(&mut g.neither, &e.to_string(), at);
            return;
        }
    };
    // a graph (DXIL's) runs as the regions `structurize` makes of it, each exit leaving its innermost construct, as
    // the older lift structures it as it lifts
    let p = match structured(p) {
        Ok(p) => p,
        Err(e) => {
            tally(&mut g.unrun, &format!("B not structured: {e}"), at);
            return;
        }
    };
    took.lift = lap(&mut mark);
    if matches!(old.interface.stage, Stage::Compute | Stage::Mesh) {
        compute(g, &old, p, at, took);
        return;
    }
    match iridia::stage_functions(&old.interface) {
        Ok(Some(sigs)) => {
            functions(g, &old, &sigs, p, at, took);
            return;
        }
        Ok(None) => {}
        Err(e) => {
            tally(&mut g.a_fails, &format!("A's functions: {e}"), at);
            return;
        }
    }
    let a = match ProgramShader::new(old.clone()) {
        Ok(a) => a,
        Err(e) => {
            tally(&mut g.a_fails, &format!("A not built: {e}"), at);
            return;
        }
    };
    g.a += 1;
    let b = match ProgramShader::from_ir(p.clone()) {
        Ok(b) => b,
        Err(e) => {
            tally(&mut g.unrun, &format!("B: {e}"), at);
            return;
        }
    };
    took.lift += lap(&mut mark);
    let older = older_runs(&a, CASES, a.needs_quad() || b.needs_quad());
    took.a = lap(&mut mark);
    let (verdict, skipped) = compare(&older, &b, true);
    let quads = older.furthest;
    took.b = lap(&mut mark);
    match quads {
        QuadInputs::Own => {}
        QuadInputs::Coherent => g.coherent += 1,
        QuadInputs::First => g.identical += 1,
    }
    if quads == QuadInputs::Coherent && matches!(verdict, Verdict::Agree) {
        let runs = coherent_runs(&b);
        g.coherent_no_derivative += usize::from(!runs.derivatives);
        g.coherent_derivatives_discarded += usize::from(runs.derivatives && !runs.kept);
        took.b += lap(&mut mark);
    }
    match verdict {
        Verdict::Agree => {
            g.agree_b += 1;
            for e in &skipped {
                tally(&mut g.skipped, e, at);
            }
        }
        Verdict::AFails(e) => {
            tally(&mut g.a_fails, &e, at);
            return;
        }
        Verdict::Differs(d) => g.differ.push(format!("{at} (B)\n  {d}")),
    }
    let p = match passed(p) {
        Ok(p) => p,
        Err(e) => {
            tally(&mut g.unrun, &format!("P: {e}"), at);
            return;
        }
    };
    let pp = match ProgramShader::from_ir(p.clone()) {
        Ok(pp) => pp,
        Err(e) => {
            tally(&mut g.unrun, &format!("P: {e}"), at);
            return;
        }
    };
    // A's runs serve P too, unless P runs quads where B did not or the other way (a pass removed a derivative)
    let quad = a.needs_quad() || pp.needs_quad();
    let older = if quad == older.quad {
        older
    } else {
        older_runs(&a, CASES, quad)
    };
    let p_verdict = compare(&older, &pp, false).0;
    took.p = lap(&mut mark);
    match p_verdict {
        Verdict::Agree => g.agree_p += 1,
        Verdict::AFails(e) => tally(&mut g.a_fails, &e, at),
        Verdict::Differs(d) => g.differ.push(format!("{at} (P)\n  {d}")),
    }
    let Some(c) = g.lowered(&old, &p, at) else {
        took.c = lap(&mut mark);
        return;
    };
    let c = match ProgramShader::new(c) {
        Ok(c) => c,
        Err(e) => {
            tally(&mut g.c_unbuilt, &format!("not compiled: {e}"), at);
            took.c = lap(&mut mark);
            return;
        }
    };
    let quad = a.needs_quad() || c.needs_quad();
    let older = if quad == older.quad {
        older
    } else {
        older_runs(&a, CASES, quad)
    };
    match compare(&older, &c, false).0 {
        // A's failures are B's and P's, counted there
        Verdict::Agree | Verdict::AFails(_) => g.agree_c += 1,
        Verdict::Differs(d) => g.differ.push(format!("{at} (C)\n  {d}")),
    }
    took.c = lap(&mut mark);
}

/// What naga's WGSL front end makes of the lowering's text cannot say everything the older module says, which the
/// executor reads: put back from the older module by position, a global's name ending in a digit (naga's namer writes
/// `cb0_` for `cb0`) and the entry point's argument and result member names and bindings (WGSL cannot say a varying
/// has no interpolation, so the front end supplies the default). Anything else that differs is left for the executor
/// to refuse or to compute differently.
fn normalise(m: &mut naga::Module, old: &naga::Module) -> Result<(), String> {
    for (_, g) in m.global_variables.iter_mut() {
        if let Some(n) = &mut g.name
            && n.len() > 1
            && n.ends_with('_')
            && n[..n.len() - 1].ends_with(|c: char| c.is_ascii_digit())
        {
            n.pop();
        }
    }
    let (Some(ep), Some(oe)) = (m.entry_points.first_mut(), old.entry_points.first()) else {
        return Err("no entry point".into());
    };
    if ep.function.arguments.len() != oe.function.arguments.len() {
        return Err(format!(
            "{} arguments, where A's entry point has {}",
            ep.function.arguments.len(),
            oe.function.arguments.len()
        ));
    }
    for (a, o) in ep.function.arguments.iter_mut().zip(&oe.function.arguments) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    match (&ep.function.result, &oe.function.result) {
        (None, None) => Ok(()),
        (Some(r), Some(or)) => {
            let (rt, ort) = (r.ty, or.ty);
            let naga::TypeInner::Struct { members: om, .. } = &old.types[ort].inner else {
                return Err("A's result is not a structure".into());
            };
            let mut t = m.types[rt].clone();
            let naga::TypeInner::Struct { members, .. } = &mut t.inner else {
                return Err("the result is not a structure".into());
            };
            if members.len() != om.len() {
                return Err(format!(
                    "{} result members, where A's has {}",
                    members.len(),
                    om.len()
                ));
            }
            for (a, o) in members.iter_mut().zip(om) {
                a.name.clone_from(&o.name);
                a.binding.clone_from(&o.binding);
            }
            t.name.clone_from(&old.types[ort].name);
            m.types.replace(rt, t);
            Ok(())
        }
        _ => Err("a result where A's entry point has none, or none where it has one".into()),
    }
}

/// Why C has no program to run.
enum Unlowered {
    /// The lowering refuses the program: counted by its reason.
    Refused(String),
    /// The lowering made WGSL that is not the older program's interface, or not WGSL naga reads back: a failure.
    Unbuilt(String),
}

/// C's program for `iridia-exec`: `p` lowered for naga's own dialect and written as WGSL, read back by naga's front
/// end, with the older lift's interface and each of its resources bound to the global its binding plan names.
fn lower_c(old: &iridia::Program, p: &iridia_ir::Program) -> Result<iridia::Program, Unlowered> {
    // C runs where A does, in the executor of naga-path programs, which validates no 64-bit value
    let wide = |s: iridia_ir::Scalar| {
        matches!(
            s,
            iridia_ir::Scalar::U64 | iridia_ir::Scalar::I64 | iridia_ir::Scalar::F64
        )
    };
    if std::iter::once(&p.entry)
        .chain(p.patch.as_ref())
        .any(|f| f.values.iter().any(|d| wide(d.ty.scalar)))
    {
        return Err(Unlowered::Refused(
            "a 64-bit value, which the executor of naga-path programs does not validate".into(),
        ));
    }
    // a diagnostic: C's text from a file (`one`), to try a change to it
    let wgsl = match std::env::var("IR_GATE_C_WGSL") {
        Ok(path) => std::fs::read_to_string(path).unwrap(),
        // a module naga rejects after the preflight took the program is the lowering's fault, not a refusal
        Err(_) => iridia_naga::wgsl(p, &iridia_naga::Profile::NAGA).map_err(|e| {
            if e.0.starts_with("lowering bug") {
                Unlowered::Unbuilt(e.0)
            } else {
                Unlowered::Refused(e.0)
            }
        })?,
    };
    let unbuilt = |e: String| Unlowered::Unbuilt(e);
    let mut module = naga::front::wgsl::parse_str(&wgsl)
        .map_err(|e| unbuilt(format!("naga's WGSL front end: {}", e.emit_to_string(&wgsl))))?;
    normalise(&mut module, &old.module).map_err(unbuilt)?;
    let mut bindings = Vec::new();
    for r in &old.interface.resources {
        // the older DXIL lift's buffer of a view's hidden counter is the plan's counter of that view
        let (owner, suffix) = match r.provenance.counter_of {
            Some(view) => (
                old.interface
                    .resources
                    .iter()
                    .find(|v| v.id == view)
                    .ok_or_else(|| {
                        unbuilt(format!("{} counts a view the interface lacks", r.provenance.name))
                    })?,
                "_counter",
            ),
            None => (r, ""),
        };
        let d = owner
            .provenance
            .d3d
            .ok_or_else(|| unbuilt(format!("resource {} has no D3D binding", r.provenance.name)))?;
        let prefix = match d.class {
            iridia::D3dClass::ConstantBuffer => "cb",
            iridia::D3dClass::ShaderResource => "t",
            iridia::D3dClass::Sampler => "s",
            iridia::D3dClass::UnorderedAccess => "u",
        };
        // the binding plan's names (`iridia_naga::BindingPlan::d3d`)
        let name = if d.space == 0 {
            format!("{prefix}{}{suffix}", d.register)
        } else {
            format!("{prefix}{}_space{}{suffix}", d.register, d.space)
        };
        let g = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(&name))
            .map(|(h, _)| h)
            .ok_or_else(|| unbuilt(format!("no global {name}")))?;
        bindings.push((r.id, g));
    }
    let mut c = old.clone();
    c.module = module;
    c.bindings = bindings;
    Ok(c)
}

/// The gate over `corpus`'s containers: a refusal is a failure unless its reason contains one of `not_yet`.
fn gate_over(corpus: &common::Corpus, not_yet: &[&str]) {
    let name = &corpus.name;
    let limit = common::diff_limit();
    let trace = std::env::var_os("IR_GATE_TRACE").is_some();
    let files = &corpus.names[..corpus.names.len().min(limit)];
    // each program is checked on its own, as many at a time as the machine runs threads; their tallies are merged in
    // the corpus's order, so the report (the first example of each reason among them) is the one a serial run gives
    let next = std::sync::atomic::AtomicUsize::new(0);
    let checked: Vec<std::sync::Mutex<Option<(Gate, Took)>>> =
        files.iter().map(|_| Default::default()).collect();
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(file) = files.get(i) else {
                        break;
                    };
                    let at = corpus.at(file);
                    if trace {
                        eprintln!("  {name} {}: {at}", i + 1);
                    } else if (i + 1).is_multiple_of(500) {
                        eprintln!("  {name}: {} programs", i + 1);
                    }
                    let (mut g, mut took) = (Gate::default(), Took::default());
                    let started = Instant::now();
                    // a folder's files are read one at a time ([`common::Corpus`]); the checks run at once
                    let bytes = corpus.read(file);
                    check(&mut g, &bytes, &at, &mut took);
                    // what a program left uncompared took (its lift refused, A not built) counts as lifting
                    took.lift += started.elapsed().saturating_sub(took.total());
                    *checked[i].lock().unwrap() = Some((g, took));
                }
            });
        }
    });
    let mut g = Gate::default();
    let mut took = Vec::with_capacity(files.len());
    for (file, c) in files.iter().zip(checked) {
        let (one, t) = c.into_inner().unwrap().expect("every program is checked");
        g.absorb(one);
        took.push((t, corpus.at(file)));
    }
    if std::env::var_os("IR_GATE_TIMES").is_some() {
        report_times(&mut took, workers);
    }
    println!(
        "{name}: {} programs, A runs {} ({} compute, {} mesh, {} of functions), B agrees {}, P agrees {}, C agrees {} \
         ({} not lowered, {} not built), {} differ, IR refuses {}, only the IR lifts {}, neither lifts {}",
        g.programs,
        g.a,
        g.compute,
        g.mesh,
        g.functions,
        g.agree_b,
        g.agree_p,
        g.agree_c,
        g.c_unlowered.values().map(|(n, _)| n).sum::<usize>(),
        g.c_unbuilt.values().map(|(n, _)| n).sum::<usize>(),
        g.differ.len(),
        g.refused.values().map(|(n, _)| n).sum::<usize>(),
        g.only_ir.values().map(|(n, _)| n).sum::<usize>(),
        g.neither.values().map(|(n, _)| n).sum::<usize>(),
    );
    println!(
        "  quads rerun with coherent inputs {}, with the first lane's in every lane {}; of the coherent, {} take no derivative on those vectors and {} discard every lane that takes one",
        g.coherent, g.identical, g.coherent_no_derivative, g.coherent_derivatives_discarded
    );
    for (why, (n, e)) in &g.refused {
        println!("  refused {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.unrun {
        println!("  not run {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.a_fails {
        println!("  A fails {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.skipped {
        println!("  skipped {n:5}  vectors of agreeing programs A cannot evaluate: {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.over_budget {
        println!("  skipped {n:5}  vectors {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.only_ir {
        println!("  IR only {n:5}  A: {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.neither {
        println!("  neither {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.c_unlowered {
        println!("  C not lowered {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.c_unbuilt {
        println!("  C NOT BUILT {n:5}  {why}  (e.g. {e})");
    }
    for d in g.differ.iter().take(20) {
        println!("  DIFFER {d}");
    }
    // every one by name, past the first twenty in full
    for d in g.differ.iter().skip(20) {
        println!("  differs {}", d.lines().next().unwrap_or_default());
    }
    assert!(g.differ.is_empty(), "{} programs differ", g.differ.len());
    assert!(
        g.c_unbuilt.is_empty(),
        "{} lowered programs C does not build or run",
        g.c_unbuilt.values().map(|(n, _)| n).sum::<usize>()
    );
    assert!(
        g.unrun.is_empty(),
        "{} IR-lifted programs do not run",
        g.unrun.values().map(|(n, _)| n).sum::<usize>()
    );
    let regressions: Vec<&String> = g
        .refused
        .keys()
        .filter(|why| !not_yet.iter().any(|n| why.contains(n)))
        .collect();
    assert!(
        regressions.is_empty(),
        "the IR lifter refuses programs the old one lifts: {regressions:?}"
    );
}

/// Unordered-access texels by view, element, texel and layer: their four lanes.
type Texels = BTreeMap<(u32, u32, [u32; 3], Option<u32>), [u32; 4]>;

/// The memory a compute dispatch leaves: each read-write buffer's words, each texel its stores and atomic operations
/// reached, and each counter.
#[derive(Debug, PartialEq, Eq)]
struct Memory {
    buffers: BTreeMap<ResourceId, Vec<u32>>,
    elements: BTreeMap<(ResourceId, u32), Vec<u32>>,
    texels: Texels,
    counters: BTreeMap<(u32, u32), u32>,
    /// A mesh program's: the words of the meshlet each thread group wrote, in the order of the dispatch.
    meshlets: Vec<Vec<u32>>,
    /// A ray tracing program's: each `TraceRay`, `ReportHit` and `CallShader` it made, in order, as the words it was
    /// given (see [`Group`]).
    rays: Vec<Vec<u32>>,
}

impl Memory {
    /// The memory with every word that is a NaN as one pattern, as outputs are compared: a float a program stores
    /// keeps the NaN its arithmetic made, whose sign and payload IEEE 754 leaves open.
    fn canonical(mut self) -> Self {
        for words in self
            .buffers
            .values_mut()
            .chain(self.elements.values_mut())
            .chain(&mut self.meshlets)
            .chain(&mut self.rays)
        {
            for w in words.iter_mut() {
                *w = canonical(*w);
            }
        }
        for t in self.texels.values_mut() {
            *t = t.map(canonical);
        }
        self
    }
}

/// A compute run's backend: textures answered by the hash, unordered-access texels kept as the program leaves them
/// (a texel never written reads as its hash), counters starting from a hash of their view. A ray tracing program's
/// operations are answered as a pipeline whose rays hit nothing and whose programs change nothing would answer them,
/// a reported hit accepted by its distance and kind; each is kept with what it was given.
struct Group {
    hashed: Hashed,
    texels: Texels,
    counters: BTreeMap<(u32, u32), u32>,
    rays: Vec<Vec<u32>>,
}

impl Group {
    /// The backend of test vector `c`.
    fn for_case(c: u32) -> Self {
        Group {
            hashed: Hashed::for_case(c),
            texels: BTreeMap::new(),
            counters: BTreeMap::new(),
            rays: Vec::new(),
        }
    }
}

/// An element of the interface as the one word the hashed answers take.
fn element_word(e: Element) -> u32 {
    e.id.0 | e.index << 16
}

impl Backend for Group {
    fn trace_ray(&mut self, r: TraceRay<'_>) -> Result<(), String> {
        let f = f32::to_bits;
        self.rays.push(vec![
            157,
            r.scene.0,
            r.flags,
            r.mask,
            r.hit_group_offset,
            r.hit_group_stride,
            r.miss_index,
            f(r.origin[0]),
            f(r.origin[1]),
            f(r.origin[2]),
            f(r.t_min),
            f(r.direction[0]),
            f(r.direction[1]),
            f(r.direction[2]),
            f(r.t_max),
        ]);
        Ok(())
    }
    fn report_hit(&mut self, r: ReportHit<'_>) -> Result<bool, String> {
        let t = canonical(r.t.to_bits());
        self.rays.push(vec![158, t, r.hit_kind]);
        Ok((t ^ r.hit_kind.rotate_left(16)).count_ones().is_multiple_of(2))
    }
    fn call_shader(&mut self, r: CallShader<'_>) -> Result<(), String> {
        self.rays.push(vec![159, r.index]);
        Ok(())
    }
    fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        self.hashed.sample_image(ImageSample {
            texture: element_word(r.texture),
            sampler: element_word(r.sampler),
            dim: r.dim,
            coord: r.coord,
            layer: r.layer,
            lod: r.lod,
            offset: r.offset,
            gather: r.gather,
            depth_ref: r.depth_ref,
            depth: r.depth,
        })
    }
    fn load(&mut self, r: ImageLoad<Element>) -> Result<[u32; 4], String> {
        if r.storage
            && !texel_inside(
                element_word(r.texture),
                r.dim,
                r.texel.map(i32::cast_unsigned),
                r.layer.map(i32::cast_unsigned),
                None,
                None,
            )
        {
            return Ok([0; 4]);
        }
        if r.storage {
            let key = (
                r.texture.id.0,
                r.texture.index,
                r.texel.map(i32::cast_unsigned),
                r.layer.map(i32::cast_unsigned),
            );
            if let Some(t) = self.texels.get(&key) {
                return Ok(*t);
            }
            if self.hashed.zeroed {
                return Ok([0; 4]);
            }
        }
        self.hashed.load_image(ImageLoad {
            texture: element_word(r.texture),
            dim: r.dim,
            storage: r.storage,
            texel: r.texel,
            layer: r.layer,
            sample: r.sample,
            mip: r.mip,
        })
    }
    fn query(&mut self, r: ImageQuery<Element>) -> Result<[u32; 4], String> {
        self.hashed.query_image(ImageQuery {
            texture: element_word(r.texture),
            dim: r.dim,
            kind: r.kind,
            level: r.level,
        })
    }
    fn store_image(&mut self, r: ImageStore<ResourceId>) -> Result<(), String> {
        let word = element_word(Element {
            id: r.view,
            index: r.index,
        });
        if !texel_inside(word, r.dim, r.texel, r.layer, None, None) {
            return Ok(());
        }
        let key = (r.view.0, r.index, r.texel, r.layer);
        let mut t = self.texels.get(&key).copied().unwrap_or_default();
        for (to, from) in t.iter_mut().zip(r.value).take(usize::from(r.lanes)) {
            *to = from;
        }
        self.texels.insert(key, t);
        Ok(())
    }
    fn atomic_image(&mut self, r: ImageAtomic<ResourceId>) -> Result<u32, String> {
        let key = (r.view.0, r.index, r.texel, r.layer);
        let fresh = if self.hashed.zeroed {
            0
        } else {
            common::mix(hash_words(&[
                5, r.view.0, r.index, r.texel[0], r.texel[1], r.texel[2],
            ]))
        };
        let t = self.texels.entry(key).or_insert([fresh, 0, 0, 0]);
        let old = t[0];
        t[0] = r.op.apply(old, r.value, r.compare, r.signed);
        Ok(old)
    }
    fn atomic_buffer(&mut self, r: BufferAtomic) -> Result<u32, String> {
        Err(format!(
            "resource {}: a buffer the run does not hold",
            r.view.id.0
        ))
    }
    fn counter(&mut self, view: Element, decrement: bool) -> Result<u32, String> {
        let zeroed = self.hashed.zeroed;
        let c = self
            .counters
            .entry((view.id.0, view.index))
            .or_insert_with(|| counter_start(zeroed, view.id.0, view.index));
        if decrement {
            *c = c.wrapping_sub(1);
            Ok(*c)
        } else {
            let before = *c;
            *c = c.wrapping_add(1);
            Ok(before)
        }
    }
}

/// The count a view's hidden counter starts at: 0 in the zeroed vector, else a hash of the view.
fn counter_start(zeroed: bool, view: u32, element: u32) -> u32 {
    if zeroed {
        0
    } else {
        common::mix(hash_words(&[6, view, element])) % 64
    }
}

/// The thread groups a compute program is dispatched over.
const GROUPS: [u32; 3] = [2, 1, 1];

/// The elements of an array of resources a compute run binds: all of a small array, 32 of a larger or unbounded
/// one (a program indexing past them reads a binding it was not given, which the run refuses).
fn bound_elements(count: u32) -> u32 {
    count.min(32)
}

/// What a compute run binds and changes, seeded by case: the constant and read-only buffers bound (an array's
/// elements each), the read-write ones the dispatch changes. Each buffer has 64 elements of small words, so an
/// address the program computes from one stays mostly inside.
struct Seeded {
    resources: Resources,
    buffers: BTreeMap<ResourceId, Vec<u32>>,
    elements: BTreeMap<(ResourceId, u32), Vec<u32>>,
}

fn seeded(program: &iridia::Program, c: u32) -> Seeded {
    let mut s = Seeded {
        resources: Resources::default(),
        buffers: BTreeMap::new(),
        elements: BTreeMap::new(),
    };
    for r in &program.interface.resources {
        let array = r.count != 1;
        let (kind, storage) = match &r.kind {
            ResourceKind::Buffer(k) => (k, false),
            ResourceKind::Storage {
                target: StorageTarget::Buffer(k),
                ..
            } => (k, true),
            ResourceKind::ConstantBuffer { layout } => {
                let rows = layout.size.div_ceil(16).max(1);
                for e in 0..bound_elements(r.count) {
                    let v = (0..rows)
                        .map(|row| {
                            std::array::from_fn(|k| {
                                if c == ZEROED {
                                    0
                                } else {
                                    common::mix(hash_words(&[7, c, r.id.0, e, row, k as u32])) % 97
                                }
                            })
                        })
                        .collect();
                    if array {
                        s.resources.constant_elements.insert((r.id, e), v);
                    } else {
                        s.resources.constants.insert(r.id, v);
                    }
                }
                continue;
            }
            _ => continue,
        };
        // a counter the older DXIL lift makes a buffer of its own (a word per element of an array of views) starts
        // where the backend's counter of its view does
        if let Some(view) = r.provenance.counter_of {
            let v = (0..256).map(|k| counter_start(c == ZEROED, view.0, k)).collect();
            s.buffers.insert(r.id, v);
            continue;
        }
        let words = match kind {
            BufferKind::Typed(_) => 64 * 4,
            BufferKind::Structured { stride } => 64 * stride.div_ceil(4),
            BufferKind::Raw => 256,
        };
        for e in 0..bound_elements(r.count) {
            // the zeroed vector's read-write buffers start cleared
            let v: Vec<u32> = (0..words)
                .map(|w| {
                    if storage && c == ZEROED {
                        0
                    } else {
                        common::mix(hash_words(&[8, c, r.id.0, e, w])) % 251
                    }
                })
                .collect();
            match (storage, array) {
                (true, true) => {
                    s.elements.insert((r.id, e), v);
                }
                (true, false) => {
                    s.buffers.insert(r.id, v);
                }
                (false, true) => {
                    s.resources.elements.insert((r.id, e), v);
                }
                (false, false) => {
                    s.resources.buffers.insert(r.id, v);
                }
            }
        }
    }
    s
}

/// One dispatch of `exe` (of `program`'s interface) on case `c`'s memory, and the memory it leaves.
fn dispatch(exe: &Executable, program: &iridia::Program, c: u32) -> Result<Memory, String> {
    let mut s = seeded(program, c);
    let bound = exe.bind_resources(&s.resources)?;
    let mut backend = Group::for_case(c);
    let mut meshlets = Vec::new();
    if program.interface.stage == Stage::Mesh {
        exe.dispatch_mesh_elements(
            &bound,
            &mut backend,
            &mut s.buffers,
            &mut s.elements,
            GROUPS,
            32,
            &[],
            &mut |_, words| {
                meshlets.push(words.to_vec());
                Ok(())
            },
        )?;
    } else {
        exe.dispatch_elements(&bound, &mut backend, &mut s.buffers, &mut s.elements, GROUPS, 32)?;
    }
    // a counter kept in a buffer of its own (the older DXIL lift's, a word per element) is that view's counter, as the
    // backend keeps an IR program's; a counter left where it started is not told from one the run never touched
    let mut counters = backend.counters;
    for r in &program.interface.resources {
        if let Some(view) = r.provenance.counter_of
            && let Some(words) = s.buffers.remove(&r.id)
        {
            for (k, w) in (0..).zip(words) {
                counters.entry((view.0, k)).or_insert(w);
            }
        }
    }
    let zeroed = c == ZEROED;
    counters.retain(|&(view, element), count| *count != counter_start(zeroed, view, element));
    Ok(Memory {
        buffers: s.buffers,
        elements: s.elements,
        texels: backend.texels,
        counters,
        meshlets,
        rays: backend.rays,
    }
    .canonical())
}

/// Whether a run stopped at the executor's step limit ([`iridia_exec::STEP_LIMIT`]).
fn past_budget(run: &Result<Memory, String>) -> bool {
    matches!(run, Err(e) if e.ends_with("execution budget exceeded"))
}

/// A compute or mesh program: A, B and P dispatched on the same seeded memory must leave the same memory, and a mesh
/// program's thread groups the same meshlets.
fn compute(g: &mut Gate, old: &iridia::Program, p: iridia_ir::Program, at: &str, took: &mut Took) {
    let mut mark = Instant::now();
    let a = match Executable::new(old) {
        Ok(a) => a,
        Err(e) => return tally(&mut g.a_fails, &format!("A not built: {e}"), at),
    };
    g.a += 1;
    *if old.interface.stage == Stage::Mesh {
        &mut g.mesh
    } else {
        &mut g.compute
    } += 1;
    took.lift += lap(&mut mark);
    // A's memory after each vector, which B and P are both held to
    let older: Vec<_> = (0..CASES).map(|c| dispatch(&a, old, c)).collect();
    took.a = lap(&mut mark);
    let passed = passed(p.clone());
    for column in ["B", "P"] {
        let ir = if column == "B" {
            Ok(p.clone())
        } else {
            passed.clone()
        };
        let b = match ir.and_then(|p| Executable::from_ir(&p).map_err(|e| e.to_string())) {
            Ok(b) => b,
            Err(e) => return tally(&mut g.unrun, &format!("{column}: {e}"), at),
        };
        let (mut compared, mut skipped) = (0, Vec::new());
        for (c, x) in (0..).zip(&older) {
            if let Err(e) = x {
                skipped.push(e.clone());
                continue;
            }
            let y = dispatch(&b, old, c);
            if past_budget(&y) {
                g.past_budget(column, at);
                continue;
            }
            if *x != y {
                let d = format!("case {c}: A {x:x?}\n  but {y:x?}");
                let d: String = d.chars().take(4000).collect();
                return g.differ.push(format!("{at} ({column})\n  {d}"));
            }
            compared += 1;
        }
        *if column == "B" { &mut took.b } else { &mut took.p } += lap(&mut mark);
        if !g.settle(column, compared, &skipped, at) {
            return;
        }
    }
    let Ok(passed) = passed else { return };
    let c = g.lowered(old, &passed, at).map(|c| Executable::new(&c));
    match c {
        None => {}
        Some(Err(e)) => tally(&mut g.c_unbuilt, &format!("not compiled: {e}"), at),
        Some(Ok(c)) => {
            let differs = (0..).zip(&older).find_map(|(n, x)| {
                let y = dispatch(&c, old, n);
                if x.is_ok() && past_budget(&y) {
                    g.past_budget("C", at);
                    return None;
                }
                (x.is_ok() && *x != y).then(|| format!("case {n}: A {x:x?}\n  but {y:x?}"))
            });
            match differs {
                None => g.agree_c += 1,
                Some(d) => {
                    let d: String = d.chars().take(4000).collect();
                    g.differ.push(format!("{at} (C)\n  {d}"));
                }
            }
        }
    }
    took.c = lap(&mut mark);
}

/// A lane of seeded words of a varying of `scalar`, as [`common::case_with`] seeds an input's: a float in about
/// [-0.25, 1.75), a small integer.
fn seeded_lane(scalar: iridia::ScalarType, h: u32) -> u32 {
    match scalar {
        iridia::ScalarType::Float => (common::unit(h) * 2.0 - 0.25).to_bits(),
        iridia::ScalarType::Uint => h % 4,
        iridia::ScalarType::Sint => (h % 5).wrapping_sub(2),
        _ => h & 1,
    }
}

/// The registers stage function `sig` of `program` takes on case `c`: each point of an array argument and each
/// varying seeded by its scalar type, and the values the pipeline gives within what the stage's facts allow (an
/// output point the hull program has, a domain location inside the patch, an instance the geometry program runs).
fn arguments(program: &iridia::Program, sig: &FunctionSig, c: u32, pointers: &[Vec<u32>]) -> Vec<[u32; 4]> {
    let interface = &program.interface;
    let patch: &[iridia::Varying] = match &interface.facts {
        StageFacts::Hull(h) => &h.patch_constants,
        StageFacts::Domain(d) => &d.patch_constants,
        _ => &[],
    };
    let scalar = |list: ItemList, v: Option<usize>| {
        let varyings = match list {
            ItemList::Inputs => &interface.inputs[..],
            ItemList::Outputs => &interface.outputs[..],
            ItemList::PatchConstants => patch,
            ItemList::Pipeline => &[],
        };
        v.and_then(|v| varyings.get(v))
            .map_or(iridia::ScalarType::Uint, |v| v.value.scalar)
    };
    let seeded = |k: usize, point: u32, s: iridia::ScalarType| -> [u32; 4] {
        std::array::from_fn(|lane| {
            seeded_lane(s, common::mix(hash_words(&[9, c, k as u32, point, lane as u32])))
        })
    };
    let mut registers = Vec::new();
    let mut pointer = pointers.iter();
    for (k, item) in sig.arguments.iter().enumerate() {
        let h = common::mix(hash_words(&[10, c, k as u32]));
        match (item.ty, item.name.as_str()) {
            // a pointer argument's words, four to a register
            (IoType::Memory, _) => {
                let words = pointer.next().map_or(&[][..], Vec::as_slice);
                registers.extend(
                    words
                        .chunks(4)
                        .map(|w| std::array::from_fn(|i| w.get(i).copied().unwrap_or(0))),
                );
            }
            (IoType::Array(_, n), _) => {
                let s = scalar(item.list, item.varying);
                registers.extend((0..n).map(|point| seeded(k, point, s)));
            }
            (_, "SV_OutputControlPointID") => {
                let points = match &interface.facts {
                    StageFacts::Hull(h) => h.output_control_points.max(1),
                    _ => 1,
                };
                registers.push([c % points, 0, 0, 0]);
            }
            (_, "SV_PrimitiveID") => registers.push([c, 0, 0, 0]),
            (_, "SV_GSInstanceID") => {
                let instances = match &interface.facts {
                    StageFacts::Geometry(g) => g.instances.max(1),
                    _ => 1,
                };
                registers.push([c % instances, 0, 0, 0]);
            }
            (_, "SV_DomainLocation") => {
                let u = common::unit(h);
                let v = common::unit(common::mix(u64::from(h) ^ 1)) * (1.0 - u);
                let w = match &interface.facts {
                    StageFacts::Domain(d) if d.domain == iridia::TessDomain::Triangle => 1.0 - u - v,
                    _ => 0.0,
                };
                registers.push([u.to_bits(), v.to_bits(), w.to_bits(), 0]);
            }
            _ => registers.push(seeded(k, 0, scalar(item.list, item.varying))),
        }
    }
    registers
}

/// One run of stage function `exe` (of `program`'s interface, as `sig`) on case `c`'s arguments and memory: its
/// result's members (NaNs alike) and the memory it leaves.
/// The scalar type of each 32-bit leaf of the structure each pointer argument of function `name` of A's module points
/// to (a ray tracing program's payload, then its attributes), in order. Both columns pass the structure as a word per
/// leaf, in order: the executor lays A's naga structure out so, and the IR lays it out as Direct3D does, which for
/// leaves of 32 bits (the naga-path lift refuses memory of other widths) is one word after another.
fn pointer_leaves(old: &iridia::Program, name: &str) -> Result<Vec<Vec<iridia::ScalarType>>, String> {
    let module = &old.module;
    let f = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some(name))
        .map(|(_, f)| f)
        .ok_or_else(|| format!("A has no function {name}"))?;
    fn leaves(
        module: &naga::Module,
        ty: naga::Handle<naga::Type>,
        out: &mut Vec<iridia::ScalarType>,
    ) -> Result<(), String> {
        let scalar = |s: naga::Scalar| -> Result<iridia::ScalarType, String> {
            match (s.kind, s.width) {
                (naga::ScalarKind::Float, 4) => Ok(iridia::ScalarType::Float),
                (naga::ScalarKind::Uint, 4) => Ok(iridia::ScalarType::Uint),
                (naga::ScalarKind::Sint, 4) => Ok(iridia::ScalarType::Sint),
                (naga::ScalarKind::Bool, _) => Ok(iridia::ScalarType::Bool),
                _ => Err(format!("a pointer argument's leaf of {s:?}")),
            }
        };
        match &module.types[ty].inner {
            naga::TypeInner::Scalar(s) => out.push(scalar(*s)?),
            naga::TypeInner::Vector { size, scalar: s } => {
                out.extend(std::iter::repeat_n(scalar(*s)?, *size as usize));
            }
            naga::TypeInner::Array {
                base,
                size: naga::ArraySize::Constant(n),
                ..
            } => {
                for _ in 0..n.get() {
                    leaves(module, *base, out)?;
                }
            }
            naga::TypeInner::Struct { members, .. } => {
                for m in members {
                    leaves(module, m.ty, out)?;
                }
            }
            other => return Err(format!("a pointer argument of {other:?}")),
        }
        Ok(())
    }
    let mut all = Vec::new();
    for a in &f.arguments {
        if let naga::TypeInner::Pointer { base, .. } = module.types[a.ty].inner {
            let mut out = Vec::new();
            leaves(module, base, &mut out)?;
            all.push(out);
        }
    }
    Ok(all)
}

/// One run of a stage function's result members, its pointer arguments' words after it (see [`pointer_leaves`]) and
/// the memory it leaves.
type Call = (Vec<[u32; 4]>, Vec<Vec<u32>>, Memory);

fn call(
    exe: &Executable,
    program: &iridia::Program,
    sig: &FunctionSig,
    c: u32,
    leaves: &[Vec<iridia::ScalarType>],
) -> Result<Call, String> {
    let s = seeded(program, c);
    let bound = exe.bind_resources(&s.resources)?;
    let mut backend = Group::for_case(c);
    // each leaf of each pointer argument seeded by its scalar type, a word each
    let mut pointers = Vec::with_capacity(leaves.len());
    for (m, of) in leaves.iter().enumerate() {
        let n = exe
            .pointer_words(m)
            .ok_or_else(|| format!("no pointer argument {m}"))? as usize;
        if n != of.len() {
            return Err(format!(
                "pointer argument {m} of {n} words, A's structure of {} leaves",
                of.len()
            ));
        }
        let words = (0u32..)
            .zip(of)
            .map(|(i, &scalar)| seeded_lane(scalar, common::mix(hash_words(&[12, c, m as u32, i]))))
            .collect();
        pointers.push(words);
    }
    let registers = arguments(program, sig, c, &pointers);
    let (values, pointees) = {
        let mut runner = Runner::<1>::new(exe, &bound, &mut backend, None);
        let [out] = runner.run([&registers])?;
        (out.values, out.pointees)
    };
    let after = pointees
        .into_iter()
        .map(|words| words.into_iter().map(canonical).collect())
        .collect();
    Ok((
        values.into_iter().map(|v| v.map(canonical)).collect(),
        after,
        Memory {
            buffers: s.buffers,
            elements: s.elements,
            texels: backend.texels,
            counters: backend.counters,
            meshlets: Vec::new(),
            rays: backend.rays,
        }
        .canonical(),
    ))
}

/// A program made of functions: each function of A, B and P run on the same seeded arguments and memory must give
/// the same members and leave the same memory.
fn functions(
    g: &mut Gate,
    old: &iridia::Program,
    sigs: &[FunctionSig],
    p: iridia_ir::Program,
    at: &str,
    took: &mut Took,
) {
    let mut mark = Instant::now();
    // a program made of functions has no WGSL entry point: C counts the lowering's refusal, or fails one it takes
    if let Ok(passed) = passed(p.clone())
        && g.lowered(old, &passed, at).is_some()
    {
        tally(
            &mut g.c_unbuilt,
            "a program made of functions lowered as an entry point",
            at,
        );
    }
    let mut a = Vec::with_capacity(sigs.len());
    let mut leaves = Vec::with_capacity(sigs.len());
    for sig in sigs {
        match Executable::function(old, sig.name) {
            Ok(e) => a.push(e),
            Err(e) => return tally(&mut g.a_fails, &format!("A not built: {e}"), at),
        }
        match pointer_leaves(old, sig.name) {
            Ok(l) => leaves.push(l),
            Err(e) => return tally(&mut g.a_fails, &format!("A's pointer arguments: {e}"), at),
        }
    }
    g.a += 1;
    g.functions += 1;
    took.lift += lap(&mut mark);
    // A's result of each function on each vector, which B and P are both held to
    let older: Vec<Vec<_>> = sigs
        .iter()
        .zip(&a)
        .zip(&leaves)
        .map(|((sig, a), leaves)| (0..CASES).map(|c| call(a, old, sig, c, leaves)).collect())
        .collect();
    took.a = lap(&mut mark);
    for column in ["B", "P"] {
        let ir = if column == "B" {
            Ok(p.clone())
        } else {
            passed(p.clone())
        };
        let ir = match ir {
            Ok(ir) => ir,
            Err(e) => return tally(&mut g.unrun, &format!("{column}: {e}"), at),
        };
        // each function is compared on a vector at least, or A fails the program
        let mut skipped = Vec::new();
        for ((sig, runs), leaves) in sigs.iter().zip(&older).zip(&leaves) {
            let b = match Executable::function_from_ir(&ir, sig.name) {
                Ok(b) => b,
                Err(e) => return tally(&mut g.unrun, &format!("{column} {}: {e}", sig.name), at),
            };
            let mut compared = 0;
            for (c, x) in (0..).zip(runs) {
                if let Err(e) = x {
                    skipped.push(format!("{}: {e}", sig.name));
                    continue;
                }
                let y = call(&b, old, sig, c, leaves);
                if *x != y {
                    let d = format!("{} case {c}: A {x:x?}\n  but {y:x?}", sig.name);
                    let d: String = d.chars().take(4000).collect();
                    return g.differ.push(format!("{at} ({column})\n  {d}"));
                }
                compared += 1;
            }
            if compared == 0 {
                return tally(
                    &mut g.a_fails,
                    skipped.last().map_or("no vector", String::as_str),
                    at,
                );
            }
        }
        *if column == "B" { &mut took.b } else { &mut took.p } += lap(&mut mark);
        g.settle(column, 1, &skipped, at);
    }
}

/// Every DXBC fixture of the crate's tests (no game data needed).
#[test]
fn fixture_programs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut fixtures = common::folder(&root);
    fixtures.name = "fixtures".into();
    assert!(fixtures.names.len() >= 20, "{} fixtures", fixtures.names.len());
    gate_over(&fixtures, &[]);
}

/// What `b` does on the gate's vectors with coherent quad inputs (a vector it cannot run does nothing, as [`compare`]
/// skips it).
#[derive(Clone, Copy)]
struct CoherentRuns {
    /// Whether it takes a screen derivative on any.
    derivatives: bool,
    /// Whether a lane of any takes a derivative and is kept (not discarded).
    kept: bool,
}

fn coherent_runs(b: &ProgramShader) -> CoherentRuns {
    let mut runs = CoherentRuns {
        derivatives: false,
        kept: false,
    };
    for c in 0..CASES {
        let case = gate_case(b, c, true, QuadInputs::Coherent);
        let buffers = common::refs(&case[0], b);
        let mut sampler = Hashed::for_shader(b, c);
        if let Ok(quad) = b.execute_quad(
            [&case[0].inputs, &case[1].inputs, &case[2].inputs, &case[3].inputs],
            &buffers,
            &mut sampler,
        ) {
            runs.derivatives |= quad.iter().any(|e| e.derivatives > 0);
            runs.kept |= quad.iter().any(|e| e.derivatives > 0 && !e.discarded);
        }
    }
    runs
}

#[test]
fn sekiro_programs() {
    gate(Game::Sekiro);
}
#[test]
fn riders_republic_programs() {
    gate(Game::Riders);
}
#[test]
fn unity_programs() {
    gate(Game::Sledding);
}

/// What the DXIL IR lifter does not lift yet, which the older lift does: a refusal naming one of these is not a
/// regression (each piece of the lifter that lands takes its words out).
const DXIL_NOT_YET: &[&str] = &["onto the IR yet"];

/// The DXIL fixtures of `iridia-dxil` (`dxc` output from the HLSL beside each).
#[test]
fn dxil_fixture_programs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../iridia-dxil/tests/fixtures");
    let mut fixtures = common::folder(&root);
    fixtures.name = "dxil fixtures".into();
    fixtures.retain(|b| has_part(b, b"DXIL"));
    assert!(
        fixtures.names.len() >= 50,
        "{} DXIL fixtures",
        fixtures.names.len()
    );
    gate_over(&fixtures, DXIL_NOT_YET);
}

/// The slim set's DXIL programs (`oracles.corpus.general/slim`, `programs/dxil`).
#[test]
fn slim_dxil_programs() {
    let query = inputs::Query::Oracle(inputs::OracleSet::Corpus, Game::General, "slim");
    let Some(tree) = inputs::oracle_tree(inputs::OracleSet::Corpus, Game::General, "slim") else {
        return;
    };
    let programs = common::Corpus::matching("slim dxil".into(), tree, |n| {
        n.strip_prefix("programs/dxil/")
            .is_some_and(|f| !f.contains('/') && f.ends_with(".dxil"))
    });
    inputs::expect_found(query, "DXIL programs", programs.names.len());
    gate_over(&programs, DXIL_NOT_YET);
}

/// Elden Ring's DXIL containers (`oracles.corpus.elden/shaders`).
#[test]
fn elden_programs() {
    let Some(corpus) = common::corpus(Game::Elden) else {
        return;
    };
    gate_over(&corpus, DXIL_NOT_YET);
}

/// The gate over a folder of containers outside the configured corpora, named by the debug key `shader.outside` and
/// walked as the fixtures are; without it, nothing to run.
#[test]
fn outside_programs() {
    let Some(root) = inputs::debug("shader.outside").map(std::path::PathBuf::from) else {
        return;
    };
    gate_over(&common::folder(&root), &[]);
}

/// One program's lifts and runs, for a disagreement the gate prints: `IR_GATE_ONE=<container> cargo test -p
/// iridia-dxbc --test ir_gate one -- --ignored --nocapture`. A and B as text (not with `IR_GATE_QUIET`), then each
/// vector's runs of A, B and C, the first request C makes otherwise than A, and C on a quad of the first lane's inputs;
/// `IR_GATE_C_WGSL=<file>` runs that WGSL as C instead of the lowering's (to try a change to it); `IR_GATE_P=1` prints
/// P as text and the lowering's WGSL.
#[test]
#[ignore = "a diagnostic: names its program in IR_GATE_ONE"]
fn one() {
    let path = std::env::var("IR_GATE_ONE").expect("IR_GATE_ONE names a container");
    let bytes = std::fs::read(&path).unwrap();
    let (old, p) = lift_both(&bytes);
    let old = old.unwrap();
    let quiet = std::env::var_os("IR_GATE_QUIET").is_some();
    if !quiet {
        println!(
            "A (old lift) as WGSL:\n{}",
            iridia::emit::wgsl(&old).unwrap_or_else(|e| e.to_string())
        );
    }
    let p = p.unwrap();
    if !quiet {
        println!("B (IR lift):\n{}", iridia_ir::text::print(&p.entry));
    }
    let p = structured(p).unwrap();
    if std::env::var_os("IR_GATE_P").is_some() {
        match passed(p.clone()) {
            Ok(pp) => {
                println!(
                    "P (after the passes):
{}",
                    iridia_ir::text::print(&pp.entry)
                );
                match iridia_naga::wgsl(&pp, &iridia_naga::Profile::NAGA) {
                    Ok(w) => println!(
                        "C's WGSL:
{w}"
                    ),
                    Err(e) => println!("C's WGSL: {}", e.0),
                }
            }
            Err(e) => println!("P: {e}"),
        }
    }
    // C, where the lowering takes the program
    let c_program = passed(p.clone()).ok().and_then(|pp| match lower_c(&old, &pp) {
        Ok(c) => Some(c),
        Err(Unlowered::Refused(e) | Unlowered::Unbuilt(e)) => {
            println!("C: {e}");
            None
        }
    });
    if old.interface.stage == Stage::Compute {
        return one_compute(&old, &p, c_program.as_ref());
    }
    let a = ProgramShader::new(old).unwrap();
    let b = ProgramShader::from_ir(p).unwrap();
    let c_shader = c_program.map(|c| ProgramShader::new(c).unwrap());
    for c in 0..CASES {
        let quad = a.needs_quad() || b.needs_quad();
        let case = gate_case(&a, c, quad, QuadInputs::Own);
        println!(
            "case {c}: inputs {:x?}",
            case.iter().map(|q| &q.inputs).collect::<Vec<_>>()
        );
        let (mut sa, mut sb) = (Hashed::for_shader(&a, c), Hashed::for_shader(&b, c));
        let (x, y) = (run_asking(&a, &case, &mut sa), run_asking(&b, &case, &mut sb));
        println!("  A {x:x?}\n  B {y:x?}");
        if let Some(cs) = &c_shader {
            let mut sc = Hashed::for_shader(cs, c);
            let z = run_asking(cs, &case, &mut sc);
            println!("  C {z:x?}");
            for (k, (p, q)) in sa.asked.iter().zip(&sc.asked).enumerate() {
                if p != q {
                    println!("  request {k} differs:\n    A {p}\n    C {q}");
                    break;
                }
            }
            if sa.asked.len() != sc.asked.len() {
                println!("  A made {} requests, C {}", sa.asked.len(), sc.asked.len());
            }
            // the quad with every lane the first lane's inputs (no lane leaves a loop before the others)
            if quad {
                let first = gate_case(&a, c, quad, QuadInputs::First);
                let (mut ta, mut tc) = (Hashed::for_shader(&a, c), Hashed::for_shader(cs, c));
                let (xa, xc) = (run_asking(&a, &first, &mut ta), run_asking(cs, &first, &mut tc));
                println!(
                    "  first lane's inputs everywhere: A and C agree: {}",
                    same_run(&xa, &xc, false)
                );
            }
        }
        if !quiet {
            for (k, (p, q)) in sa.asked.iter().zip(&sb.asked).enumerate() {
                if p != q {
                    println!("  request {k} differs:\n    A {p}\n    B {q}");
                }
            }
            if c == 0 || x != y {
                println!("  A asked {:#?}\n  B asked {:#?}", sa.asked, sb.asked);
            }
        }
        if sa.asked.len() != sb.asked.len() {
            println!("  A made {} requests, B {}", sa.asked.len(), sb.asked.len());
        }
    }
}

/// [`one`] for a compute program: each case's memory after A, B and C, and the first words where B or C leaves
/// other memory than A.
fn one_compute(old: &iridia::Program, p: &iridia_ir::Program, c: Option<&iridia::Program>) {
    let a = Executable::new(old).unwrap();
    let b = Executable::from_ir(p).unwrap();
    let c = c.map(|c| (Executable::new(c).unwrap(), c));
    for case in 0..CASES {
        let x = dispatch(&a, old, case);
        let y = dispatch(&b, old, case);
        let z = c.as_ref().map(|(c, program)| dispatch(c, program, case));
        println!("case {case}:");
        let Ok(x) = x else {
            println!("  A {x:?}");
            continue;
        };
        for (column, y) in [("B", Some(y)), ("C", z)] {
            match y {
                None => {}
                Some(Err(e)) => println!("  {column} {e}"),
                Some(Ok(y)) => memory_differences(column, &x, &y),
            }
        }
    }
}

/// Up to a few of the places where `y` leaves other memory than `x`.
fn memory_differences(column: &str, x: &Memory, y: &Memory) {
    let mut lines = Vec::new();
    for (id, words) in &x.buffers {
        match y.buffers.get(id) {
            None => lines.push(format!("buffer {id:?}: missing")),
            Some(other) => {
                for (k, (p, q)) in words.iter().zip(other).enumerate() {
                    if p != q {
                        lines.push(format!("buffer {id:?} word {k}: A {p:x} {column} {q:x}"));
                    }
                }
            }
        }
    }
    for (key, words) in &x.elements {
        if y.elements.get(key) != Some(words) {
            lines.push(format!(
                "element {key:?}: A {words:x?} {column} {:x?}",
                y.elements.get(key)
            ));
        }
    }
    for (key, texel) in &x.texels {
        if y.texels.get(key) != Some(texel) {
            lines.push(format!(
                "texel {key:?}: A {texel:x?} {column} {:x?}",
                y.texels.get(key)
            ));
        }
    }
    for key in y.texels.keys().filter(|k| !x.texels.contains_key(k)) {
        lines.push(format!("texel {key:?}: only {column} wrote it"));
    }
    if x.counters != y.counters {
        lines.push(format!(
            "counters: A {:x?} {column} {:x?}",
            x.counters, y.counters
        ));
    }
    let total = lines.len();
    for l in lines.iter().take(12) {
        println!("  {l}");
    }
    if total > 12 {
        println!("  ... {total} differences in all");
    }
    if total == 0 {
        println!("  {column} agrees");
    }
}

/// Properties of the gate's own parts, on generated runs: what its comparison flags and what it leaves, and the
/// derivatives its coherent quads give.
mod properties {
    use super::*;
    use iridia_exec::Execution;
    use proptest::prelude::*;

    fn word() -> impl Strategy<Value = Word> {
        prop_oneof![
            (0..4u32, 0..16u32).prop_map(|(view, word)| Word::Buffer { view, word }),
            (0..4u32, 0..4u32, 0..4u32).prop_map(|(view, element, word)| Word::Element {
                view,
                element,
                word,
                stride: 16
            }),
            (0..4u32, 0..4u32, 0..4u32).prop_map(|(view, x, lane)| Word::Texel {
                view,
                dim: 2,
                texel: [x, 0, 0],
                layer: None,
                lane
            }),
        ]
    }

    /// Words written, each one to four times, with every NaN as one pattern (as a run leaves them).
    fn written() -> impl Strategy<Value = Written> {
        prop::collection::btree_map(word(), prop::collection::vec(any::<u32>(), 1..4), 0..6)
            .prop_map(|writes| Written { writes }.canonical())
    }

    /// One to three output registers, each with a mask, of an execution kept or discarded.
    fn execution() -> impl Strategy<Value = Execution> {
        (
            prop::collection::vec((any::<[u32; 4]>(), 0..16u8), 1..4),
            any::<bool>(),
        )
            .prop_map(|(registers, discarded)| Execution {
                outputs: registers.iter().map(|r| r.0).collect(),
                output_masks: registers.iter().map(|r| r.1).collect(),
                discarded,
                ..Execution::default()
            })
    }

    /// One invocation's execution or a quad's four.
    fn executions() -> impl Strategy<Value = Vec<Execution>> {
        prop_oneof![
            prop::collection::vec(execution(), 1),
            prop::collection::vec(execution(), 4)
        ]
    }

    /// Read requests as [`Hashed`] keeps them: a tag (1 to 3 a texture request, 4 to 6 another) and its words.
    fn reads() -> impl Strategy<Value = Vec<Vec<u32>>> {
        prop::collection::vec(
            (1..7u32, prop::collection::vec(0..4u32, 0..3)).prop_map(|(tag, mut words)| {
                words.insert(0, tag);
                words
            }),
            0..6,
        )
    }

    fn run_of(executions: &[Execution], written: Written, reads: Vec<Vec<u32>>) -> Run {
        Ok((executions.iter().map(invocation).collect(), written, reads))
    }

    /// A NaN whose payload `bits` chooses, and another of the opposite sign: one NaN for another.
    fn nans(bits: u32) -> (u32, u32) {
        (
            0x7f80_0001 | (bits & 0x003f_ffff),
            0xffc0_0000 | (bits >> 10 & 0x003f_ffff),
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// A run agrees with itself, with and without its texture requests compared.
        #[test]
        fn a_run_agrees_with_itself(e in executions(), w in written(), r in reads(), textures in any::<bool>()) {
            let (x, y) = (run_of(&e, w.clone(), r.clone()), run_of(&e, w, r));
            prop_assert!(same_run(&x, &y, textures));
        }

        /// A written lane of a kept invocation that gives other bits (not one NaN for another) is a difference, in a
        /// single invocation and in any lane of a quad.
        #[test]
        fn a_changed_output_lane_is_flagged(
            mut e in executions(),
            pick in any::<(prop::sample::Index, prop::sample::Index, prop::sample::Index)>(),
            bit in 0..32u32,
        ) {
            let i = pick.0.index(e.len());
            let r = pick.1.index(e[i].outputs.len());
            let k = pick.2.index(4);
            e[i].output_masks[r] |= 1 << k;
            e[i].discarded = false;
            let mut f = e.clone();
            f[i].outputs[r][k] ^= 1 << bit;
            prop_assume!(canonical(e[i].outputs[r][k]) != canonical(f[i].outputs[r][k]));
            let (x, y) = (run_of(&e, Written::default(), vec![]), run_of(&f, Written::default(), vec![]));
            prop_assert!(!same_run(&x, &y, true));
        }

        /// What reaches no target is not compared: a lane the invocation did not write, any lane of a discarded
        /// invocation, a NaN given for another.
        #[test]
        fn what_no_target_takes_is_not_compared(
            mut e in executions(),
            pick in any::<(prop::sample::Index, prop::sample::Index, prop::sample::Index)>(),
            how in 0..3u8,
            word in any::<u32>(),
        ) {
            let i = pick.0.index(e.len());
            let r = pick.1.index(e[i].outputs.len());
            let k = pick.2.index(4);
            let mut f = e.clone();
            match how {
                0 => {
                    e[i].output_masks[r] &= !(1 << k);
                    f[i].output_masks[r] = e[i].output_masks[r];
                    f[i].outputs[r][k] = word;
                }
                1 => {
                    e[i].discarded = true;
                    f[i].discarded = true;
                    f[i].outputs[r][k] = word;
                }
                _ => {
                    let (one, another) = nans(word);
                    e[i].outputs[r][k] = one;
                    f[i].outputs[r][k] = another;
                }
            }
            let (x, y) = (run_of(&e, Written::default(), vec![]), run_of(&f, Written::default(), vec![]));
            prop_assert!(same_run(&x, &y, true));
        }

        /// Memory that ends otherwise is a difference: a word given another value (not one NaN for another), a write
        /// more, a word more.
        #[test]
        fn changed_memory_is_flagged(
            w in written(),
            at in word(),
            pick in any::<(prop::sample::Index, prop::sample::Index)>(),
            how in 0..3u8,
            value in any::<u32>(),
        ) {
            let mut v = w.clone();
            let key = (!w.writes.is_empty()).then(|| *w.writes.keys().nth(pick.0.index(w.writes.len())).unwrap());
            match (how, key) {
                (0, Some(key)) => {
                    let seq = v.writes.get_mut(&key).unwrap();
                    let j = pick.1.index(seq.len());
                    prop_assume!(canonical(seq[j]) != canonical(value));
                    seq[j] = value;
                }
                (1, Some(key)) => v.write(key, value),
                _ => {
                    prop_assume!(!w.writes.contains_key(&at));
                    v.write(at, value);
                }
            }
            let (x, y) = (run_of(&[], w, vec![]), run_of(&[], v.canonical(), vec![]));
            prop_assert!(!same_run(&x, &y, false));
        }

        /// One NaN written for another is the same memory.
        #[test]
        fn a_nan_written_for_another_is_the_same_memory(at in word(), bits in any::<u32>()) {
            let (one, another) = nans(bits);
            let (mut v, mut w) = (Written::default(), Written::default());
            v.write(at, one);
            w.write(at, another);
            let (x, y) = (run_of(&[], v.canonical(), vec![]), run_of(&[], w.canonical(), vec![]));
            prop_assert!(same_run(&x, &y, false));
        }

        /// B's texture requests must be among A's: any subset of them (with any other reads) passes, one A does not
        /// make is a difference; without textures compared, neither matters.
        #[test]
        fn texture_requests_must_be_among_the_older_lifts(
            r in reads(),
            keep in prop::collection::vec(any::<bool>(), 6),
            other in reads(),
            extra in (1..4u32, prop::collection::vec(0..4u32, 0..3)),
        ) {
            let x = run_of(&[], Written::default(), r.clone());
            let mut subset: Vec<Vec<u32>> =
                r.iter().zip(&keep).filter(|(_, k)| **k).map(|(q, _)| q.clone()).collect();
            subset.extend(other.into_iter().filter(|q| !matches!(q.first(), Some(1..=3))));
            prop_assert!(same_run(&x, &run_of(&[], Written::default(), subset.clone()), true));
            let mut request = extra.1;
            request.insert(0, extra.0);
            prop_assume!(!r.contains(&request));
            subset.push(request);
            let y = run_of(&[], Written::default(), subset);
            prop_assert!(!same_run(&x, &y, true));
            prop_assert!(same_run(&x, &y, false));
        }

        /// A compute or function run's memory compares every word, each NaN as one: a word changed otherwise is a
        /// difference, one NaN for another is not.
        #[test]
        fn compute_memory_compares_every_word(
            words in prop::collection::vec(any::<u32>(), 1..16),
            pick in any::<prop::sample::Index>(),
            value in any::<u32>(),
            nan in any::<bool>(),
        ) {
            let memory = |words: Vec<u32>| Memory {
                buffers: BTreeMap::from([(ResourceId(1), words.clone())]),
                elements: BTreeMap::from([((ResourceId(2), 0), words)]),
                texels: Texels::new(),
                counters: BTreeMap::new(),
                meshlets: Vec::new(),
                rays: Vec::new(),
            }
            .canonical();
            let j = pick.index(words.len());
            if nan {
                let (one, another) = nans(value);
                let (mut was, mut now) = (words.clone(), words);
                was[j] = one;
                now[j] = another;
                prop_assert_eq!(memory(was), memory(now));
            } else {
                prop_assume!(canonical(words[j]) != canonical(value));
                let mut now = words.clone();
                now[j] = value;
                prop_assert_ne!(memory(words), memory(now));
            }
        }

        /// Each interpolated input lane of a coherent quad, from any first-lane value the gate seeds, steps by
        /// amounts of different size along x and y, neither zero, in the coarse derivatives and the fine ones: a
        /// derivative taken along the wrong axis gives another value.
        #[test]
        fn coherent_quads_step_differently_along_x_and_y(base in -0.25f32..1.75) {
            for c in 0..CASES {
                for register in 0..32 {
                    for lane in 0..4 {
                        let q = coherent_quad(base, c, register, lane);
                        for (x, y) in [(q[1] - q[0], q[2] - q[0]), (q[3] - q[2], q[3] - q[1])] {
                            prop_assert!(
                                x != 0.0 && y != 0.0 && x.abs() != y.abs(),
                                "vector {c}, v{register} lane {lane}, first lane {base}: steps {x} and {y}"
                            );
                        }
                    }
                }
            }
        }
    }
}
