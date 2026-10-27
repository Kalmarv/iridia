use std::collections::BTreeMap;

use super::flat::{
    AtomOp, BinOp, CvtOp, Executable, GlobalKind, LevelOp, Loc, MathOp, Member, Op, Pick, Ptr, Reg, Res, St,
    UnOp, WaveGather, WaveKind, WaveOp, WaveScan,
};
use super::mask::{Mask, Wide};
use super::rayquery::{Built, Query, Reported, Scenes};
use super::request::{
    Backend, Element, ImageAtomic, ImageLoad, ImageLod, ImageQuery, ImageSample, ImageStore,
};
use iridia::ResourceId;

/// The most operations and loop turns one invocation may execute: a program that does not end is refused.
pub const STEP_LIMIT: u64 = 1 << 24;

/// Where a global's words are during a run.
#[derive(Debug)]
enum Words<'a> {
    Owned(Vec<u32>),
    /// A constant buffer as the caller holds it: rows of four lanes.
    Rows(&'a [[u32; 4]]),
    Slice(&'a [u32]),
    /// The elements of an array of buffers; one the caller gave none of is `None`.
    Elements(Vec<Option<&'a [u32]>>),
    /// A buffer the backend answers reads of.
    Backend,
}

/// The buffers a program reads, ready for runs: built once per program and reused for every pixel.
#[derive(Debug)]
pub struct Bound<'a> {
    data: Vec<Words<'a>>,
    scenes: Scenes,
}

impl Bound<'_> {
    /// Give the acceleration structure of resource `id`: what the program's ray queries search.
    pub fn with_scene(mut self, id: ResourceId, scene: std::sync::Arc<Built>) -> Self {
        self.scenes.insert(id, scene);
        self
    }
}

/// What a run reads besides its inputs: constant buffers (rows of four lanes) and buffers (32-bit words), by resource.
#[derive(Clone, Debug, Default)]
pub struct Resources {
    /// Constant buffers by interface resource: rows of four 32-bit words.
    pub constants: BTreeMap<ResourceId, Vec<[u32; 4]>>,
    /// Buffers by interface resource: 32-bit words.
    pub buffers: BTreeMap<ResourceId, Vec<u32>>,
    /// The elements of arrays of buffers, by resource and element.
    pub elements: BTreeMap<(ResourceId, u32), Vec<u32>>,
    /// The acceleration structures, by resource.
    pub scenes: BTreeMap<ResourceId, super::rayquery::Scene>,
}

impl Executable {
    /// Bind the buffers the program reads, as `constants` and `buffers` find them by resource. A constant buffer the
    /// program reads and the caller does not give is an error; a buffer not given is one the backend answers reads of
    /// ([`Backend::read_buffer`]). Nothing is copied: the caller's buffers are read in place.
    pub fn bind<'a>(
        &self,
        constants: impl Fn(ResourceId) -> Option<&'a [[u32; 4]]>,
        buffers: impl Fn(ResourceId) -> Option<&'a [u32]>,
    ) -> Result<Bound<'a>, String> {
        self.bind_elements(constants, buffers, |_, _| None)
    }

    /// [`Executable::bind`] for a program that also reads arrays of buffers: `elements` finds the words of element
    /// `i` of the array resource `id`. An element not given errors when a run reads it.
    pub fn bind_elements<'a>(
        &self,
        constants: impl Fn(ResourceId) -> Option<&'a [[u32; 4]]>,
        buffers: impl Fn(ResourceId) -> Option<&'a [u32]>,
        elements: impl Fn(ResourceId, u32) -> Option<&'a [u32]>,
    ) -> Result<Bound<'a>, String> {
        let mut data = Vec::with_capacity(self.globals.len());
        for g in &self.globals {
            data.push(match &g.kind {
                GlobalKind::Constants(id) => Words::Rows(
                    constants(*id).ok_or_else(|| format!("no constant buffer for resource {}", id.0))?,
                ),
                GlobalKind::Private(words) => Words::Owned(words.clone()),
                GlobalKind::Storage(id) => buffers(*id).map_or(Words::Backend, Words::Slice),
                GlobalKind::StorageArray(id, count) => {
                    Words::Elements((0..*count).map(|k| elements(*id, k)).collect())
                }
                // thread-group memory is the run's own
                GlobalKind::Workgroup(_) => Words::Owned(Vec::new()),
            });
        }
        Ok(Bound {
            data,
            scenes: BTreeMap::new(),
        })
    }

    /// [`Executable::bind`] over a [`Resources`].
    pub fn bind_resources<'a>(&self, resources: &'a Resources) -> Result<Bound<'a>, String> {
        let mut bound = self.bind_elements(
            |id| resources.constants.get(&id).map(Vec::as_slice),
            |id| resources.buffers.get(&id).map(Vec::as_slice),
            |id, k| resources.elements.get(&(id, k)).map(Vec::as_slice),
        )?;
        for (id, scene) in &resources.scenes {
            bound.scenes.insert(*id, std::sync::Arc::new(Built::new(scene)?));
        }
        Ok(bound)
    }
}

/// What a ray tracing program decided about the hit it was offered, besides its payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Decision {
    /// Nothing: the hit stands.
    #[default]
    None,
    /// `IgnoreHit`: an any-hit program rejects the candidate.
    Ignore,
    /// `AcceptHitAndEndSearch`: an any-hit program accepts it and ends the search.
    AcceptAndEnd,
}

/// What one invocation produced.
#[derive(Clone, Debug)]
pub struct Output {
    /// The words of each pointer argument (a ray tracing program's payload, attributes or parameter) as the program
    /// left them, in argument order.
    pub pointees: Vec<Vec<u32>>,
    /// What the program decided about the hit it was offered; `Decision::None` for any other program.
    pub decision: Decision,
    /// One entry per member of the result, lanes beyond the member's width 0.
    pub values: Vec<[u32; 4]>,
    /// Whether the invocation discarded.
    pub discarded: bool,
    /// Operations and loop turns executed.
    pub steps: u64,
    /// Sample requests this lane made.
    pub samples: u64,
}

/// The registers and local memory of a [`Runner`], kept from one run of a program to the next so that the literals are
/// written once and nothing is allocated per pixel.
#[derive(Debug)]
pub struct Scratch<const N: usize> {
    regs: Vec<Reg>,
    mem: Vec<u32>,
    /// Each lane's result members, their room kept.
    outs: [Vec<Reg>; N],
}
impl<const N: usize> Scratch<N> {
    /// Registers and local memory for `N` lanes of `exe`, with its literals already written.
    pub fn new(exe: &Executable) -> Self {
        let mut regs = vec![[0u32; 4]; exe.slots.saturating_mul(N)];
        for (s, v) in &exe.inits {
            for l in 0..N {
                if let Some(register) = regs.get_mut(slot_at::<N>(*s).saturating_add(l)) {
                    *register = *v;
                }
            }
        }
        Self {
            regs,
            mem: vec![0; exe.local_words.saturating_mul(N)],
            outs: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// Runs an [`Executable`] for a 2 x 2 quad in lockstep (`N` = 4: lanes 0 and 1 the top row, 2 and 3 the bottom), as a
/// GPU does: lanes that diverge are masked, a screen derivative is a difference inside the quad, a lane that is
/// discarded goes on as a helper. Any other `N` runs that many invocations of their own (`N` = 1, one), in lockstep
/// and masked where they diverge, each as it would run alone: its own steps and budget, a discard ending it, the
/// derivatives of its inputs the ones the caller gives for every lane. A runner is reused for any number of runs.
pub struct Runner<'a, const N: usize, M: Mask = u8> {
    c: &'a Executable,
    b: &'a Bound<'a>,
    regs: Vec<Reg>,
    mem: Vec<u32>,
    sampler: &'a mut dyn Backend,
    ftz: bool,
    grouping: Grouping,
    /// The steps of a quad or a thread group, counted once for all its lanes.
    steps: u64,
    /// The steps of each lane of independent invocations.
    lane_steps: [u64; N],
    samples: [u64; N],
    outs: [Vec<Reg>; N],
    /// What an any-hit program decided, per lane.
    decisions: [Decision; N],
    killed: M,
    /// The memory a run changes, by global: thread-group memory and the buffers a compute run writes (empty for a
    /// pixel run, which hands its writes to the backend).
    store: Vec<Option<Vec<u32>>>,
    /// The ray queries of each invocation, by query and lane.
    queries: Vec<Vec<Option<Query>>>,
    /// The same for the elements of arrays of buffers, by global and element.
    elem_store: Vec<Vec<Option<Vec<u32>>>>,
    /// The lanes of a thread group, and those of them that have left the program.
    group: M,
    retired: M,
    /// The lanes in use and the width of a wave of them (0 outside a compute run).
    lanes: usize,
    wave: usize,
    /// The derivatives of the arguments (scalar mode): per argument, d/dx then d/dy lanes.
    derivs: Option<&'a [[Reg; 2]]>,
}

impl<const N: usize, M: Mask> std::fmt::Debug for Runner<'_, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // independent invocations: the most steps a lane has taken
        let steps = match self.grouping {
            Grouping::Independent => self.lane_steps.iter().copied().max().unwrap_or(0),
            Grouping::Quad | Grouping::Group => self.steps,
        };
        f.debug_struct("Runner")
            .field("lanes", &N)
            .field("grouping", &self.grouping)
            .field("steps", &steps)
            .finish_non_exhaustive()
    }
}

/// What the lanes of a run are to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grouping {
    /// Invocations of their own.
    Independent,
    /// A 2 x 2 quad.
    Quad,
    /// A compute thread group.
    Group,
}

#[derive(Clone, Copy)]
struct Exits<M> {
    brk: M,
    cont: M,
    ret: M,
}
impl<M: Mask> Default for Exits<M> {
    fn default() -> Self {
        Self {
            brk: M::none(),
            cont: M::none(),
            ret: M::none(),
        }
    }
}

/// `$body` for each lane `$l` of `$active` (a runner's lanes): without a test per lane when every lane is active.
macro_rules! each {
    ($active:expr, $l:ident, $body:block) => {
        if $active == M::first(N) {
            for $l in 0..N $body
        } else {
            for $l in 0..N {
                if $active.has($l) $body
            }
        }
    };
}

/// A position in one of the tables of the compiled program or its bound data (a global, an operation). The compile pass
/// numbers them, so a miss is an internal error, reported with the position instead of a panic.
trait At<T> {
    fn at(&self, i: usize) -> Result<&T, String>;
}
impl<T> At<T> for [T] {
    fn at(&self, i: usize) -> Result<&T, String> {
        self.get(i)
            .ok_or_else(|| format!("internal error: position {i} of a table of {}", self.len()))
    }
}

/// The word offset `o + extra`, an error where it does not fit.
fn offset(o: u32, extra: u32) -> Result<usize, String> {
    o.checked_add(extra)
        .map(|word| word as usize)
        .ok_or_else(|| format!("word offset {o} + {extra} overflows"))
}

/// The `n` (at most 4) words at `at` of `words`, as the leading words of a register; `what` names `words` in the error.
fn take_words(words: &[u32], at: usize, n: usize, what: &str) -> Result<Reg, String> {
    let from = at
        .checked_add(n)
        .and_then(|end| words.get(at..end))
        .ok_or_else(|| format!("{what} index out of range"))?;
    let mut r = [0u32; 4];
    for (to, word) in r.iter_mut().zip(from) {
        *to = *word;
    }
    Ok(r)
}

/// The first `n` words of the register `v` written at `at` of `words`.
fn put_words(words: &mut [u32], at: usize, n: usize, v: &Reg, what: &str) -> Result<(), String> {
    let to = at
        .checked_add(n)
        .and_then(|end| words.get_mut(at..end))
        .ok_or_else(|| format!("{what} index out of range"))?;
    for (to, word) in to.iter_mut().zip(v) {
        *to = *word;
    }
    Ok(())
}

/// The first `n` words of a register as the coordinates of a texel (the rest 0).
fn texel_of(r: &Reg, n: usize) -> [u32; 3] {
    let [x, y, z, _] = *r;
    let keep = |i: usize, w: u32| if i < n { w } else { 0 };
    [keep(0, x), keep(1, y), keep(2, z)]
}

/// Component `axis` of a 3-vector of an id or a size (0 past its three).
fn comp3(v: [u32; 3], axis: usize) -> u32 {
    v.get(axis).copied().unwrap_or(0)
}

/// How many invocations a thread group of this size has.
fn group_lanes(size: [u32; 3]) -> Result<u32, String> {
    let [x, y, z] = size;
    x.checked_mul(y)
        .and_then(|xy| xy.checked_mul(z))
        .ok_or_else(|| format!("a thread group of {x} by {y} by {z} invocations does not fit in 32 bits"))
}

/// Component `i` of a register; 0 past its four (a scalar operand repeated reads component 0).
#[inline(always)]
fn comp(r: &Reg, i: usize) -> u32 {
    r.get(i).copied().unwrap_or(0)
}

/// HLSL's float to signed integer: saturating, a NaN 0, as the bits of the integer.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)] // the saturating conversion is the operation
fn float_to_int(x: f32) -> u32 {
    (x as i32) as u32
}

/// HLSL's float to unsigned integer: saturating, a NaN 0.
#[inline(always)]
#[allow(clippy::cast_possible_truncation)] // the saturating conversion is the operation
fn float_to_uint(x: f32) -> u32 {
    x as u32
}

/// The register index of the first lane of slot `slot`: the slots outermost and the lanes innermost, as in `Scratch`.
#[inline(always)]
fn slot_at<const N: usize>(slot: u32) -> usize {
    (slot as usize).saturating_mul(N)
}

/// A bug of the compile pass, as an error naming the register: the compile pass numbers a run's registers.
fn no_register(index: usize, len: usize) -> String {
    format!("internal error: register {index}, where the run has {len}")
}

#[inline(always)]
fn fl(x: f32, on: bool) -> f32 {
    if on && x.is_subnormal() {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}
#[inline(always)]
fn f(x: u32) -> f32 {
    f32::from_bits(x)
}

impl<'a, const N: usize, M: Mask> Runner<'a, N, M> {
    /// A runner of `exe` over `bound`, answering requests through `backend`. Invocations of their own that read
    /// derivatives of their inputs take them in `derivatives`, the same for every lane; a quad takes its own.
    pub fn new(
        exe: &'a Executable,
        bound: &'a Bound<'a>,
        backend: &'a mut dyn Backend,
        derivatives: Option<&'a [[[u32; 4]; 2]]>,
    ) -> Self {
        Self::with_scratch(exe, bound, backend, derivatives, Scratch::new(exe))
    }

    /// [`Runner::new`] on registers kept from an earlier runner of the same program ([`Runner::into_scratch`]): the
    /// literals are already in them.
    pub fn with_scratch(
        exe: &'a Executable,
        bound: &'a Bound<'a>,
        backend: &'a mut dyn Backend,
        derivatives: Option<&'a [[[u32; 4]; 2]]>,
        scratch: Scratch<N>,
    ) -> Self {
        let c = exe;
        let (b, sampler, ftz, derivs) = (bound, backend, exe.flush_denormals, derivatives);
        Self {
            c,
            b,
            regs: scratch.regs,
            mem: scratch.mem,
            sampler,
            ftz,
            grouping: if N == 4 {
                Grouping::Quad
            } else {
                Grouping::Independent
            },
            steps: 0,
            lane_steps: [0; N],
            samples: [0; N],
            outs: scratch.outs,
            decisions: [Decision::None; N],
            killed: M::none(),
            store: Vec::new(),
            elem_store: Vec::new(),
            queries: Vec::new(),
            group: M::none(),
            retired: M::none(),
            lanes: 0,
            wave: 0,
            derivs,
        }
    }

    /// Give the registers back, to start the next runner of this program with.
    pub fn into_scratch(self) -> Scratch<N> {
        Scratch {
            regs: self.regs,
            mem: self.mem,
            outs: self.outs,
        }
    }

    /// Register `lane` of the slot whose first lane is `base` (see [`slot_at`]).
    #[inline(always)]
    fn rd(&self, base: usize, lane: usize) -> Result<Reg, String> {
        let index = base.saturating_add(lane);
        self.regs
            .get(index)
            .copied()
            .ok_or_else(|| no_register(index, self.regs.len()))
    }
    /// Set register `lane` of the slot whose first lane is `base` to `value`.
    #[inline(always)]
    fn wr(&mut self, base: usize, lane: usize, value: Reg) -> Result<(), String> {
        let index = base.saturating_add(lane);
        let len = self.regs.len();
        *self.regs.get_mut(index).ok_or_else(|| no_register(index, len))? = value;
        Ok(())
    }

    /// Count `n` steps the lanes of `active` take, refusing a run past its budget: each lane's own for independent
    /// invocations, the quad's or the group's once.
    #[inline(always)]
    fn count(&mut self, active: M, n: u64) -> Result<(), String> {
        let over = if self.grouping == Grouping::Independent {
            let mut over = false;
            for (l, steps) in self.lane_steps.iter_mut().enumerate() {
                if active.has(l) {
                    *steps = steps.saturating_add(n);
                    over |= *steps > STEP_LIMIT;
                }
            }
            over
        } else {
            self.steps = self.steps.saturating_add(n);
            self.steps > STEP_LIMIT
        };
        if over {
            return Err("execution budget exceeded".into());
        }
        Ok(())
    }

    /// Where local word `word` of lane `lane` is: the words outermost and the lanes innermost, as in the registers.
    #[inline(always)]
    fn local(word: usize, lane: usize) -> usize {
        word.saturating_mul(N).saturating_add(lane)
    }
    /// The first local word of `n` from `word`, refusing words past the program's locals.
    #[inline(always)]
    fn locals(&self, word: usize, n: usize) -> Result<usize, String> {
        if word.checked_add(n).is_none_or(|end| end > self.c.local_words) {
            return Err("local index out of range".into());
        }
        Ok(word)
    }
    /// Local word `word` of lane `lane`.
    #[inline(always)]
    fn local_word(&self, word: usize, lane: usize) -> Result<u32, String> {
        let at = Self::local(word, lane);
        self.mem.get(at).copied().ok_or_else(|| {
            format!(
                "internal error: local word {at}, where the run has {}",
                self.mem.len()
            )
        })
    }
    /// Set local word `word` of lane `lane`.
    #[inline(always)]
    fn set_local_word(&mut self, word: usize, lane: usize, value: u32) -> Result<(), String> {
        let at = Self::local(word, lane);
        let len = self.mem.len();
        *self
            .mem
            .get_mut(at)
            .ok_or_else(|| format!("internal error: local word {at}, where the run has {len}"))? = value;
        Ok(())
    }
    /// Lane `lane`'s local words `at..at + n`.
    fn read_locals(&self, lane: usize, at: usize, n: usize) -> Result<Vec<u32>, String> {
        (at..at.saturating_add(n))
            .map(|w| self.local_word(w, lane))
            .collect()
    }
    /// Lane `lane`'s local words from `at` set to `words`.
    fn write_locals(&mut self, lane: usize, at: usize, words: &[u32]) -> Result<(), String> {
        for (k, w) in words.iter().enumerate() {
            self.set_local_word(at.saturating_add(k), lane, *w)?;
        }
        Ok(())
    }

    fn mem_get(&mut self, loc: Loc, lane: usize, extra: u32, n8: u8) -> Result<Reg, String> {
        let n = n8 as usize;
        match loc {
            Loc::Local(o) => {
                let base = self.locals(offset(o, extra)?, n)?;
                let mut r = [0u32; 4];
                for (i, w) in r.iter_mut().take(n).enumerate() {
                    *w = self.local_word(base.saturating_add(i), lane)?;
                }
                Ok(r)
            }
            Loc::Elem(g, o, pick) => {
                let base = offset(o, extra)?;
                let e = self.element_index(g, pick, lane)?;
                let d = self.element_words(g, e)?;
                take_words(d, base, n, "buffer")
            }
            Loc::Global(g, o) => {
                let base = offset(o, extra)?;
                if let Some(Some(d)) = self.store.get(g as usize) {
                    return take_words(d, base, n, "memory");
                }
                let bound = self.b;
                match bound.data.at(g as usize)? {
                    Words::Elements(_) => Err("an array of buffers read without an element".into()),
                    Words::Owned(d) => take_words(d, base, n, "buffer"),
                    Words::Slice(d) => take_words(d, base, n, "buffer"),
                    Words::Rows(rows) => {
                        let mut r = [0u32; 4];
                        if n == 4 && base.is_multiple_of(4) {
                            r = *rows.get(base / 4).ok_or("buffer index out of range")?;
                        } else {
                            for (i, w) in r.iter_mut().take(n).enumerate() {
                                let word = base.saturating_add(i);
                                *w = rows
                                    .get(word / 4)
                                    .and_then(|row| row.get(word % 4))
                                    .copied()
                                    .ok_or("buffer index out of range")?;
                            }
                        }
                        Ok(r)
                    }
                    // a buffer the caller answers for word by word
                    Words::Backend => {
                        let GlobalKind::Storage(view) = self.c.globals.at(g as usize)?.kind else {
                            return Err("a global that is neither bound nor a buffer".into());
                        };
                        let mut r = self.sampler.read_buffer(
                            view,
                            u32::try_from(base).map_err(|_| "buffer index out of range")?,
                            n8,
                        )?;
                        for w in r.iter_mut().skip(n) {
                            *w = 0;
                        }
                        Ok(r)
                    }
                }
            }
        }
    }
    fn mem_put(&mut self, loc: Loc, lane: usize, extra: u32, n: u8, v: Reg) -> Result<(), String> {
        let count = n as usize;
        match loc {
            Loc::Local(o) => {
                let base = self.locals(offset(o, extra)?, count)?;
                for (i, w) in v.iter().take(count).enumerate() {
                    self.set_local_word(base.saturating_add(i), lane, *w)?;
                }
                Ok(())
            }
            Loc::Elem(g, o, pick) => {
                let e = self.element_index(g, pick, lane)?;
                let at = offset(o, extra)?;
                let Some(Some(d)) = self
                    .elem_store
                    .get_mut(g as usize)
                    .and_then(|v| v.get_mut(e as usize))
                else {
                    return Err("a store to an element of an array of buffers the run does not write".into());
                };
                put_words(d, at, count, &v, "memory")
            }
            Loc::Global(g, o) => {
                if let Some(Some(d)) = self.store.get_mut(g as usize) {
                    let at = offset(o, extra)?;
                    return put_words(d, at, count, &v, "memory");
                }
                // a write to a storage buffer is the caller's effect (it reaches none of this evaluation's reads)
                let GlobalKind::Storage(view) = self.c.globals.at(g as usize)?.kind else {
                    return Err("store to a read-only global".into());
                };
                let first = o.checked_add(extra).ok_or("memory index out of range")?;
                self.sampler.store_buffer(view, first, v, crate::lane_mask(n))
            }
        }
    }
    /// The element of an array of buffers a location names in `lane`.
    fn element_index(&self, g: u32, pick: Pick, lane: usize) -> Result<u32, String> {
        let index = match pick {
            Pick::One => 0,
            Pick::Fixed(i) => i,
            Pick::Slot(s) => self.rd(slot_at::<N>(s), lane)?[0],
        };
        match &self.c.globals.at(g as usize)?.kind {
            GlobalKind::StorageArray(id, count) if index >= *count => {
                Err(format!("buffer {}: element {index} of an array of {count}", id.0))
            }
            _ => Ok(index),
        }
    }
    /// The words of element `e` of the array of buffers `g`: those the run writes, else those it was bound with.
    fn element_words(&self, g: u32, e: u32) -> Result<&[u32], String> {
        if let Some(Some(d)) = self.elem_store.get(g as usize).and_then(|v| v.get(e as usize)) {
            return Ok(d);
        }
        match self.b.data.at(g as usize)? {
            Words::Elements(v) => v
                .get(e as usize)
                .copied()
                .flatten()
                .ok_or_else(|| format!("no words given for element {e} of an array of buffers")),
            _ => Err("a global that is not an array of buffers".into()),
        }
    }
    fn dyn_extra(&self, p: &Ptr, lane: usize) -> Result<u32, String> {
        match p.dynamic {
            None => Ok(0),
            Some((s, stride, len)) => {
                let i = self.rd(slot_at::<N>(s), lane)?[0];
                if i >= len {
                    return Err(format!("dynamic index {i} past the end ({len})"));
                }
                i.checked_mul(stride)
                    .ok_or_else(|| format!("dynamic index {i} of a stride of {stride} words overflows"))
            }
        }
    }

    fn ops(&mut self, lo: u32, hi: u32, active: M) -> Result<(), String> {
        self.count(active, u64::from(hi.saturating_sub(lo)))?;
        let c = self.c;
        let ftz = self.ftz;
        for k in lo..hi {
            let (dst, op) = c.ops.at(k as usize)?;
            let dst = slot_at::<N>(*dst);
            match op {
                // a local's words, at offsets the compiler checked: read straight from every lane's memory
                Op::Load {
                    loc: Loc::Local(o),
                    n,
                } => {
                    let (base, n) = (self.locals(*o as usize, *n as usize)?, *n as usize);
                    each!(active, l, {
                        let mut r = [0u32; 4];
                        for (i, w) in r.iter_mut().take(n).enumerate() {
                            *w = self.local_word(base.saturating_add(i), l)?;
                        }
                        self.wr(dst, l, r)?;
                    })
                }
                Op::LoadLanes {
                    loc: Loc::Local(o),
                    offs,
                    n,
                } => {
                    let (o, n) = (*o as usize, *n as usize);
                    for &k in offs.iter().take(n) {
                        self.locals(o.saturating_add(k as usize), 1)?;
                    }
                    each!(active, l, {
                        let mut r = [0u32; 4];
                        for (w, &k) in r.iter_mut().zip(offs.iter().take(n)) {
                            *w = self.local_word(o.saturating_add(k as usize), l)?;
                        }
                        self.wr(dst, l, r)?;
                    })
                }
                Op::Load { loc, n } => each!(active, l, {
                    let loaded = self.mem_get(*loc, l, 0, *n)?;
                    self.wr(dst, l, loaded)?;
                }),
                Op::LoadLanes { loc, offs, n } => each!(active, l, {
                    let mut r = [0u32; 4];
                    for (w, &k) in r.iter_mut().zip(offs.iter().take(*n as usize)) {
                        *w = self.mem_get(*loc, l, u32::from(k), 1)?[0];
                    }
                    self.wr(dst, l, r)?;
                }),
                Op::LoadDyn { ptr, n } => each!(active, l, {
                    let x = self.dyn_extra(ptr, l)?;
                    let loaded = self.mem_get(ptr.loc, l, x, *n)?;
                    self.wr(dst, l, loaded)?;
                }),
                Op::Swizzle { src, pat, n } => {
                    let src = slot_at::<N>(*src);
                    each!(active, l, {
                        let s = self.rd(src, l)?;
                        let mut r = [0u32; 4];
                        for (w, &p) in r.iter_mut().zip(pat.iter()).take(*n as usize) {
                            *w = comp(&s, p as usize);
                        }
                        self.wr(dst, l, r)?;
                    })
                }
                Op::Lane { src, i } => {
                    let (src, i) = (slot_at::<N>(*src), *i as usize);
                    each!(active, l, {
                        let s = self.rd(src, l)?;
                        self.wr(dst, l, [comp(&s, i); 4])?;
                    })
                }
                Op::Splat { src, n } => {
                    let _ = n;
                    let src = slot_at::<N>(*src);
                    each!(active, l, {
                        let s = self.rd(src, l)?;
                        self.wr(dst, l, [comp(&s, 0); 4])?;
                    })
                }
                Op::Compose { parts, np } => each!(active, l, {
                    let mut r = [0u32; 4];
                    let mut at = 0usize;
                    for p in parts.iter().take(*np as usize) {
                        let s = self.rd(slot_at::<N>(p.0), l)?;
                        for w in s.iter().take(p.1 as usize) {
                            if let Some(slot) = r.get_mut(at) {
                                *slot = *w;
                            }
                            at = at.saturating_add(1);
                        }
                    }
                    self.wr(dst, l, r)?;
                }),
                Op::Un { op, src, n } => {
                    let (src, n) = (slot_at::<N>(*src), *n as usize);
                    // each operation's own loop over the lanes
                    macro_rules! un {
                        ($w:ident => $e:expr) => {
                            each!(active, l, {
                                let s = self.rd(src, l)?;
                                let mut r = [0u32; 4];
                                for (out, &$w) in r.iter_mut().zip(s.iter()).take(n) {
                                    *out = $e;
                                }
                                self.wr(dst, l, r)?;
                            })
                        };
                    }
                    match op {
                        UnOp::NegF => un!(w => w ^ 0x8000_0000),
                        UnOp::NegI => un!(w => (w as i32).wrapping_neg() as u32),
                        UnOp::Not => un!(w => (w == 0) as u32),
                        UnOp::BitNot => un!(w => !w),
                    }
                }
                Op::Bin { op, a, b, n, a1, b1 } => {
                    let (a, b, n) = (slot_at::<N>(*a), slot_at::<N>(*b), *n as usize);
                    // a scalar operand's lane 0 for every component
                    let (sa, sb) = (usize::from(!*a1), usize::from(!*b1));
                    // each operation's own loop over the lanes, `bin` of a constant operation inlined into it
                    macro_rules! bin {
                        ($($o:ident),*) => {
                            match op {
                                $(BinOp::$o => each!(active, l, {
                                    let x = self.rd(a, l)?;
                                    let y = self.rd(b, l)?;
                                    let mut r = [0u32; 4];
                                    for (i, out) in r.iter_mut().enumerate().take(n) {
                                        *out = bin(BinOp::$o, comp(&x, i.saturating_mul(sa)), comp(&y, i.saturating_mul(sb)), ftz)?;
                                    }
                                    self.wr(dst, l, r)?;
                                }),)*
                            }
                        };
                    }
                    bin!(
                        AddF, SubF, MulF, DivF, ModF, AddU, SubU, MulU, DivU, ModU, And, Or, Xor, Shl, ShrU,
                        ShrI, EqW, NeW, EqF, NeF, LtF, LeF, GtF, GeF, LtU, LeU, GtU, GeU, LtI, LeI, GtI, GeI,
                        LAnd, LOr
                    )
                }
                Op::Select { c: cs, a, b, n, c1 } => {
                    let (cs, a, b, n) = (slot_at::<N>(*cs), slot_at::<N>(*a), slot_at::<N>(*b), *n as usize);
                    let sc = usize::from(!*c1);
                    each!(active, l, {
                        let cc = self.rd(cs, l)?;
                        let x = self.rd(a, l)?;
                        let y = self.rd(b, l)?;
                        let mut r = [0u32; 4];
                        for (i, out) in r.iter_mut().enumerate().take(n) {
                            *out = if comp(&cc, i.saturating_mul(sc)) != 0 {
                                comp(&x, i)
                            } else {
                                comp(&y, i)
                            };
                        }
                        self.wr(dst, l, r)?;
                    })
                }
                Op::Cvt { op, src, n } => {
                    let (src, n) = (slot_at::<N>(*src), *n as usize);
                    macro_rules! cvt {
                        ($w:ident => $e:expr) => {
                            each!(active, l, {
                                let s = self.rd(src, l)?;
                                let mut r = [0u32; 4];
                                for (out, &$w) in r.iter_mut().zip(s.iter()).take(n) {
                                    *out = $e;
                                }
                                self.wr(dst, l, r)?;
                            })
                        };
                    }
                    match op {
                        CvtOp::F2I => cvt!(w => float_to_int(f(w))),
                        CvtOp::F2U => cvt!(w => float_to_uint(f(w))),
                        CvtOp::I2F => cvt!(w => (w as i32 as f32).to_bits()),
                        CvtOp::U2F => cvt!(w => (w as f32).to_bits()),
                        CvtOp::B2F => cvt!(w => (if w != 0 { 1.0f32 } else { 0.0 }).to_bits()),
                        CvtOp::B2U => cvt!(w => u32::from(w != 0)),
                    }
                }
                Op::Math {
                    f: mf,
                    a,
                    b,
                    c: cc,
                    n,
                    c1,
                } => {
                    let (a, b, cc) = (slot_at::<N>(*a), slot_at::<N>(*b), slot_at::<N>(*cc));
                    each!(active, l, {
                        let x = self.rd(a, l)?;
                        let y = self.rd(b, l)?;
                        let mut z = self.rd(cc, l)?;
                        if *c1 {
                            z = [z[0]; 4];
                        }
                        self.wr(dst, l, math(*mf, x, y, z, *n, ftz))?;
                    })
                }
                Op::CopyWords { src, dst, words } => {
                    for l in 0..N {
                        if active.has(l) {
                            for k in 0..*words {
                                let w = self.mem_get(*src, l, k, 1)?[0];
                                let at = self.locals(offset(*dst, k)?, 1)?;
                                self.set_local_word(at, l, w)?;
                            }
                        }
                    }
                }
                Op::RqHit { q, committed, first } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let b = self.b;
                        let query = self
                            .query_slot(*q, l)?
                            .as_ref()
                            .ok_or("a ray query read before it was initialized")?;
                        let scene = b
                            .scenes
                            .get(&query.scene())
                            .ok_or("a ray query without its scene")?;
                        let hit = if *committed {
                            &query.committed
                        } else {
                            &query.candidate
                        };
                        let r = hit
                            .as_ref()
                            .map_or_else(Reported::none, |h| Reported::of(h, scene));
                        let first = *first as usize;
                        let mut put = |k: usize, v: [u32; 4]| -> Result<(), String> {
                            self.wr(first.saturating_add(k).saturating_mul(N), l, v)
                        };
                        put(0, [r.kind; 4])?;
                        put(1, [r.t.to_bits(); 4])?;
                        put(2, [r.custom; 4])?;
                        put(3, [r.instance; 4])?;
                        put(4, [r.sbt; 4])?;
                        put(5, [r.geometry; 4])?;
                        put(6, [r.primitive; 4])?;
                        let [bx, by] = r.barycentrics;
                        put(7, [bx.to_bits(), by.to_bits(), 0, 0])?;
                        put(8, [u32::from(r.front); 4])?;
                        for (c, (o, w)) in r.object_to_world.iter().zip(&r.world_to_object).enumerate() {
                            let [ox, oy, oz] = o.map(f32::to_bits);
                            put(9usize.saturating_add(c), [ox, oy, oz, 0])?;
                            let [wx, wy, wz] = w.map(f32::to_bits);
                            put(13usize.saturating_add(c), [wx, wy, wz, 0])?;
                        }
                    }
                }
                Op::ArrLen {
                    gid,
                    elem_words,
                    pick: Some(pick),
                } => {
                    for l in 0..N {
                        if active.has(l) {
                            let e = self.element_index(*gid, *pick, l)?;
                            let words = u32::try_from(self.element_words(*gid, e)?.len())
                                .map_err(|_| "an array of buffers over 4294967295 words")?;
                            self.wr(dst, l, [words.checked_div(*elem_words).unwrap_or(0); 4])?;
                        }
                    }
                }
                Op::ArrLen {
                    gid,
                    elem_words,
                    pick: None,
                } => {
                    let bound = self.b;
                    let count =
                        |n: usize| u32::try_from(n).map_err(|_| "a buffer over 4294967295 words".to_owned());
                    let words = match bound.data.at(*gid as usize)? {
                        _ if matches!(self.store.get(*gid as usize), Some(Some(_))) => count(
                            self.store
                                .get(*gid as usize)
                                .and_then(Option::as_ref)
                                .map_or(0, Vec::len),
                        )?,
                        Words::Elements(_) => {
                            return Err("arrayLength of an array of buffers without an element".into());
                        }
                        Words::Owned(d) => count(d.len())?,
                        Words::Slice(d) => count(d.len())?,
                        Words::Rows(d) => count(d.len())?.saturating_mul(4),
                        Words::Backend => match self.c.globals.at(*gid as usize)?.kind {
                            GlobalKind::Storage(view) => self.sampler.buffer_words(view)?,
                            _ => return Err("arrayLength of a global that is not a buffer".into()),
                        },
                    };
                    let len = words.checked_div(*elem_words).unwrap_or(0);
                    for l in 0..N {
                        self.wr(dst, l, [len; 4])?;
                    }
                }
                Op::Deriv {
                    src,
                    axis,
                    fine,
                    n,
                    prov,
                } => {
                    if self.grouping == Grouping::Quad {
                        if active != M::first(4) {
                            return Err("a quad's lanes diverge at a derivative".into());
                        }
                        let mut v = [[0u32; 4]; 4];
                        for (l, lane) in v.iter_mut().enumerate() {
                            *lane = self.rd(slot_at::<N>(*src), l)?;
                        }
                        for l in 0..4 {
                            let (from, to) = match (*axis == 0, *fine) {
                                (true, false) => (0, 1),
                                (false, false) => (0, 2),
                                (true, true) => {
                                    if l < 2 {
                                        (0, 1)
                                    } else {
                                        (2, 3)
                                    }
                                }
                                (false, true) => {
                                    if l % 2 == 0 {
                                        (0, 2)
                                    } else {
                                        (1, 3)
                                    }
                                }
                            };
                            let mut r = [0u32; 4];
                            let (a, b) = (
                                v.get(to).copied().unwrap_or_default(),
                                v.get(from).copied().unwrap_or_default(),
                            );
                            for (out, (&x, &y)) in r.iter_mut().zip(a.iter().zip(&b)).take(*n as usize) {
                                *out = fl(fl(f(x), ftz) - fl(f(y), ftz), ftz).to_bits();
                            }
                            self.wr(dst, l, r)?;
                        }
                    } else {
                        let p = prov.as_ref().ok_or("a derivative of a computed value (a scalar evaluation has those of its inputs only)")?;
                        let d = self.derivs.ok_or("no derivatives given")?;
                        let mut r = [0u32; 4];
                        for (out, &(arg, lane)) in r.iter_mut().zip(p.iter()).take(*n as usize) {
                            *out = d
                                .get(arg as usize)
                                .ok_or("no derivatives given for the argument")?
                                .get(*axis as usize)
                                .and_then(|axis| axis.get(lane as usize))
                                .copied()
                                .ok_or("no derivative given for the axis or the lane")?;
                        }
                        for l in 0..N {
                            if active.has(l) {
                                self.wr(dst, l, r)?;
                            }
                        }
                    }
                }
                Op::Sample {
                    tex,
                    smp,
                    coord,
                    cn,
                    layer,
                    level,
                    dim,
                    offset,
                    gather,
                    depth_ref,
                    depth,
                } => {
                    let cn = *cn as usize;
                    let c3 = |me: &Self, l: usize, s: u32| -> Result<[f32; 3], String> {
                        let [x, y, z, _] = me.rd(slot_at::<N>(s), l)?;
                        let lane = |i: usize, w: u32| if i < cn { fl(f(w), ftz) } else { 0.0 };
                        Ok([lane(0, x), lane(1, y), lane(2, z)])
                    };
                    // a quad's implicit level of detail comes from the coordinates' differences
                    let footprint = if self.grouping == Grouping::Quad
                        && matches!(level, LevelOp::Auto | LevelOp::Bias(_))
                    {
                        if active != M::first(4) {
                            return Err("a quad's lanes diverge at an implicit-LOD sample".into());
                        }
                        let mut uv = [[0f32; 3]; 4];
                        for (l, corner) in uv.iter_mut().enumerate() {
                            *corner = c3(self, l, *coord)?;
                        }
                        let [origin, right, below, _] = uv;
                        let d = |a: [f32; 3]| -> [f32; 3] {
                            let [ax, ay, az] = a;
                            let [ox, oy, oz] = origin;
                            [ax - ox, ay - oy, az - oz]
                        };
                        Some([d(right), d(below)])
                    } else {
                        None
                    };
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let rd = |s: u32, me: &Self| me.rd(slot_at::<N>(s), l);
                        let lod = match (level, footprint) {
                            (LevelOp::Auto, None) => ImageLod::Implicit,
                            (LevelOp::Auto, Some(g)) => ImageLod::Gradients(g),
                            (LevelOp::Zero, _) => ImageLod::Level(0.0),
                            (LevelOp::Exact(s), _) => ImageLod::Level(fl(f(rd(*s, self)?[0]), ftz)),
                            (LevelOp::Bias(s), None) => ImageLod::Bias(fl(f(rd(*s, self)?[0]), ftz)),
                            (LevelOp::Bias(s), Some(g)) => {
                                ImageLod::GradientsBias(g, fl(f(rd(*s, self)?[0]), ftz))
                            }
                            (LevelOp::Gradient(x, y), _) => {
                                ImageLod::Gradients([c3(self, l, *x)?, c3(self, l, *y)?])
                            }
                        };
                        let req = ImageSample {
                            texture: self.element(tex, l)?,
                            sampler: self.element(smp, l)?,
                            dim: *dim,
                            coord: c3(self, l, *coord)?,
                            layer: layer.map(|s| rd(s, self).map(|r| r[0] as f32)).transpose()?,
                            lod,
                            offset: *offset,
                            gather: *gather,
                            depth_ref: depth_ref
                                .map(|s| rd(s, self).map(|r| fl(f(r[0]), ftz)))
                                .transpose()?,
                            depth: *depth,
                        };
                        let r = self.sampler.sample(req)?;
                        if let Some(count) = self.samples.get_mut(l) {
                            *count = count.saturating_add(1);
                        }
                        let [r, g, b, a] = r;
                        self.wr(dst, l, [r.to_bits(), g.to_bits(), b.to_bits(), a.to_bits()])?;
                    }
                }
                Op::Texel {
                    tex,
                    coord,
                    cn,
                    dim,
                    storage,
                    layer,
                    sample,
                    mip,
                } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let cd = self.rd(slot_at::<N>(*coord), l)?;
                        let one = |s: &Option<u32>| -> Result<Option<i32>, String> {
                            s.map(|s| self.rd(slot_at::<N>(s), l).map(|r| r[0] as i32))
                                .transpose()
                        };
                        let req = ImageLoad {
                            texture: self.element(tex, l)?,
                            dim: *dim,
                            storage: *storage,
                            texel: texel_of(&cd, *cn as usize).map(|t| t as i32),
                            layer: one(layer)?,
                            sample: one(sample)?,
                            mip: one(mip)?,
                        };
                        let loaded = self.sampler.load(req)?;
                        self.wr(dst, l, loaded)?;
                    }
                }
                Op::Query {
                    tex,
                    dim,
                    kind,
                    level,
                } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let level = level
                            .map(|s| self.rd(slot_at::<N>(s), l).map(|r| r[0]))
                            .transpose()?;
                        let texture = self.element(tex, l)?;
                        let answer = self.sampler.query(ImageQuery {
                            texture,
                            dim: *dim,
                            kind: *kind,
                            level,
                        })?;
                        self.wr(dst, l, answer)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The lanes' states of ray query `q`.
    fn query_slots(&mut self, q: u32) -> Result<&mut Vec<Option<Query>>, String> {
        if self.queries.len() <= q as usize {
            self.queries.resize_with((q as usize).saturating_add(1), Vec::new);
        }
        let len = self.queries.len();
        let slots = self
            .queries
            .get_mut(q as usize)
            .ok_or_else(|| format!("internal error: ray query {q}, where the run has {len}"))?;
        if slots.len() < N {
            slots.resize(N, None);
        }
        Ok(slots)
    }

    /// The state of ray query `q` in lane `lane`.
    fn query_slot(&mut self, q: u32, lane: usize) -> Result<&mut Option<Query>, String> {
        self.query_slots(q)?
            .get_mut(lane)
            .ok_or_else(|| format!("internal error: ray query lane {lane}"))
    }

    /// The wave width and the lane count of a compute run; wave operations outside one are errors.
    fn wave_shape(&self) -> Result<(usize, usize), String> {
        if self.wave == 0 || self.wave > 128 {
            return Err("a wave operation outside a compute run (or of a wave wider than 128 lanes)".into());
        }
        Ok((self.wave, self.lanes))
    }

    /// The resource an operation names for lane `l`: for an array, the element its index picks in that lane.
    fn element(&self, res: &Res, l: usize) -> Result<Element, String> {
        let index = match res.pick {
            Pick::One => 0,
            Pick::Fixed(i) => i,
            Pick::Slot(s) => self.rd(slot_at::<N>(s), l)?[0],
        };
        if !matches!(res.pick, Pick::One) && res.count != iridia::Resource::UNBOUNDED && index >= res.count {
            return Err(format!(
                "resource {}: element {index} of an array of {} (D3D12 leaves it undefined)",
                res.id.0, res.count
            ));
        }
        Ok(Element { id: res.id, index })
    }

    fn block(&mut self, b: &'a [St], active: M) -> Result<Exits<M>, String> {
        let mut ex = Exits::<M>::default();
        let mut active = active;
        for st in b {
            if active.is_empty() {
                break;
            }
            match st {
                St::Run(lo, hi) => self.ops(*lo, *hi, active)?,
                St::Store {
                    loc: Loc::Local(o),
                    src,
                    n,
                } => {
                    let (base, n, src) = (
                        self.locals(*o as usize, *n as usize)?,
                        *n as usize,
                        slot_at::<N>(*src),
                    );
                    for l in 0..N {
                        if active.has(l) {
                            let v = self.rd(src, l)?;
                            for (i, w) in v.iter().take(n).enumerate() {
                                self.set_local_word(base.saturating_add(i), l, *w)?;
                            }
                        }
                    }
                }
                St::Store { loc, src, n } => {
                    for l in 0..N {
                        if active.has(l) {
                            let v = self.rd(slot_at::<N>(*src), l)?;
                            self.mem_put(*loc, l, 0, *n, v)?;
                        }
                    }
                }
                St::StoreDyn { ptr, src, n } => {
                    for l in 0..N {
                        if active.has(l) {
                            let x = self.dyn_extra(ptr, l)?;
                            let v = self.rd(slot_at::<N>(*src), l)?;
                            self.mem_put(ptr.loc, l, x, *n, v)?;
                        }
                    }
                }
                St::ImageStore {
                    tex,
                    dim,
                    coord,
                    cn,
                    layer,
                    value,
                    vn,
                } => {
                    for l in 0..N {
                        if active.has(l) {
                            let cd = self.rd(slot_at::<N>(*coord), l)?;
                            let element = self.element(tex, l)?;
                            self.sampler.store_image(ImageStore {
                                view: element.id,
                                index: element.index,
                                dim: *dim,
                                texel: texel_of(&cd, *cn as usize),
                                layer: layer
                                    .map(|s| self.rd(slot_at::<N>(s), l).map(|r| r[0]))
                                    .transpose()?,
                                value: self.rd(slot_at::<N>(*value), l)?,
                                lanes: *vn,
                            })?;
                        }
                    }
                }
                St::ImageAtomic {
                    tex,
                    dim,
                    coord,
                    cn,
                    layer,
                    op,
                    signed,
                    value,
                    result,
                } => {
                    for l in 0..N {
                        if active.has(l) {
                            let cd = self.rd(slot_at::<N>(*coord), l)?;
                            let element = self.element(tex, l)?;
                            let old = self.sampler.atomic_image(ImageAtomic {
                                view: element.id,
                                index: element.index,
                                dim: *dim,
                                texel: texel_of(&cd, *cn as usize),
                                layer: layer
                                    .map(|s| self.rd(slot_at::<N>(s), l).map(|r| r[0]))
                                    .transpose()?,
                                op: *op,
                                signed: *signed,
                                value: self.rd(slot_at::<N>(*value), l)?[0],
                            })?;
                            if let Some(r) = result {
                                self.wr(slot_at::<N>(*r), l, [old, 0, 0, 0])?;
                            }
                        }
                    }
                }
                St::Atomic {
                    ptr,
                    fun,
                    signed,
                    value,
                    compare,
                    result,
                    exchanged,
                } => {
                    for l in 0..N {
                        if active.has(l) {
                            let (g, o, elem) = match ptr.loc {
                                Loc::Global(g, o) => (g, o, None),
                                Loc::Elem(g, o, pick) => (g, o, Some(self.element_index(g, pick, l)?)),
                                Loc::Local(_) => return Err("atomic on a local".into()),
                            };
                            let word = offset(o, self.dyn_extra(ptr, l)?)?;
                            let v = self.rd(slot_at::<N>(*value), l)?[0];
                            let expected = match compare {
                                Some(c) => self.rd(slot_at::<N>(*c), l)?[0],
                                None => 0,
                            };
                            let memory = match elem {
                                Some(e) => self
                                    .elem_store
                                    .get_mut(g as usize)
                                    .and_then(|v| v.get_mut(e as usize)),
                                None => self.store.get_mut(g as usize),
                            };
                            let (old, swapped) = if let Some(Some(d)) = memory {
                                let cell = d.get_mut(word).ok_or("atomic past the end of the memory")?;
                                let old = *cell;
                                let (new, swapped) = atomic(*fun, *signed, old, v, expected);
                                *cell = new;
                                (old, swapped)
                            } else {
                                let view = match self.c.globals.at(g as usize)?.kind {
                                    GlobalKind::Storage(view) => view,
                                    GlobalKind::StorageArray(..) => {
                                        return Err("atomic on an element of an array of buffers this run does not hold (only a compute run writes them; the backend is not asked for element atomics)".into());
                                    }
                                    _ => return Err("atomic on a read-only global".into()),
                                };
                                if *fun != AtomOp::Add {
                                    return Err(format!(
                                        "atomic {fun:?} on a buffer the backend holds (only add is passed on)"
                                    ));
                                }
                                (
                                    self.sampler.atomic_add(
                                        view,
                                        u32::try_from(word)
                                            .map_err(|_| "atomic past the end of the memory")?,
                                        v,
                                    )?,
                                    true,
                                )
                            };
                            if let Some(r) = result {
                                self.wr(slot_at::<N>(*r), l, [old; 4])?;
                            }
                            if let Some(x) = exchanged {
                                self.wr(slot_at::<N>(*x), l, [u32::from(swapped); 4])?;
                            }
                        }
                    }
                }
                St::WaveBallot { predicate, result } => {
                    let (w, lanes) = self.wave_shape()?;
                    for base in (0..lanes).step_by(w) {
                        let end = base.saturating_add(w).min(lanes);
                        let mut bits = [0u32; 4];
                        for l in base..end {
                            let on = match predicate {
                                Some(p) => self.rd(slot_at::<N>(*p), l)?[0] != 0,
                                None => true,
                            };
                            if active.has(l) && on {
                                let bit = l.saturating_sub(base);
                                if let Some(word) = bits.get_mut(bit / 32) {
                                    *word |= 1u32
                                        .checked_shl(u32::try_from(bit % 32).unwrap_or(0))
                                        .unwrap_or(0);
                                }
                            }
                        }
                        for l in base..end {
                            if active.has(l) {
                                self.wr(slot_at::<N>(*result), l, bits)?;
                            }
                        }
                    }
                }
                St::WaveGather {
                    mode,
                    value,
                    n,
                    result,
                } => {
                    let (w, lanes) = self.wave_shape()?;
                    for base in (0..lanes).step_by(w) {
                        let end = base.saturating_add(w).min(lanes);
                        let first = (base..end).find(|&l| active.has(l));
                        let mut taken: Vec<(usize, Reg)> = Vec::new();
                        for l in base..end {
                            if !active.has(l) {
                                continue;
                            }
                            let at = l.saturating_sub(base);
                            let arg = |s: u32| -> Result<usize, String> {
                                self.rd(slot_at::<N>(s), l).map(|r| r[0] as usize)
                            };
                            let from = match mode {
                                WaveGather::First => first.unwrap_or(l),
                                WaveGather::Lane(s) | WaveGather::Shuffle(s) => base.saturating_add(arg(*s)?),
                                WaveGather::Down(s) => base.saturating_add(at).saturating_add(arg(*s)?),
                                WaveGather::Up(s) => {
                                    let step = arg(*s)?;
                                    if step <= at {
                                        base.saturating_add(at.saturating_sub(step))
                                    } else {
                                        base.saturating_add(at)
                                    }
                                }
                                WaveGather::Xor(s) => base.saturating_add(at ^ arg(*s)?),
                                WaveGather::QuadLane(s) => {
                                    base.saturating_add(at & !3).saturating_add(arg(*s)? & 3)
                                }
                                WaveGather::QuadSwap(x) => base.saturating_add(at ^ *x as usize),
                            };
                            // a lane outside the wave is undefined in D3D: the lane's own value
                            let from = if from < end { from } else { l };
                            let mut r = self.rd(slot_at::<N>(*value), from)?;
                            for w in r.iter_mut().skip(*n as usize) {
                                *w = 0;
                            }
                            taken.push((l, r));
                        }
                        for (l, r) in taken {
                            self.wr(slot_at::<N>(*result), l, r)?;
                        }
                    }
                }
                St::WaveOp {
                    op,
                    scan,
                    value,
                    n,
                    kind,
                    result,
                } => {
                    let (w, lanes) = self.wave_shape()?;
                    let ftz = self.ftz;
                    for base in (0..lanes).step_by(w) {
                        let end = base.saturating_add(w).min(lanes);
                        let members: Vec<usize> = (base..end).filter(|&l| active.has(l)).collect();
                        for i in 0..*n as usize {
                            let values: Vec<u32> = members
                                .iter()
                                .map(|&l| {
                                    self.rd(slot_at::<N>(*value), l)
                                        .map(|r| r.get(i).copied().unwrap_or(0))
                                })
                                .collect::<Result<_, _>>()?;
                            let total = values
                                .iter()
                                .copied()
                                .reduce(|a, b| wave_combine(*op, *kind, a, b, ftz))
                                .unwrap_or(0);
                            let mut running = wave_identity(*op, *kind);
                            for (k, &l) in members.iter().enumerate() {
                                let value = values.get(k).copied().unwrap_or(0);
                                let out = match scan {
                                    WaveScan::Reduce => total,
                                    WaveScan::Inclusive => {
                                        running = if k == 0 {
                                            value
                                        } else {
                                            wave_combine(*op, *kind, running, value, ftz)
                                        };
                                        running
                                    }
                                    WaveScan::Exclusive => {
                                        let before = running;
                                        running = if k == 0 {
                                            value
                                        } else {
                                            wave_combine(*op, *kind, running, value, ftz)
                                        };
                                        before
                                    }
                                };
                                let mut r = self.rd(slot_at::<N>(*result), l)?;
                                if let Some(lane) = r.get_mut(i) {
                                    *lane = out;
                                }
                                self.wr(slot_at::<N>(*result), l, r)?;
                            }
                        }
                    }
                }
                St::StoreAgg { loc, members } => {
                    for l in 0..N {
                        if active.has(l) {
                            let mut at = 0;
                            for &(s, n) in members {
                                let v = self.rd(slot_at::<N>(s), l)?;
                                self.mem_put(*loc, l, at, n, v)?;
                                at = at.saturating_add(u32::from(n));
                            }
                        }
                    }
                }
                St::RqInit {
                    q,
                    scene,
                    flags,
                    mask,
                    tmin,
                    tmax,
                    origin,
                    direction,
                } => {
                    if !self.b.scenes.contains_key(scene) {
                        return Err(format!(
                            "no acceleration structure given for resource {}",
                            scene.0
                        ));
                    }
                    for l in 0..N {
                        if active.has(l) {
                            let reg = |s: u32| self.rd(slot_at::<N>(s), l);
                            let float = |s: u32| -> Result<f32, String> { Ok(fl(f(reg(s)?[0]), self.ftz)) };
                            let vec3 = |s: u32| -> Result<[f32; 3], String> {
                                let [x, y, z, _] = reg(s)?;
                                Ok([fl(f(x), self.ftz), fl(f(y), self.ftz), fl(f(z), self.ftz)])
                            };
                            let query = Query::new(
                                *scene,
                                reg(*flags)?[0],
                                reg(*mask)?[0],
                                float(*tmin)?,
                                float(*tmax)?,
                                vec3(*origin)?,
                                vec3(*direction)?,
                            );
                            *self.query_slot(*q, l)? = Some(query);
                        }
                    }
                }
                St::RqProceed { q, result } => {
                    for l in 0..N {
                        if active.has(l) {
                            let b = self.b;
                            let query = self
                                .query_slot(*q, l)?
                                .as_mut()
                                .ok_or("a ray query proceeded before it was initialized")?;
                            let scene = b
                                .scenes
                                .get(&query.scene())
                                .ok_or("a ray query without its scene")?;
                            let more = query.proceed(scene);
                            self.wr(slot_at::<N>(*result), l, [u32::from(more); 4])?;
                        }
                    }
                }
                St::RqConfirm { q } => {
                    for l in 0..N {
                        if active.has(l) {
                            self.query_slot(*q, l)?
                                .as_mut()
                                .ok_or("a ray query confirmed before it was initialized")?
                                .confirm()?;
                        }
                    }
                }
                St::RtDecide(decision) => {
                    for l in 0..N {
                        if active.has(l)
                            && let Some(slot) = self.decisions.get_mut(l)
                        {
                            *slot = *decision;
                        }
                    }
                }
                St::RtTrace {
                    scene,
                    operands,
                    payload,
                } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let operand = |k: usize| -> Result<Reg, String> {
                            let slot = operands.at(k)?;
                            self.rd(slot_at::<N>(*slot), l)
                        };
                        let word = |k: usize| -> Result<u32, String> { Ok(operand(k)?[0]) };
                        let vec3 = |k: usize| -> Result<[f32; 3], String> {
                            let [x, y, z, _] = operand(k)?;
                            Ok([f(x), f(y), f(z)])
                        };
                        let (flags, mask, hit_offset, stride, miss) =
                            (word(0)?, word(1)?, word(2)?, word(3)?, word(4)?);
                        let (origin, direction) = (vec3(5)?, vec3(7)?);
                        let (t_min, t_max) = (f(word(6)?), f(word(8)?));
                        let at = payload.0 as usize;
                        let mut words = self.read_locals(l, at, payload.1 as usize)?;
                        self.sampler.trace_ray(super::request::TraceRay {
                            scene: *scene,
                            flags,
                            mask,
                            hit_group_offset: hit_offset,
                            hit_group_stride: stride,
                            miss_index: miss,
                            origin,
                            t_min,
                            direction,
                            t_max,
                            payload: &mut words,
                        })?;
                        self.write_locals(l, at, &words)?;
                    }
                }
                St::RtReport {
                    t,
                    kind,
                    attributes,
                    result,
                } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let at = attributes.0 as usize;
                        let mut words = self.read_locals(l, at, attributes.1 as usize)?;
                        let accepted = self.sampler.report_hit(super::request::ReportHit {
                            t: f(self.rd(slot_at::<N>(*t), l)?[0]),
                            hit_kind: self.rd(slot_at::<N>(*kind), l)?[0],
                            attributes: &mut words,
                        })?;
                        self.write_locals(l, at, &words)?;
                        if let Some(r) = result {
                            self.wr(slot_at::<N>(*r), l, [u32::from(accepted); 4])?;
                        }
                    }
                }
                St::RtCall { index, parameter } => {
                    for l in 0..N {
                        if !active.has(l) {
                            continue;
                        }
                        let at = parameter.0 as usize;
                        let mut words = self.read_locals(l, at, parameter.1 as usize)?;
                        self.sampler.call_shader(super::request::CallShader {
                            index: self.rd(slot_at::<N>(*index), l)?[0],
                            parameter: &mut words,
                        })?;
                        self.write_locals(l, at, &words)?;
                    }
                }
                St::RqGenerate { q, t } => {
                    for l in 0..N {
                        if active.has(l) {
                            let t = fl(f(self.rd(slot_at::<N>(*t), l)?[0]), self.ftz);
                            self.query_slot(*q, l)?
                                .as_mut()
                                .ok_or("an intersection generated before the query was initialized")?
                                .generate(t)?;
                        }
                    }
                }
                St::RqTerminate { q } => {
                    for l in 0..N {
                        if active.has(l) {
                            self.query_slot(*q, l)?
                                .as_mut()
                                .ok_or("a ray query ended before it was initialized")?
                                .terminate();
                        }
                    }
                }
                St::Barrier => {
                    // lockstep makes every write before the barrier visible after it; what a barrier needs is that
                    // every invocation still in the program is at it (D3D leaves a divergent barrier undefined)
                    if self.store.is_empty() {
                        return Err("a barrier outside a compute run".into());
                    }
                    if active != self.group.minus(self.retired) {
                        return Err("a barrier that not every invocation of the group reaches".into());
                    }
                }
                St::If { c, a, r } => {
                    let mut take = M::none();
                    for l in 0..N {
                        if active.has(l) && self.rd(slot_at::<N>(*c), l)?[0] != 0 {
                            take.insert(l);
                        }
                    }
                    let e1 = self.block(a, take)?;
                    let e2 = self.block(r, active.minus(take))?;
                    let (b, cn, rt) = (e1.brk.or(e2.brk), e1.cont.or(e2.cont), e1.ret.or(e2.ret));
                    ex.brk = ex.brk.or(b);
                    ex.cont = ex.cont.or(cn);
                    ex.ret = ex.ret.or(rt);
                    active = active.minus(b.or(cn).or(rt));
                }
                St::Loop { body, cont, break_if } => {
                    let mut lm = active;
                    loop {
                        self.count(lm, 1)?;
                        let e = self.block(body, lm)?;
                        ex.ret = ex.ret.or(e.ret);
                        lm = lm.minus(e.ret.or(e.brk));
                        if lm.is_empty() {
                            break;
                        }
                        let e2 = self.block(cont, lm)?;
                        ex.ret = ex.ret.or(e2.ret);
                        lm = lm.minus(e2.ret.or(e2.brk));
                        if let Some(bi) = break_if {
                            for l in 0..N {
                                if lm.has(l) && self.rd(slot_at::<N>(*bi), l)?[0] != 0 {
                                    lm.remove(l);
                                }
                            }
                        }
                        if lm.is_empty() {
                            break;
                        }
                    }
                    active = active.minus(ex.ret);
                }
                St::Switch { sel, cases } => {
                    let mut chosen = [usize::MAX; N];
                    for (l, slot) in chosen.iter_mut().enumerate() {
                        if active.has(l) {
                            let v = self.rd(slot_at::<N>(*sel), l)?[0];
                            *slot = cases
                                .iter()
                                .position(|c| c.0 == Some(v))
                                .or_else(|| cases.iter().position(|c| c.0.is_none()))
                                .unwrap_or(usize::MAX);
                        }
                    }
                    let mut carry = M::none();
                    for (i, (_, fall, body)) in cases.iter().enumerate() {
                        let mut enter = carry;
                        for (l, slot) in chosen.iter().enumerate() {
                            if *slot == i {
                                enter.insert(l);
                            }
                        }
                        carry = M::none();
                        if enter.is_empty() {
                            continue;
                        }
                        let e = self.block(body, enter)?;
                        ex.cont = ex.cont.or(e.cont);
                        ex.ret = ex.ret.or(e.ret);
                        let done = e.brk.or(e.cont).or(e.ret);
                        if *fall {
                            carry = enter.minus(done);
                        }
                    }
                    active = active.minus(ex.cont.or(ex.ret));
                }
                St::Break => {
                    ex.brk = ex.brk.or(active);
                    active = M::none();
                }
                St::Continue => {
                    ex.cont = ex.cont.or(active);
                    active = M::none();
                }
                St::Return(result) => {
                    self.retired = self.retired.or(active);
                    for l in 0..N {
                        if active.has(l) {
                            // into the lane's own room, kept from run to run
                            let mut members = std::mem::take(
                                self.outs
                                    .get_mut(l)
                                    .ok_or("internal error: a result for a lane the run does not have")?,
                            );
                            members.clear();
                            for member in result {
                                match *member {
                                    Member::Reg(s, n) => {
                                        let mut r = [0u32; 4];
                                        let v = self.rd(slot_at::<N>(s), l)?;
                                        for (to, from) in r.iter_mut().zip(v.iter()).take(n as usize) {
                                            *to = *from;
                                        }
                                        members.push(r);
                                    }
                                    Member::Mem {
                                        word, count, width, ..
                                    } => {
                                        for k in 0..count as usize {
                                            let at = (word as usize)
                                                .saturating_add(k.saturating_mul(width as usize));
                                            let mut r = [0u32; 4];
                                            for (i, w) in r.iter_mut().take(width as usize).enumerate() {
                                                *w = self.local_word(at.saturating_add(i), l)?;
                                            }
                                            members.push(r);
                                        }
                                    }
                                }
                            }
                            if let Some(room) = self.outs.get_mut(l) {
                                *room = members;
                            }
                        }
                    }
                    ex.ret = ex.ret.or(active);
                    active = M::none();
                }
                St::Kill => {
                    self.killed = self.killed.or(active);
                    self.retired = self.retired.or(active);
                    if self.grouping == Grouping::Independent {
                        ex.ret = ex.ret.or(active);
                        active = M::none();
                    }
                }
            }
        }
        Ok(ex)
    }

    /// Run once, each lane with its arguments (one register per entry argument, the lanes of its value first).
    pub fn run(&mut self, args: [&[[u32; 4]]; N]) -> Result<[Output; N], String> {
        self.run_first(args, N)
    }

    /// [`Runner::run`] with lanes `0..count` only; the outputs of the others are empty (independent invocations).
    pub fn run_first(&mut self, args: [&[[u32; 4]]; N], count: usize) -> Result<[Output; N], String> {
        self.run_in_place(args, count)?;
        let c: &'a Executable = self.c;
        let mut outputs = Vec::with_capacity(N);
        for l in 0..N {
            let (values, discarded, steps, samples) = self.lane(l);
            outputs.push(Output {
                pointees: c
                    .pointer_args
                    .iter()
                    .map(|&(_, off, words)| self.read_locals(l, off as usize, words as usize))
                    .collect::<Result<_, _>>()?,
                decision: self.decisions.get(l).copied().unwrap_or_default(),
                values: values.to_vec(),
                discarded,
                steps,
                samples,
            });
        }
        outputs
            .try_into()
            .map_err(|_| "internal error: a run of N lanes made another number of outputs".to_owned())
    }

    /// [`Runner::run_first`] with the results left in the runner, for [`Runner::lane`] to read: a run that
    /// allocates nothing once the runner's room is grown.
    pub(super) fn run_in_place(&mut self, args: [&[[u32; 4]]; N], count: usize) -> Result<(), String> {
        if count > N || (count < N && self.grouping != Grouping::Independent) {
            return Err(format!(
                "{count} of {N} lanes run: only independent invocations run fewer than all"
            ));
        }
        self.steps = 0;
        self.lane_steps = [0; N];
        self.mem.fill(0);
        self.samples = [0; N];
        self.killed = M::none();
        self.queries.clear();
        self.decisions = [Decision::None; N];
        for o in self.outs.iter_mut() {
            o.clear();
        }
        for (l, given) in args.iter().enumerate() {
            if given.len() < self.c.arg_len {
                return Err(format!(
                    "{} argument registers given, the function takes {}",
                    given.len(),
                    self.c.arg_len
                ));
            }
            for (i, (s, n)) in self.c.args.iter().enumerate() {
                if *n != 0 {
                    let argument = given.at(*self.c.arg_at.at(i)?)?;
                    self.wr(slot_at::<N>(*s), l, *argument)?;
                }
            }
            for &(at, off, count, width) in &self.c.array_args {
                for k in 0..count as usize {
                    let base = (off as usize).saturating_add(k.saturating_mul(width as usize));
                    let element = given.at(at.saturating_add(k))?;
                    for (i, w) in element.iter().take(width as usize).enumerate() {
                        self.set_local_word(base.saturating_add(i), l, *w)?;
                    }
                }
            }
            for &(at, off, words) in &self.c.pointer_args {
                for w in 0..words as usize {
                    let register = given.at(at.saturating_add(w / 4))?;
                    self.set_local_word((off as usize).saturating_add(w), l, comp(register, w % 4))?;
                }
            }
        }
        let c: &'a Executable = self.c;
        self.block(&c.body, M::first(count))?;
        Ok(())
    }

    /// Lane `l` of the last run: its result members, whether it was discarded, its steps and its samples.
    pub(super) fn lane(&self, l: usize) -> (&[Reg], bool, u64, u64) {
        let steps = match self.grouping {
            Grouping::Independent => self.lane_steps.get(l).copied().unwrap_or(0),
            Grouping::Quad | Grouping::Group => self.steps,
        };
        (
            self.outs.get(l).map_or(&[], Vec::as_slice),
            self.killed.has(l),
            steps,
            self.samples.get(l).copied().unwrap_or(0),
        )
    }
}

/// Two words combined by a wave operation (booleans are 0 and 1).
fn wave_combine(op: WaveOp, kind: WaveKind, a: u32, b: u32, ftz: bool) -> u32 {
    let g = |v: u32| fl(f(v), ftz);
    match (op, kind) {
        (WaveOp::All, _) => u32::from(a != 0 && b != 0),
        (WaveOp::Any, _) => u32::from(a != 0 || b != 0),
        (WaveOp::And, _) => a & b,
        (WaveOp::Or, _) => a | b,
        (WaveOp::Xor, _) => a ^ b,
        (WaveOp::Add, WaveKind::F) => fl(g(a) + g(b), ftz).to_bits(),
        (WaveOp::Mul, WaveKind::F) => fl(g(a) * g(b), ftz).to_bits(),
        (WaveOp::Min, WaveKind::F) => g(a).min(g(b)).to_bits(),
        (WaveOp::Max, WaveKind::F) => g(a).max(g(b)).to_bits(),
        (WaveOp::Add, _) => a.wrapping_add(b),
        (WaveOp::Mul, _) => a.wrapping_mul(b),
        (WaveOp::Min, WaveKind::I) => (a as i32).min(b as i32) as u32,
        (WaveOp::Max, WaveKind::I) => (a as i32).max(b as i32) as u32,
        (WaveOp::Min, WaveKind::U) => a.min(b),
        (WaveOp::Max, WaveKind::U) => a.max(b),
    }
}
/// What a wave operation over no lanes gives, and an exclusive scan's first lane.
fn wave_identity(op: WaveOp, kind: WaveKind) -> u32 {
    match (op, kind) {
        (WaveOp::All, _) => 1,
        (WaveOp::Any | WaveOp::Or | WaveOp::Xor, _) => 0,
        (WaveOp::And, _) => u32::MAX,
        (WaveOp::Add, WaveKind::F) => 0,
        (WaveOp::Add, _) => 0,
        (WaveOp::Mul, WaveKind::F) => 1.0f32.to_bits(),
        (WaveOp::Mul, _) => 1,
        (WaveOp::Min, WaveKind::F) => f32::INFINITY.to_bits(),
        (WaveOp::Max, WaveKind::F) => f32::NEG_INFINITY.to_bits(),
        (WaveOp::Min, WaveKind::I) => i32::MAX as u32,
        (WaveOp::Max, WaveKind::I) => i32::MIN as u32,
        (WaveOp::Min, WaveKind::U) => u32::MAX,
        (WaveOp::Max, WaveKind::U) => 0,
    }
}

/// The new word and whether a compare-exchange swapped, of an atomic operation on `old`.
fn atomic(fun: AtomOp, signed: bool, old: u32, v: u32, expected: u32) -> (u32, bool) {
    let new = match fun {
        AtomOp::Add => old.wrapping_add(v),
        AtomOp::Sub => old.wrapping_sub(v),
        AtomOp::And => old & v,
        AtomOp::Or => old | v,
        AtomOp::Xor => old ^ v,
        AtomOp::Min if signed => (old as i32).min(v as i32) as u32,
        AtomOp::Max if signed => (old as i32).max(v as i32) as u32,
        AtomOp::Min => old.min(v),
        AtomOp::Max => old.max(v),
        AtomOp::Exchange => v,
        AtomOp::CompareExchange => {
            return if old == expected { (v, true) } else { (old, false) };
        }
    };
    (new, true)
}

#[inline(always)]
fn bin(op: BinOp, x: u32, y: u32, ftz: bool) -> Result<u32, String> {
    use BinOp::*;
    let fa = || fl(f(x), ftz);
    let fb = || fl(f(y), ftz);
    Ok(match op {
        AddF => fl(fa() + fb(), ftz).to_bits(),
        SubF => fl(fa() - fb(), ftz).to_bits(),
        MulF => fl(fa() * fb(), ftz).to_bits(),
        DivF => fl(fa() / fb(), ftz).to_bits(),
        // C fmod: the remainder of a truncating division, the sign of the dividend
        ModF => fl(fa() % fb(), ftz).to_bits(),
        AddU => x.wrapping_add(y),
        SubU => x.wrapping_sub(y),
        MulU => x.wrapping_mul(y),
        DivU => x.checked_div(y).ok_or("integer divide by zero in the program")?,
        ModU => x.checked_rem(y).ok_or("integer modulo by zero in the program")?,
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        Shl => x.wrapping_shl(y),
        ShrU => x.wrapping_shr(y),
        ShrI => (x as i32).wrapping_shr(y) as u32,
        EqW => (x == y) as u32,
        NeW => (x != y) as u32,
        EqF => (fa() == fb()) as u32,
        NeF => (fa() != fb()) as u32,
        LtF => (fa() < fb()) as u32,
        LeF => (fa() <= fb()) as u32,
        GtF => (fa() > fb()) as u32,
        GeF => (fa() >= fb()) as u32,
        LtU => (x < y) as u32,
        LeU => (x <= y) as u32,
        GtU => (x > y) as u32,
        GeU => (x >= y) as u32,
        LtI => ((x as i32) < (y as i32)) as u32,
        LeI => ((x as i32) <= (y as i32)) as u32,
        GtI => ((x as i32) > (y as i32)) as u32,
        GeI => ((x as i32) >= (y as i32)) as u32,
        LAnd => (x != 0 && y != 0) as u32,
        LOr => (x != 0 || y != 0) as u32,
    })
}

fn math(mf: MathOp, a: Reg, b: Reg, c: Reg, n: u8, z: bool) -> Reg {
    let mut r = [0u32; 4];
    if mf == MathOp::Dot {
        let mut acc = 0.0f32;
        for i in 0..n as usize {
            let prod = fl(fl(f(comp(&a, i)), z) * fl(f(comp(&b, i)), z), z);
            acc = if i == 0 { prod } else { fl(acc + prod, z) };
        }
        r[0] = acc.to_bits();
        return r;
    }
    match mf {
        MathOp::All => {
            r[0] = u32::from(a.iter().take(n as usize).all(|&w| w != 0));
            return r;
        }
        MathOp::Any => {
            r[0] = u32::from(a.iter().take(n as usize).any(|&w| w != 0));
            return r;
        }
        MathOp::Normalize => {
            let lanes = a.get(..n as usize).unwrap_or_default();
            let mut sum = 0.0f32;
            for (i, &lane) in lanes.iter().enumerate() {
                let sq = fl(fl(f(lane), z) * fl(f(lane), z), z);
                sum = if i == 0 { sq } else { fl(sum + sq, z) };
            }
            let length = fl(crate::narrow(f64::from(sum).sqrt()), z);
            for (out, &lane) in r.iter_mut().zip(lanes) {
                *out = fl(fl(f(lane), z) / length, z).to_bits();
            }
            return r;
        }
        _ => {}
    }
    if mf == MathOp::PackHalf {
        r[0] = half::f16::from_f32(f(a[0])).to_bits() as u32
            | ((half::f16::from_f32(f(a[1])).to_bits() as u32) << 16);
        return r;
    }
    if mf == MathOp::UnpackHalf {
        let packed = comp(&a, 0);
        let half = |bits: u32| {
            half::f16::from_bits(u16::try_from(bits & 0xffff).unwrap_or(0))
                .to_f32()
                .to_bits()
        };
        r[0] = half(packed);
        r[1] = half(packed >> 16);
        return r;
    }
    for i in 0..n as usize {
        let (x, y, w) = (comp(&a, i), comp(&b, i), comp(&c, i));
        let g = |v: f32| fl(v, z);
        let one = |h: &dyn Fn(f32) -> f32| g(h(g(f(x)))).to_bits();
        let out = match mf {
            MathOp::AbsF => (f(x).abs()).to_bits(),
            MathOp::AbsI => (x as i32).wrapping_abs() as u32,
            MathOp::MinF => g(g(f(x)).min(g(f(y)))).to_bits(),
            MathOp::MaxF => g(g(f(x)).max(g(f(y)))).to_bits(),
            MathOp::MinU => x.min(y),
            MathOp::MaxU => x.max(y),
            MathOp::MinI => (x as i32).min(y as i32) as u32,
            MathOp::MaxI => (x as i32).max(y as i32) as u32,
            MathOp::ClampF => g(g(f(x)).max(g(f(y))).min(g(f(w)))).to_bits(),
            MathOp::SatF => one(&|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }),
            MathOp::Floor => one(&|v| v.floor()),
            MathOp::Ceil => one(&|v| v.ceil()),
            MathOp::Trunc => one(&|v| v.trunc()),
            MathOp::Round => one(&|v| v.round_ties_even()),
            MathOp::Fract => one(&|v| v - v.floor()),
            MathOp::Exp2 => one(&|v| crate::narrow(f64::from(v).exp2())),
            MathOp::Log2 => one(&|v| crate::narrow(f64::from(v).log2())),
            MathOp::Sin => one(&|v| crate::narrow(f64::from(v).sin())),
            MathOp::Cos => one(&|v| crate::narrow(f64::from(v).cos())),
            // to the nearest half, ties to even, and back
            MathOp::QuantizeF16 => one(&|v| half::f16::from_f32(v).to_f32()),
            MathOp::Tan => one(&|v| crate::narrow(f64::from(v).tan())),
            MathOp::Asin => one(&|v| crate::narrow(f64::from(v).asin())),
            MathOp::Acos => one(&|v| crate::narrow(f64::from(v).acos())),
            MathOp::Atan => one(&|v| crate::narrow(f64::from(v).atan())),
            MathOp::Sinh => one(&|v| crate::narrow(f64::from(v).sinh())),
            MathOp::Cosh => one(&|v| crate::narrow(f64::from(v).cosh())),
            MathOp::Tanh => one(&|v| crate::narrow(f64::from(v).tanh())),
            // one rounding, as a fused multiply-add
            MathOp::Fma => g(g(f(x)).mul_add(g(f(y)), g(f(w)))).to_bits(),
            MathOp::Sqrt => one(&|v| crate::narrow(f64::from(v).sqrt())),
            MathOp::Rsqrt => one(&|v| crate::narrow(1.0 / f64::from(v).sqrt())),
            MathOp::Popcnt => x.count_ones(),
            MathOp::RevBits => x.reverse_bits(),
            MathOp::Clz => x.leading_zeros(),
            MathOp::Ctz => {
                if x == 0 {
                    u32::MAX
                } else {
                    x.trailing_zeros()
                }
            }
            // x * (1 - a) + y * a, each product and the sum rounded
            MathOp::Mix => g(g(g(f(x)) * g(1.0 - g(f(w)))) + g(g(f(y)) * g(f(w)))).to_bits(),
            MathOp::SignF => {
                let v = f(x);
                (if v > 0.0 {
                    1.0f32
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                })
                .to_bits()
            }
            MathOp::Pow => g(crate::narrow(f64::from(g(f(x))).powf(f64::from(g(f(y)))))).to_bits(),
            MathOp::SignI => (x as i32).signum() as u32,
            MathOp::IsNan => u32::from(f(x).is_nan()),
            MathOp::IsInf => u32::from(f(x).is_infinite()),
            MathOp::Dot
            | MathOp::PackHalf
            | MathOp::UnpackHalf
            | MathOp::All
            | MathOp::Any
            | MathOp::Normalize => 0,
        };
        if let Some(slot) = r.get_mut(i) {
            *slot = out;
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------------- compute

/// What a run of a mesh program does with the memory of a thread group's output variable when the group has finished.
pub(super) type GroupHook<'a> = dyn FnMut([u32; 3], &[u32]) -> Result<(), String> + 'a;

/// What a compute run did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dispatched {
    /// Thread groups run.
    pub groups: u64,
    /// Operations and loop turns executed, summed over the groups (a group's lanes run each together).
    pub steps: u64,
}

impl Executable {
    /// The thread group of a compute program (`numthreads`), or `None` for any other.
    pub fn workgroup_size(&self) -> Option<[u32; 3]> {
        self.workgroup
    }

    /// Run `groups` thread groups of a compute program. Each group's invocations run in lockstep (a barrier is the
    /// point where every invocation still in the program has arrived), thread-group memory starts at zero in every
    /// group, and the groups run one after another in x, y, z order. `buffers` holds the words of every read-write
    /// buffer by resource: writes and atomics change them and later reads see the changes, as they do between
    /// groups. A buffer not given is one the backend answers reads of and takes writes to ([`Backend::read_buffer`],
    /// [`Backend::store_buffer`]). `wave` is the wave width the wave operations and lane built-ins report; it must
    /// divide the thread group's invocations into whole waves only for programs that care.
    pub fn dispatch(
        &self,
        bound: &Bound<'_>,
        backend: &mut dyn Backend,
        buffers: &mut BTreeMap<ResourceId, Vec<u32>>,
        groups: [u32; 3],
        wave: u32,
    ) -> Result<Dispatched, String> {
        self.dispatch_elements(bound, backend, buffers, &mut BTreeMap::new(), groups, wave)
    }

    /// Run `groups` thread groups of a mesh program, handing the memory of the output variable (its words, as
    /// [`Executable::mesh_output`] says) to `after` when each group has finished.
    pub(super) fn dispatch_mesh(
        &self,
        bound: &Bound<'_>,
        backend: &mut dyn Backend,
        groups: [u32; 3],
        wave: u32,
        after: &mut GroupHook<'_>,
    ) -> Result<Dispatched, String> {
        let (global, _) = self.mesh_output.ok_or("not a mesh program")?;
        let size = self.workgroup.ok_or("not a mesh program")?;
        if wave == 0 {
            return Err("a wave of no lanes".into());
        }
        let ids = (0..groups[2])
            .flat_map(move |z| (0..groups[1]).flat_map(move |y| (0..groups[0]).map(move |x| [x, y, z])));
        let mut buffers = BTreeMap::new();
        let mut elements = BTreeMap::new();
        match group_lanes(size)? {
            0 => Err("a thread group of no invocations".into()),
            1..=64 => self.groups::<64>(
                bound,
                backend,
                &mut buffers,
                &mut elements,
                ids,
                size,
                wave,
                Some((global, after)),
            ),
            65..=256 => self.groups::<256>(
                bound,
                backend,
                &mut buffers,
                &mut elements,
                ids,
                size,
                wave,
                Some((global, after)),
            ),
            257..=1024 => self.groups::<1024>(
                bound,
                backend,
                &mut buffers,
                &mut elements,
                ids,
                size,
                wave,
                Some((global, after)),
            ),
            _ => Err("a thread group of more than 1024 invocations".into()),
        }
    }

    /// [`Executable::dispatch`] for a program that also writes elements of arrays of buffers, `elements` holding
    /// their words by resource and element.
    pub fn dispatch_elements(
        &self,
        bound: &Bound<'_>,
        backend: &mut dyn Backend,
        buffers: &mut BTreeMap<ResourceId, Vec<u32>>,
        elements: &mut BTreeMap<(ResourceId, u32), Vec<u32>>,
        groups: [u32; 3],
        wave: u32,
    ) -> Result<Dispatched, String> {
        let size = self.workgroup.ok_or("not a compute program")?;
        if wave == 0 {
            return Err("a wave of no lanes".into());
        }
        let ids = (0..groups[2])
            .flat_map(move |z| (0..groups[1]).flat_map(move |y| (0..groups[0]).map(move |x| [x, y, z])));
        match group_lanes(size)? {
            0 => Err("a thread group of no invocations".into()),
            1..=64 => self.groups::<64>(bound, backend, buffers, elements, ids, size, wave, None),
            65..=256 => self.groups::<256>(bound, backend, buffers, elements, ids, size, wave, None),
            257..=1024 => self.groups::<1024>(bound, backend, buffers, elements, ids, size, wave, None),
            _ => Err("a thread group of more than 1024 invocations".into()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn groups<const N: usize>(
        &self,
        bound: &Bound<'_>,
        backend: &mut dyn Backend,
        buffers: &mut BTreeMap<ResourceId, Vec<u32>>,
        elements: &mut BTreeMap<(ResourceId, u32), Vec<u32>>,
        ids: impl Iterator<Item = [u32; 3]>,
        size: [u32; 3],
        wave: u32,
        mut after: Option<(usize, &mut GroupHook<'_>)>,
    ) -> Result<Dispatched, String> {
        let lanes = group_lanes(size)? as usize;
        // the buffers move into the run, and back whatever happens
        let store: Vec<Option<Vec<u32>>> = self
            .globals
            .iter()
            .map(|g| match g.kind {
                GlobalKind::Storage(id) => buffers.remove(&id),
                _ => None,
            })
            .collect();
        let elem_store: Vec<Vec<Option<Vec<u32>>>> = self
            .globals
            .iter()
            .map(|g| match g.kind {
                GlobalKind::StorageArray(id, count) => {
                    (0..count).map(|k| elements.remove(&(id, k))).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let scratch = Scratch::<N>::new(self);
        let mut runner = Runner::<N, Wide> {
            c: self,
            b: bound,
            regs: scratch.regs,
            mem: scratch.mem,
            sampler: backend,
            ftz: self.flush_denormals,
            grouping: Grouping::Group,
            steps: 0,
            lane_steps: [0; N],
            samples: [0; N],
            outs: scratch.outs,
            decisions: [Decision::None; N],
            killed: Wide::none(),
            store,
            elem_store,
            queries: Vec::new(),
            group: Wide::first(lanes),
            retired: Wide::none(),
            lanes,
            wave: wave as usize,
            derivs: None,
        };
        let mut done = Dispatched::default();
        let mut result = Ok(());
        for id in ids {
            match runner.run_group(id, size, wave) {
                Ok(steps) => {
                    done.groups = done.groups.saturating_add(1);
                    done.steps = done.steps.saturating_add(steps);
                    if let Some((global, hook)) = after.as_mut() {
                        let words = runner
                            .store
                            .get(*global)
                            .and_then(|memory| memory.as_deref())
                            .unwrap_or(&[]);
                        if let Err(e) = hook(id, words) {
                            result = Err(format!("group {id:?}: {e}"));
                            break;
                        }
                    }
                }
                Err(e) => {
                    result = Err(format!("group {id:?}: {e}"));
                    break;
                }
            }
        }
        for (g, memory) in self.globals.iter().zip(runner.store) {
            if let (GlobalKind::Storage(id), Some(words)) = (&g.kind, memory) {
                buffers.insert(*id, words);
            }
        }
        for (g, held) in self.globals.iter().zip(runner.elem_store) {
            if let GlobalKind::StorageArray(id, _) = g.kind {
                for (k, words) in held.into_iter().enumerate() {
                    if let Some(words) = words {
                        elements.insert((id, u32::try_from(k).unwrap_or(u32::MAX)), words);
                    }
                }
            }
        }
        result.map(|()| done)
    }
}

impl<'a, const N: usize> Runner<'a, N, Wide> {
    /// Run one thread group: `id` its position in the dispatch, `size` its `numthreads`. Returns the steps taken.
    fn run_group(&mut self, id: [u32; 3], size: [u32; 3], wave: u32) -> Result<u64, String> {
        let c: &'a Executable = self.c;
        let lanes = group_lanes(size)? as usize;
        self.steps = 0;
        self.killed = Wide::none();
        self.retired = Wide::none();
        self.queries.clear();
        for (i, g) in c.globals.iter().enumerate() {
            if let GlobalKind::Workgroup(words) = g.kind {
                match self.store.get_mut(i) {
                    Some(Some(v)) => v.iter_mut().for_each(|w| *w = 0),
                    Some(slot) => *slot = Some(vec![0; words as usize]),
                    None => return Err(format!("internal error: global {i} has no memory in the run")),
                }
            }
        }
        for l in 0..lanes {
            let [width, height, _] = size;
            let index = u32::try_from(l).map_err(|_| "a thread group of over 4294967295 invocations")?;
            let local = [
                index.checked_rem(width).unwrap_or(0),
                index
                    .checked_div(width)
                    .unwrap_or(0)
                    .checked_rem(height)
                    .unwrap_or(0),
                index.checked_div(width.saturating_mul(height)).unwrap_or(0),
            ];
            for (i, (slot, n)) in c.args.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                let value = match c.builtins.get(i).copied().flatten() {
                    Some(naga::BuiltIn::WorkGroupId) => [id[0], id[1], id[2], 0],
                    Some(naga::BuiltIn::LocalInvocationId) => [local[0], local[1], local[2], 0],
                    Some(naga::BuiltIn::GlobalInvocationId) => {
                        let global = |axis: usize| {
                            comp3(id, axis)
                                .saturating_mul(comp3(size, axis))
                                .saturating_add(comp3(local, axis))
                        };
                        [global(0), global(1), global(2), 0]
                    }
                    Some(naga::BuiltIn::LocalInvocationIndex) => [index, 0, 0, 0],
                    Some(naga::BuiltIn::SubgroupInvocationId) => {
                        [index.checked_rem(wave).unwrap_or(0), 0, 0, 0]
                    }
                    Some(naga::BuiltIn::SubgroupSize) => [wave, 0, 0, 0],
                    Some(naga::BuiltIn::SubgroupId) => [index.checked_div(wave).unwrap_or(0), 0, 0, 0],
                    Some(naga::BuiltIn::NumSubgroups) => {
                        [u32::try_from(lanes).unwrap_or(u32::MAX).div_ceil(wave), 0, 0, 0]
                    }
                    other => return Err(format!("compute entry argument {i} carries {other:?}")),
                };
                self.wr(slot_at::<N>(*slot), l, value)?;
            }
            for w in 0..c.local_words {
                self.set_local_word(w, l, 0)?;
            }
        }
        self.block(&c.body, self.group)?;
        Ok(self.steps)
    }
}
