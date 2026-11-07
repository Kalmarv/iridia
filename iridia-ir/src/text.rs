//! The text form of a function, for fixtures, diffs and notes: one line per node, indented by construct, values as
//! `%n`, variables by name, labels as `'n`. The first line names the schema and its version, bumped when the form
//! changes. [`parse`] reads it back.

use alloc::string::String;
use core::fmt::Write;

use crate::{
    Address, Barrier, Binary, Body, Builtin, End, Function, GatherOffset, Level, Memory, Node, Op,
    OutOfBounds, Precision, Query, Region, Res, Scalar, Scope, Space, Target, Terminator, Ternary, Type,
    Unary, ValueId, Var, Wave,
};

mod parse;

pub use parse::{TextError, parse};

/// The schema line every text form starts with.
pub const SCHEMA: &str = "iridia-ir 1";

fn list(out: &mut String, vs: &[ValueId]) {
    for (i, v) in vs.iter().enumerate() {
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}%{}", v.0);
    }
}

fn lane(l: u8) -> char {
    match l {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        3 => 'w',
        _ => '?',
    }
}

/// The lanes of a mask by name, in ascending order: `xz` for `0b0101`.
fn lanes(mask: u8) -> String {
    (0..8u8).filter(|n| mask >> n & 1 != 0).map(lane).collect()
}

/// A literal's lanes in the type's own notation: a float that is not finite as its exact bits (`0x7fc00000`), a
/// finite one as the shortest decimal that gives its bits back, an integer as a decimal.
fn literal(out: &mut String, ty: Type, bits: &[u64; 4]) {
    let lanes = usize::from(ty.lanes);
    let _ = write!(out, "{ty}(");
    for (i, b) in bits.iter().take(lanes).enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = match ty.scalar {
            Scalar::F32 => {
                let x = f32::from_bits(u32::try_from(*b).unwrap_or(0));
                if x.is_finite() {
                    write!(out, "{x:?}")
                } else {
                    write!(out, "0x{b:08x}")
                }
            }
            Scalar::F64 => {
                let x = f64::from_bits(*b);
                if x.is_finite() {
                    write!(out, "{x:?}")
                } else {
                    write!(out, "0x{b:016x}")
                }
            }
            Scalar::Bool => write!(out, "{}", *b != 0),
            s if s.is_signed() => {
                // sign-extend from the type's width
                let shift = 64u32.saturating_sub(s.bits());
                write!(out, "{}", (b.cast_signed() << shift) >> shift)
            }
            _ => write!(out, "{b}"),
        };
    }
    out.push(')');
}

fn unary(u: &Unary) -> String {
    match u {
        Unary::FRound(r) => alloc::format!("fround.{r:?}"),
        Unary::Saturate(n) => alloc::format!("saturate.{n:?}"),
        Unary::ClampRange { lo, hi, nan } => {
            alloc::format!(
                "clamp.{nan:?}({:?},{:?})",
                f32::from_bits(*lo),
                f32::from_bits(*hi)
            )
        }
        Unary::Quantize { frac_bits, rounding } => alloc::format!("quantize.{frac_bits}.{rounding:?}"),
        Unary::Derivative(axis, fineness) => alloc::format!("deriv.{axis:?}.{fineness:?}"),
        other => alloc::format!("{other:?}"),
    }
    .to_lowercase()
}

fn binary(b: &Binary) -> String {
    match b {
        Binary::FMul(z) => alloc::format!("fmul.{z:?}"),
        Binary::FMin(n) => alloc::format!("fmin.{n:?}"),
        Binary::FMax(n) => alloc::format!("fmax.{n:?}"),
        Binary::FCmp(c) => alloc::format!("fcmp.{c:?}"),
        Binary::SCmp(c) => alloc::format!("scmp.{c:?}"),
        Binary::UCmp(c) => alloc::format!("ucmp.{c:?}"),
        Binary::UDiv(z) => alloc::format!("udiv.{z:?}"),
        Binary::URem(z) => alloc::format!("urem.{z:?}"),
        Binary::SDiv(z) => alloc::format!("sdiv.{z:?}"),
        Binary::SRem(z) => alloc::format!("srem.{z:?}"),
        other => alloc::format!("{other:?}"),
    }
    .to_lowercase()
}

fn ternary(t: &Ternary) -> &'static str {
    match t {
        Ternary::Fma => "fma",
        Ternary::BitExtract { signed: false } => "bitextract.unsigned",
        Ternary::BitExtract { signed: true } => "bitextract.signed",
    }
}

fn oob(o: OutOfBounds) -> &'static str {
    match o {
        OutOfBounds::Zero => "oob.zero",
        OutOfBounds::Undef => "oob.undef",
    }
}

fn builtin(b: Builtin) -> String {
    alloc::format!("{b:?}").to_lowercase()
}

/// A wave operation's name in the text form: `wave.ballot`, `wave.add.exclusive`.
pub(crate) fn wave(w: &Wave) -> String {
    match w {
        Wave::Ballot(_) => "wave.ballot".into(),
        Wave::First(_) => "wave.first".into(),
        Wave::Lane(..) => "wave.lane".into(),
        Wave::QuadLane(..) => "wave.quadlane".into(),
        Wave::QuadSwap(_, s) => alloc::format!("wave.quadswap.{}", alloc::format!("{s:?}").to_lowercase()),
        Wave::Collective { op, scan, .. } => alloc::format!(
            "wave.{}.{}",
            alloc::format!("{op:?}").to_lowercase(),
            alloc::format!("{scan:?}").to_lowercase()
        ),
    }
}

/// `rN`, an element of an array `rN@%i`, a non-uniform one `rN@%i!`.
fn res(r: &Res) -> String {
    match r.index {
        None => alloc::format!("r{}", r.id.0),
        Some(i) => alloc::format!("r{}@%{}{}", r.id.0, i.0, if r.nonuniform { "!" } else { "" }),
    }
}

/// `[%i * stride] + offset`, or `+ offset` for a fixed address.
fn address(a: &Address) -> String {
    match a.index {
        Some(i) => alloc::format!("[%{} * {}] + {}", i.0, a.stride, a.offset),
        None => alloc::format!(" + {}", a.offset),
    }
}

fn offsets(o: &[i8; 3]) -> String {
    alloc::format!("({},{},{})", o[0], o[1], o[2])
}

fn var_name(f: &Function, v: crate::VarId) -> &str {
    f.vars.get(v.0 as usize).map_or("?", |d| d.name.as_str())
}

fn memory(f: &Function, m: &Memory) -> String {
    match m {
        Memory::Buffer(r, a) => alloc::format!("buffer {}{}", res(r), address(a)),
        Memory::Image(r, c) => alloc::format!("image {}, %{}", res(r), c.0),
        Memory::Var(v, None) => alloc::format!("var {}", var_name(f, *v)),
        Memory::Var(v, Some(i)) => alloc::format!("var {}[%{}]", var_name(f, *v), i.0),
    }
}

/// The text of a barrier: `barrier` and what it waits for.
fn barrier(b: &Barrier) -> String {
    let mut out = String::from("barrier");
    if b.execution {
        out.push_str(" execution");
    }
    if b.workgroup {
        out.push_str(" workgroup");
    }
    match b.storage {
        Some(Scope::Workgroup) => out.push_str(" storage.workgroup"),
        Some(Scope::Device) => out.push_str(" storage.device"),
        None => {}
    }
    out
}

/// A variable's declaration after `var `: `name: T`, an array `name: [T; n]`, in workgroup memory `... workgroup`, a
/// task payload `... payload`, a ray's payload `... raypayload`, a hit's attributes `... attributes`, with a start
/// `... = {T(..), ...}`.
fn declaration(out: &mut String, v: &Var) {
    let _ = match v.len {
        None => write!(out, "{}: {}", v.name, v.ty),
        Some(n) => write!(out, "{}: [{}; {n}]", v.name, v.ty),
    };
    match v.space {
        Space::Private => {}
        Space::Workgroup => out.push_str(" workgroup"),
        Space::Payload => out.push_str(" payload"),
        Space::RayPayload => out.push_str(" raypayload"),
        Space::HitAttributes => out.push_str(" attributes"),
    }
    if !v.init.is_empty() {
        out.push_str(" = {");
        for (i, e) in v.init.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            literal(out, v.ty, e);
        }
        out.push('}');
    }
}

fn op(out: &mut String, f: &Function, ty: Type, o: &Op) {
    let _ = match o {
        Op::Literal(bits) => {
            literal(out, ty, bits);
            Ok(())
        }
        Op::Undef => write!(out, "undef"),
        Op::Param => write!(out, "param"),
        Op::Unary(u, a) => write!(out, "{} %{}", unary(u), a.0),
        Op::Binary(b, x, y) => write!(out, "{} %{}, %{}", binary(b), x.0, y.0),
        Op::Select { cond, accept, reject } => {
            write!(out, "select %{}, %{}, %{}", cond.0, accept.0, reject.0)
        }
        Op::Compose(parts) => {
            out.push_str("compose ");
            list(out, parts);
            Ok(())
        }
        Op::Swizzle(a, lanes) => {
            let names = lanes.iter().take(usize::from(ty.lanes)).map(|&l| lane(l));
            write!(out, "%{}.{}", a.0, names.collect::<String>())
        }
        Op::Splat(a) => write!(out, "splat %{}", a.0),
        Op::Dot(x, y) => write!(out, "dot %{}, %{}", x.0, y.0),
        Op::Convert(a, rule) => write!(
            out,
            "convert.{} %{}",
            alloc::format!("{rule:?}").to_lowercase(),
            a.0
        ),
        Op::Bitcast(a) => write!(out, "bitcast %{}", a.0),
        Op::LoadInput(slot, mask) => write!(out, "input {slot}.{}", lanes(*mask)),
        Op::ReportHit { t, kind, attributes } => write!(
            out,
            "reporthit %{} kind %{} attributes {}",
            t.0,
            kind.0,
            var_name(f, *attributes)
        ),
        Op::LoadVar(v) => match f.vars.get(v.0 as usize) {
            Some(var) => write!(out, "load {}", var.name),
            None => write!(out, "load ?{}", v.0),
        },
        Op::LoadState(r) => write!(out, "state r{}", r.0),
        Op::LoadBuffer {
            resource,
            address: a,
            oob: rule,
        } => write!(out, "buffer {}{} {}", res(resource), address(a), oob(*rule)),
        Op::Sample {
            texture,
            sampler,
            coord,
            level,
            compare,
            offset,
            clamp,
        } => {
            let _ = write!(out, "sample {}, {}, %{}", res(texture), res(sampler), coord.0);
            let _ = match level {
                Level::Auto => Ok(()),
                Level::Zero => write!(out, " level 0"),
                Level::Bias(x) => write!(out, " bias %{}", x.0),
                Level::Exact(x) => write!(out, " level %{}", x.0),
                Level::Grad(x, y) => write!(out, " grad %{}, %{}", x.0, y.0),
            };
            if let Some(c) = compare {
                let _ = write!(out, " compare %{}", c.0);
            }
            if *offset != [0; 3] {
                let _ = write!(out, " offset {}", offsets(offset));
            }
            match clamp {
                Some(c) => write!(out, " clamp %{}", c.0),
                None => Ok(()),
            }
        }
        Op::Ternary(t, a, b, c) => write!(out, "{} %{}, %{}, %{}", ternary(t), a.0, b.0, c.0),
        Op::BitInsert {
            base,
            insert,
            offset,
            count,
        } => write!(
            out,
            "bitinsert %{}, %{}, %{}, %{}",
            base.0, insert.0, offset.0, count.0
        ),
        Op::LoadInputAt(slot, mask, at) => write!(out, "input {slot}.{} [%{}]", lanes(*mask), at.0),
        Op::LoadOutputAt(slot, mask, at) => write!(out, "output {slot}.{} [%{}]", lanes(*mask), at.0),
        Op::LoadPatch(slot, mask) => write!(out, "patch {slot}.{}", lanes(*mask)),
        Op::Builtin(b) => write!(out, "builtin {}", builtin(*b)),
        Op::LoadElement(v, i, rule) => write!(out, "load {}[%{}] {}", var_name(f, *v), i.0, oob(*rule)),
        Op::Gather {
            texture,
            sampler,
            coord,
            component,
            compare,
            offset,
        } => {
            let _ = write!(
                out,
                "gather.{} {}, {}, %{}",
                lane(*component),
                res(texture),
                res(sampler),
                coord.0
            );
            if let Some(c) = compare {
                let _ = write!(out, " compare %{}", c.0);
            }
            match offset {
                GatherOffset::Fixed(o) if *o == [0; 3] => Ok(()),
                GatherOffset::Fixed(o) => write!(out, " offset {}", offsets(o)),
                GatherOffset::Dynamic(v) => write!(out, " offset %{}", v.0),
            }
        }
        Op::LoadTexel {
            texture,
            coord,
            level,
            sample,
            offset,
        } => {
            let _ = write!(out, "texel {}, %{}", res(texture), coord.0);
            if let Some(l) = level {
                let _ = write!(out, " level %{}", l.0);
            }
            if let Some(x) = sample {
                let _ = write!(out, " sample %{}", x.0);
            }
            if *offset != [0; 3] {
                let _ = write!(out, " offset {}", offsets(offset));
            }
            Ok(())
        }
        Op::Query(r, q) => match q {
            Query::Size(None) => write!(out, "query.size {}", res(r)),
            Query::Size(Some(l)) => write!(out, "query.size {} level %{}", res(r), l.0),
            Query::Layers => write!(out, "query.layers {}", res(r)),
            Query::Levels => write!(out, "query.levels {}", res(r)),
            Query::Samples => write!(out, "query.samples {}", res(r)),
            Query::SamplePosition(i) => write!(out, "query.sampleposition {} index %{}", res(r), i.0),
            Query::Elements => write!(out, "query.elements {}", res(r)),
        },
        Op::Atomic {
            memory: m,
            op: aop,
            value,
            compare,
        } => {
            let _ = write!(
                out,
                "atomic.{} {} value %{}",
                alloc::format!("{aop:?}").to_lowercase(),
                memory(f, m),
                value.0
            );
            match compare {
                Some(c) => write!(out, " compare %{}", c.0),
                None => Ok(()),
            }
        }
        Op::Counter(r, c) => write!(
            out,
            "counter.{} {}",
            alloc::format!("{c:?}").to_lowercase(),
            res(r)
        ),
        Op::Wave(w) => {
            let _ = write!(out, "{} ", wave(w));
            list(out, &o.operands());
            Ok(())
        }
    };
}

struct Printer<'a> {
    f: &'a Function,
    out: String,
    depth: usize,
}

impl Printer<'_> {
    fn indent(&mut self) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
    }
    fn value(&mut self, v: ValueId) {
        self.indent();
        match self.f.value(v) {
            Some(d) => {
                let pp = if d.precision == Precision::AtLeast16 {
                    " pp"
                } else {
                    ""
                };
                let _ = write!(self.out, "%{}: {}{pp} = ", v.0, d.ty);
                op(&mut self.out, self.f, d.ty, &d.op);
            }
            None => {
                let _ = write!(self.out, "%{} = ?", v.0);
            }
        }
        self.out.push('\n');
    }
    fn params(&mut self, vs: &[ValueId]) {
        for (i, v) in vs.iter().enumerate() {
            let sep = if i == 0 { "" } else { ", " };
            let ty = self.f.ty(*v).map(|t| alloc::format!("{t}")).unwrap_or_default();
            let _ = write!(self.out, "{sep}%{}: {ty}", v.0);
        }
    }
    fn region(&mut self, r: &Region) {
        self.depth = self.depth.saturating_add(1);
        for n in &r.nodes {
            self.node(n);
        }
        self.indent();
        let o = &mut self.out;
        match &r.end {
            None => o.push_str("<open>"),
            Some(End::Yield(vs)) => {
                o.push_str("yield ");
                list(o, vs);
            }
            Some(End::Br(l, vs)) => {
                let _ = write!(o, "br '{} ", l.0);
                list(o, vs);
            }
            Some(End::Continue(l, vs)) => {
                let _ = write!(o, "continue '{} ", l.0);
                list(o, vs);
            }
            Some(End::Return) => o.push_str("return"),
            Some(End::Kill) => o.push_str("kill"),
            Some(End::IgnoreHit) => o.push_str("ignorehit"),
            Some(End::AcceptHitAndEndSearch) => o.push_str("accepthit"),
            Some(End::Unreachable) => o.push_str("unreachable"),
        }
        let trimmed = o.trim_end().len();
        o.truncate(trimmed);
        o.push('\n');
        self.depth = self.depth.saturating_sub(1);
    }
    fn close(&mut self, results: &[ValueId]) {
        self.indent();
        self.out.push('}');
        if !results.is_empty() {
            self.out.push_str(" -> ");
            self.params(results);
        }
        self.out.push('\n');
    }
    fn node(&mut self, n: &Node) {
        match n {
            Node::Value(v) => self.value(*v),
            Node::StoreVar(var, v) => {
                self.indent();
                let name = self.f.vars.get(var.0 as usize).map_or("?", |d| d.name.as_str());
                let _ = writeln!(self.out, "{name} = %{}", v.0);
            }
            Node::StoreOutput(slot, mask, v) => {
                self.indent();
                let _ = writeln!(self.out, "output {slot}.{} = %{}", lanes(*mask), v.0);
            }
            Node::Demote => {
                self.indent();
                self.out.push_str("demote\n");
            }
            Node::StoreElement(var, i, v, rule) => {
                self.indent();
                let name = var_name(self.f, *var);
                let _ = writeln!(self.out, "{name}[%{}] = %{} {}", i.0, v.0, oob(*rule));
            }
            Node::StoreBuffer {
                resource,
                address: a,
                value,
            } => {
                self.indent();
                let _ = writeln!(
                    self.out,
                    "store buffer {}{} = %{}",
                    res(resource),
                    address(a),
                    value.0
                );
            }
            Node::StoreTexel { image, coord, value } => {
                self.indent();
                let _ = writeln!(
                    self.out,
                    "store texel {}, %{} = %{}",
                    res(image),
                    coord.0,
                    value.0
                );
            }
            Node::StorePatch(slot, mask, v) => {
                self.indent();
                let _ = writeln!(self.out, "patch {slot}.{} = %{}", lanes(*mask), v.0);
            }
            Node::Barrier(b) => {
                self.indent();
                let _ = writeln!(self.out, "{}", barrier(b));
            }
            Node::Emit(stream) => {
                self.indent();
                let _ = writeln!(self.out, "emit {stream}");
            }
            Node::Cut(stream) => {
                self.indent();
                let _ = writeln!(self.out, "cut {stream}");
            }
            Node::SetMeshCounts { vertices, primitives } => {
                self.indent();
                let _ = writeln!(self.out, "mesh counts %{}, %{}", vertices.0, primitives.0);
            }
            Node::StoreVertex {
                vertex,
                slot,
                mask,
                value,
            } => {
                self.indent();
                let _ = writeln!(
                    self.out,
                    "store vertex %{} {slot}.{} = %{}",
                    vertex.0,
                    lanes(*mask),
                    value.0
                );
            }
            Node::StoreIndices { primitive, indices } => {
                self.indent();
                let _ = writeln!(self.out, "store indices %{} = %{}", primitive.0, indices.0);
            }
            Node::DispatchMesh { groups, payload } => {
                self.indent();
                let name = self
                    .f
                    .vars
                    .get(payload.0 as usize)
                    .map_or("?", |d| d.name.as_str());
                let _ = writeln!(self.out, "dispatch mesh %{}, {name}", groups.0);
            }
            Node::TraceRay { scene, ray, payload } => {
                self.indent();
                let name = self
                    .f
                    .vars
                    .get(payload.0 as usize)
                    .map_or("?", |d| d.name.as_str());
                let _ = writeln!(
                    self.out,
                    "trace ray {} flags %{} mask %{} offset %{} stride %{} miss %{} origin %{} tmin %{} \
                     direction %{} tmax %{} payload {name}",
                    res(scene),
                    ray.flags.0,
                    ray.mask.0,
                    ray.hit_group_offset.0,
                    ray.hit_group_stride.0,
                    ray.miss_index.0,
                    ray.origin.0,
                    ray.t_min.0,
                    ray.direction.0,
                    ray.t_max.0
                );
            }
            Node::CallShader { index, parameter } => {
                self.indent();
                let name = self
                    .f
                    .vars
                    .get(parameter.0 as usize)
                    .map_or("?", |d| d.name.as_str());
                let _ = writeln!(self.out, "call shader %{} parameter {name}", index.0);
            }
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => {
                self.indent();
                let _ = writeln!(self.out, "switch '{} %{} {{", label.0, selector.0);
                for c in cases {
                    self.indent();
                    self.out.push_str("case ");
                    for (i, v) in c.values.iter().enumerate() {
                        let sep = if i == 0 { "" } else { ", " };
                        let _ = write!(self.out, "{sep}{v}");
                    }
                    self.out.push_str(":\n");
                    self.region(&c.body);
                }
                self.indent();
                self.out.push_str("default:\n");
                self.region(default);
                self.close(results);
            }
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                self.indent();
                let _ = writeln!(self.out, "if %{} {{", cond.0);
                self.region(accept);
                self.indent();
                self.out.push_str("} else {\n");
                self.region(reject);
                self.close(results);
            }
            Node::Block { label, body, results } => {
                self.indent();
                let _ = writeln!(self.out, "block '{} {{", label.0);
                self.region(body);
                self.close(results);
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                self.indent();
                let _ = write!(self.out, "loop '{} (", label.0);
                self.params(params);
                self.out.push_str(") = (");
                list(&mut self.out, inits);
                self.out.push_str(") {\n");
                self.region(body);
                self.close(results);
            }
        }
    }
    fn target(&mut self, t: &Target) {
        let _ = write!(self.out, "b{}(", t.block);
        list(&mut self.out, &t.args);
        self.out.push(')');
    }
}

/// The text form of a function.
pub fn print(f: &Function) -> String {
    let mut p = Printer {
        f,
        out: String::new(),
        depth: 0,
    };
    let _ = writeln!(p.out, "{SCHEMA}\nfn {} {{", f.name);
    for var in &f.vars {
        p.out.push_str("  var ");
        declaration(&mut p.out, var);
        p.out.push('\n');
    }
    match &f.body {
        Body::Regions(r) => p.region(r),
        Body::Cfg(blocks) => {
            for (i, b) in blocks.iter().enumerate() {
                let _ = write!(p.out, "  b{i}(");
                p.params(&b.params);
                p.out.push_str("):\n");
                p.depth = 2;
                for n in &b.nodes {
                    p.node(n);
                }
                p.indent();
                match &b.term {
                    Terminator::Jump(t) => {
                        p.out.push_str("jump ");
                        p.target(t);
                    }
                    Terminator::Branch { cond, accept, reject } => {
                        let _ = write!(p.out, "branch %{}, ", cond.0);
                        p.target(accept);
                        p.out.push_str(", ");
                        p.target(reject);
                    }
                    Terminator::Switch {
                        selector,
                        cases,
                        default,
                    } => {
                        let _ = write!(p.out, "switch %{}", selector.0);
                        for (v, t) in cases {
                            let _ = write!(p.out, ", {v}: ");
                            p.target(t);
                        }
                        p.out.push_str(", default: ");
                        p.target(default);
                    }
                    Terminator::Return => p.out.push_str("return"),
                    Terminator::Kill => p.out.push_str("kill"),
                    Terminator::IgnoreHit => p.out.push_str("ignorehit"),
                    Terminator::AcceptHitAndEndSearch => p.out.push_str("accepthit"),
                    Terminator::Unreachable => p.out.push_str("unreachable"),
                }
                p.out.push('\n');
                p.depth = 0;
            }
        }
    }
    p.out.push_str("}\n");
    p.out
}
