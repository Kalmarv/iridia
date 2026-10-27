//! A typed builder over naga's IR for lifters. Every value it hands out carries its scalar kind and width, so a lifter
//! never asks naga's typifier and a bitcast or splat is explicit where it is needed. Nothing here knows a bytecode;
//! [`behaviour`] holds D3D's results spelled out as operations.

mod atomic;
pub mod behaviour;
mod entry;
mod mesh;
pub mod structure;

pub use entry::{EntryDecl, FunctionDecl};
pub use mesh::MeshOutput;

use naga::{
    BinaryOperator, Block, Expression, Function, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Scalar, ScalarKind, Span, Statement, SwizzleComponent, Type, TypeInner,
    UnaryOperator, VectorSize,
};

/// The scalar kind of a typed value: unsigned, signed, float or bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Unsigned 32-bit integer.
    U,
    /// Signed 32-bit integer.
    I,
    /// 32-bit float.
    F,
    /// Boolean.
    B,
}
impl Kind {
    /// The naga scalar this kind is carried as (32 bits wide; a bool is a bool).
    pub fn scalar(self) -> Scalar {
        match self {
            Kind::U => Scalar::U32,
            Kind::I => Scalar::I32,
            Kind::F => Scalar::F32,
            Kind::B => Scalar::BOOL,
        }
    }
    /// The naga scalar kind of [`Kind::scalar`].
    pub fn kind(self) -> ScalarKind {
        self.scalar().kind
    }
}

/// How many lanes a value has: one (a scalar) to four. There is no other width, so every type and vector the builder
/// is asked for is one naga has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    /// A scalar.
    One = 1,
    /// A vector of two lanes.
    Two = 2,
    /// A vector of three lanes.
    Three = 3,
    /// A vector of four lanes.
    Four = 4,
}
impl Width {
    /// Every width, from one lane to four.
    pub const ALL: [Width; 4] = [Width::One, Width::Two, Width::Three, Width::Four];

    /// The width of `lanes` lanes, `None` unless it is 1 to 4.
    pub const fn new(lanes: u8) -> Option<Width> {
        match lanes {
            1 => Some(Width::One),
            2 => Some(Width::Two),
            3 => Some(Width::Three),
            4 => Some(Width::Four),
            _ => None,
        }
    }

    /// The width of a count of lanes, `None` unless it is 1 to 4.
    pub fn of_count(lanes: usize) -> Option<Width> {
        u8::try_from(lanes).ok().and_then(Width::new)
    }

    /// The width of a naga vector of `size`.
    pub const fn of_vector(size: VectorSize) -> Width {
        match size {
            VectorSize::Bi => Width::Two,
            VectorSize::Tri => Width::Three,
            VectorSize::Quad => Width::Four,
        }
    }

    /// The number of lanes, 1 to 4.
    pub const fn get(self) -> u8 {
        self as u8
    }

    /// The number of lanes, 1 to 4, as a count.
    pub const fn count(self) -> usize {
        self as usize
    }

    /// The naga vector size of the width, `None` for a scalar.
    pub const fn vector_size(self) -> Option<VectorSize> {
        match self {
            Width::One => None,
            Width::Two => Some(VectorSize::Bi),
            Width::Three => Some(VectorSize::Tri),
            Width::Four => Some(VectorSize::Quad),
        }
    }

    /// The lanes of the width, 0 to one less than it.
    pub fn lanes(self) -> std::ops::Range<u8> {
        0..self.get()
    }
}
impl From<Width> for u8 {
    fn from(w: Width) -> u8 {
        w.get()
    }
}
impl From<Width> for u32 {
    fn from(w: Width) -> u32 {
        u32::from(w.get())
    }
}
impl From<Width> for usize {
    fn from(w: Width) -> usize {
        w.count()
    }
}
impl From<VectorSize> for Width {
    fn from(size: VectorSize) -> Width {
        Width::of_vector(size)
    }
}
impl TryFrom<u8> for Width {
    type Error = String;
    fn try_from(lanes: u8) -> Result<Width, String> {
        Width::new(lanes).ok_or_else(|| format!("a width of {lanes} lanes (a value has 1 to 4)"))
    }
}
impl TryFrom<u32> for Width {
    type Error = String;
    fn try_from(lanes: u32) -> Result<Width, String> {
        u8::try_from(lanes)
            .ok()
            .and_then(Width::new)
            .ok_or_else(|| format!("a width of {lanes} lanes (a value has 1 to 4)"))
    }
}
impl TryFrom<usize> for Width {
    type Error = String;
    fn try_from(lanes: usize) -> Result<Width, String> {
        Width::of_count(lanes).ok_or_else(|| format!("a width of {lanes} lanes (a value has 1 to 4)"))
    }
}
impl std::fmt::Display for Width {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get())
    }
}

/// An expression with its kind and width.
#[derive(Clone, Copy, Debug)]
pub struct Val {
    /// The expression holding the value.
    pub expr: Handle<Expression>,
    /// The scalar kind of every lane.
    pub kind: Kind,
    /// The number of lanes.
    pub width: Width,
}

/// `offset` rounded up to a multiple of `align`, or the reason it does not fit in 32 bits.
pub(crate) fn align_up(offset: u32, align: u32) -> Result<u32, String> {
    offset.checked_next_multiple_of(align).ok_or_else(|| {
        format!("a layout offset of {offset} bytes aligned to {align} is over 4294967295 bytes")
    })
}

/// `value`, or the reason a layout size does not fit in 32 bits.
pub(crate) fn fits(value: Option<u32>) -> Result<u32, String> {
    value.ok_or_else(|| "a layout of over 4294967295 bytes".to_owned())
}

/// The swizzle component for lane 0 (x) to 3 (w); any larger lane is w.
pub fn component(lane: u8) -> SwizzleComponent {
    match lane {
        0 => SwizzleComponent::X,
        1 => SwizzleComponent::Y,
        2 => SwizzleComponent::Z,
        _ => SwizzleComponent::W,
    }
}

/// A module and the entry function being built, with a stack of the blocks open for structured control flow.
///
/// What a lifter reads decides the shapes of the values it hands the builder, so an operation asked of values that cannot
/// do it (a vector of 2 lanes added to one of 3) is not a bug of the builder and does not panic: it records the first such
/// fault and returns a value of the shape asked for. The contract is the lifter's: it calls [`Builder::check`] where it
/// can name what it was reading (after each instruction or statement), and turns a fault into a refusal. `finish`,
/// `finish_mesh`, `finish_functions` and `end_function` give back a fault nobody checked, or control flow left open, as an
/// error: a lifter that follows the contract does not see one from its input, but a lifter's own bug is an error to
/// refuse the program with, not a panic.
#[derive(Debug)]
pub struct Builder {
    /// The module being built.
    pub module: Module,
    /// The entry function being built (its body is assembled from [`Builder::blocks`] when the module is finished).
    pub function: Function,
    /// The blocks being filled, innermost last; statements go to the last.
    pub blocks: Vec<Block>,
    /// A compute entry point's thread group size (set before [`Builder::finish`]; 0 for the other stages).
    pub workgroup_size: [u32; 3],
    /// The first operation a lifter asked for that the values it was given cannot be (a vector of 2 lanes added to one
    /// of 3, a lane that is not there): the lifter's input is at fault, not the builder, so the operation hands back a
    /// value of the shape it was asked for and the fault waits here until the lifter asks ([`Builder::check`]).
    error: Option<String>,
}
impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
impl Builder {
    /// A builder over an empty module with one open block.
    pub fn new() -> Self {
        Self {
            module: Module::default(),
            function: Function::default(),
            blocks: vec![Block::new()],
            workgroup_size: [0; 3],
            error: None,
        }
    }

    /// Record that an operation was asked of values that cannot do it; the first is kept.
    fn fail(&mut self, detail: impl Into<String>) {
        self.error.get_or_insert_with(|| detail.into());
    }

    /// The first fault of the operations since the last check, which a lifter turns into a refusal of the program at the
    /// place it is reading. A builder with a fault unchecked cannot be finished: that is the lifter's bug.
    pub fn check(&mut self) -> Result<(), String> {
        self.error.take().map_or(Ok(()), Err)
    }

    /// Whether the builder can be finished: its control flow closed, and no fault left unchecked.
    fn finishable(&mut self) -> Result<(), String> {
        if self.blocks.len() != 1 {
            return Err(format!(
                "control flow left open: {} blocks open where the function's own is the one (a lifter bug)",
                self.blocks.len()
            ));
        }
        self.check()
            .map_err(|fault| format!("a fault the lifter did not check before finishing: {fault}"))
    }

    /// The type `inner` in the module (types equal to an earlier one are shared).
    pub fn ty(&mut self, inner: TypeInner) -> Handle<Type> {
        self.module
            .types
            .insert(Type { name: None, inner }, Span::UNDEFINED)
    }
    /// The type of a value of `kind` and `width` lanes: a scalar for one.
    pub fn ty_of(&mut self, kind: Kind, width: Width) -> Handle<Type> {
        match width.vector_size() {
            None => self.ty(TypeInner::Scalar(kind.scalar())),
            Some(size) => self.ty(TypeInner::Vector {
                size,
                scalar: kind.scalar(),
            }),
        }
    }

    /// Append an expression to the function; one that is not a pre-emitted kind (a literal, an argument, a pointer)
    /// is emitted in the current block.
    pub fn app(&mut self, e: Expression) -> Handle<Expression> {
        let pre = e.needs_pre_emit();
        let start = self.function.expressions.len();
        let h = self.function.expressions.append(e, Span::UNDEFINED);
        if !pre {
            let range = self.function.expressions.range_from(start);
            self.push(Statement::Emit(range));
        }
        h
    }
    /// The type of an element of a typed buffer the program reaches with atomic operations: four atomic words.
    pub fn atomic_quad(&mut self, scalar: naga::Scalar) -> Handle<Type> {
        let atomic = self.ty(TypeInner::Atomic(scalar));
        let members = (0..4u32)
            .map(|k| naga::StructMember {
                name: Some(format!("w{k}")),
                ty: atomic,
                binding: None,
                offset: k.saturating_mul(4),
            })
            .collect();
        self.ty(TypeInner::Struct { members, span: 16 })
    }
    /// A result expression (an atomic's old value): made before the statement that fills it, covered by no emit.
    pub fn result_expr(&mut self, e: Expression) -> Handle<Expression> {
        self.function.expressions.append(e, Span::UNDEFINED)
    }
    /// `if condition { then }`: what `then` appends goes in the accept block.
    pub fn when(&mut self, condition: Val, then: impl FnOnce(&mut Self)) {
        self.blocks.push(Block::new());
        then(self);
        let accept = self.blocks.pop().expect("the block just pushed");
        self.push(Statement::If {
            condition: condition.expr,
            accept,
            reject: Block::new(),
        });
    }
    /// Open `for i in 0..count { .. }` as a loop (not unrolled): what is appended until [`Self::end_repeat`] is its
    /// body, and the value returned is `i`, read at the top of each pass.
    pub fn begin_repeat(&mut self, count: u32) -> (Val, Handle<LocalVariable>) {
        let ty = self.ty_of(Kind::U, Width::One);
        let counter = self.local("i", ty);
        let ptr = self.local_ptr(counter);
        let zero = self.lit_u(0);
        self.store(ptr, zero);
        self.blocks.push(Block::new());
        let i = self.load(ptr, Kind::U, Width::One);
        let limit = self.lit_u(count);
        let done = self.binary(BinaryOperator::GreaterEqual, i, limit);
        self.when(done, |b| b.push(Statement::Break));
        (i, counter)
    }
    /// Close the loop [`Self::begin_repeat`] opened: count the pass and go round.
    pub fn end_repeat(&mut self, counter: Handle<LocalVariable>) {
        let ptr = self.local_ptr(counter);
        let again = self.load(ptr, Kind::U, Width::One);
        let one = self.lit_u(1);
        let next = self.binary(BinaryOperator::Add, again, one);
        self.store(ptr, next);
        let inner = self.blocks.pop().expect("the loop body opened by begin_repeat");
        self.push(Statement::Loop {
            body: inner,
            continuing: Block::new(),
            break_if: None,
        });
    }
    /// Append `s` to the innermost open block.
    pub fn push(&mut self, s: Statement) {
        self.blocks
            .last_mut()
            .expect("a block is always open")
            .push(s, Span::UNDEFINED);
    }
    /// A new local variable of the function, named `name` and uninitialised.
    pub fn local(&mut self, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        self.function.local_variables.append(
            LocalVariable {
                name: Some(name.to_owned()),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        )
    }
    /// A pointer expression to the local variable `l`.
    pub fn local_ptr(&mut self, l: Handle<LocalVariable>) -> Handle<Expression> {
        self.app(Expression::LocalVariable(l))
    }
    /// A pointer expression to the global variable `g`.
    pub fn global_ptr(&mut self, g: Handle<GlobalVariable>) -> Handle<Expression> {
        self.app(Expression::GlobalVariable(g))
    }
    /// A load through `ptr`, typed by the `kind` and `width` the caller states the pointee has.
    pub fn load(&mut self, ptr: Handle<Expression>, kind: Kind, width: Width) -> Val {
        Val {
            expr: self.app(Expression::Load { pointer: ptr }),
            kind,
            width,
        }
    }
    /// Store `v` through `ptr`.
    pub fn store(&mut self, ptr: Handle<Expression>, v: Val) {
        self.push(Statement::Store {
            pointer: ptr,
            value: v.expr,
        });
    }
    /// Add the global variable `g` to the module.
    pub fn global(&mut self, g: GlobalVariable) -> Handle<GlobalVariable> {
        self.module.global_variables.append(g, Span::UNDEFINED)
    }
    /// An expression of the module's constant arena (an initialiser).
    pub fn global_expr(&mut self, e: Expression) -> Handle<Expression> {
        self.module.global_expressions.append(e, Span::UNDEFINED)
    }

    // literals
    /// An unsigned integer literal.
    pub fn lit_u(&mut self, x: u32) -> Val {
        self.lit(Literal::U32(x), Kind::U)
    }
    /// A signed integer literal.
    pub fn lit_i(&mut self, x: i32) -> Val {
        self.lit(Literal::I32(x), Kind::I)
    }
    /// A float literal. naga holds only finite ones, so a NaN or an infinity is the bits of one, reinterpreted.
    pub fn lit_f(&mut self, x: f32) -> Val {
        if x.is_finite() {
            return self.lit(Literal::F32(x), Kind::F);
        }
        let bits = self.lit_u(x.to_bits());
        self.bitcast(bits, Kind::F)
    }
    /// A boolean literal.
    pub fn lit_b(&mut self, x: bool) -> Val {
        self.lit(Literal::Bool(x), Kind::B)
    }
    fn lit(&mut self, l: Literal, kind: Kind) -> Val {
        Val {
            expr: self.app(Expression::Literal(l)),
            kind,
            width: Width::One,
        }
    }
    /// The literal of `kind` whose 32 bits are `bits` (a bool is whether they are nonzero).
    pub fn lit_bits(&mut self, kind: Kind, bits: u32) -> Val {
        match kind {
            Kind::U => self.lit_u(bits),
            Kind::I => self.lit_i(bits as i32),
            Kind::F => self.lit_f(f32::from_bits(bits)),
            Kind::B => self.lit_b(bits != 0),
        }
    }

    /// `v` repeated across `width` lanes; `v` itself when it already has that many. A `v` that is not a scalar
    /// records a fault ([`Builder::check`]) and gives `v` back.
    pub fn splat(&mut self, v: Val, width: Width) -> Val {
        if v.width == width {
            return v;
        }
        let Some(size) = width.vector_size().filter(|_| v.width == Width::One) else {
            self.fail(format!("a value of {} lanes where {width} are needed", v.width));
            return v;
        };
        Val {
            expr: self.app(Expression::Splat { size, value: v.expr }),
            kind: v.kind,
            width,
        }
    }
    /// A vector (or the scalar itself, for one part) from scalars of one kind.
    pub fn compose(&mut self, parts: &[Val]) -> Val {
        if let [only] = parts {
            return *only;
        }
        let Some(first) = parts.first() else {
            self.fail("a vector of no lanes");
            return self.lit_u(0);
        };
        let Some(width) = Width::of_count(parts.len()).filter(|_| {
            parts
                .iter()
                .all(|p| p.width == Width::One && p.kind == first.kind)
        }) else {
            self.fail(format!(
                "a vector of {} parts that are not scalars of one kind",
                parts.len()
            ));
            return *first;
        };
        let kind = first.kind;
        let ty = self.ty_of(kind, width);
        Val {
            expr: self.app(Expression::Compose {
                ty,
                components: parts.iter().map(|p| p.expr).collect(),
            }),
            kind,
            width,
        }
    }
    /// Lane `i` of a vector.
    pub fn lane(&mut self, v: Val, i: u8) -> Val {
        if i >= v.width.get() {
            self.fail(format!("lane {i} of a value of {} lanes", v.width));
            return v;
        }
        if v.width == Width::One {
            return v;
        }
        Val {
            expr: self.app(Expression::AccessIndex {
                base: v.expr,
                index: u32::from(i),
            }),
            kind: v.kind,
            width: Width::One,
        }
    }
    /// Lanes `lanes` of a vector, in that order, repeats allowed; a scalar is splat.
    pub fn swizzle(&mut self, v: Val, lanes: &[u8]) -> Val {
        if let [one] = lanes {
            return self.lane(v, *one);
        }
        let Some(width) = Width::of_count(lanes.len()).filter(|_| lanes.iter().all(|l| *l < v.width.get()))
        else {
            self.fail(format!("lanes {lanes:?} of a value of {} lanes", v.width));
            return v;
        };
        if v.width == Width::One {
            return self.splat(v, width);
        }
        let Some(size) = width.vector_size() else {
            // one lane is `lane`'s
            return self.lane(v, lanes.first().copied().unwrap_or(0));
        };
        let mut pattern = [SwizzleComponent::X; 4];
        for (slot, lane) in pattern.iter_mut().zip(lanes) {
            *slot = component(*lane);
        }
        Val {
            expr: self.app(Expression::Swizzle {
                size,
                vector: v.expr,
                pattern,
            }),
            kind: v.kind,
            width,
        }
    }
    /// Reinterpret the bits as another kind (u32, i32 or f32), width unchanged.
    pub fn bitcast(&mut self, v: Val, kind: Kind) -> Val {
        if v.kind == kind {
            return v;
        }
        Val {
            expr: self.app(Expression::As {
                expr: v.expr,
                kind: kind.kind(),
                convert: None,
            }),
            kind,
            width: v.width,
        }
    }
    /// The value as `kind`: a bitcast between u32, i32 and f32; a bool becomes a word of all ones or zero (D3D's
    /// true and false) and a word a bool by being nonzero.
    pub fn bitcast_to(&mut self, v: Val, kind: Kind) -> Val {
        match (v.kind, kind) {
            (a, b) if a == b => v,
            (Kind::B, _) => {
                let ones = self.lit_u(u32::MAX);
                let zero = self.lit_u(0);
                let word = self.select(v, ones, zero);
                self.bitcast(word, kind)
            }
            (_, Kind::B) => {
                let word = self.bitcast(v, Kind::U);
                let zero = self.lit_u(0);
                self.binary(BinaryOperator::NotEqual, word, zero)
            }
            _ => self.bitcast(v, kind),
        }
    }
    /// Convert the value to another kind (`itof`, `utof`, float to integer) with naga's own semantics; D3D's differ
    /// for float to integer and are [`behaviour`].
    pub fn convert(&mut self, v: Val, kind: Kind) -> Val {
        Val {
            expr: self.app(Expression::As {
                expr: v.expr,
                kind: kind.kind(),
                convert: Some(4),
            }),
            kind,
            width: v.width,
        }
    }
    /// A unary operation; the result has `v`'s kind and width.
    pub fn unary(&mut self, op: UnaryOperator, v: Val) -> Val {
        Val {
            expr: self.app(Expression::Unary { op, expr: v.expr }),
            kind: v.kind,
            width: v.width,
        }
    }
    /// A binary operation; a scalar operand is splat to the other's width, a comparison gives a bool.
    pub fn binary(&mut self, op: BinaryOperator, a: Val, b: Val) -> Val {
        let width = a.width.max(b.width);
        let (a, b) = (self.splat(a, width), self.splat(b, width));
        let kind = match op {
            BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::Less
            | BinaryOperator::LessEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterEqual => Kind::B,
            _ => a.kind,
        };
        Val {
            expr: self.app(Expression::Binary {
                op,
                left: a.expr,
                right: b.expr,
            }),
            kind,
            width,
        }
    }
    /// A math function of `args` (a scalar among vectors is splat to their width). The result is a scalar for `Dot`, a
    /// `u32` for `Pack2x16float`, two floats for `Unpack2x16float` and otherwise has the first argument's kind and the
    /// widest width. No arguments records a fault ([`Builder::check`]).
    pub fn math(&mut self, fun: MathFunction, args: &[Val]) -> Val {
        let Some(width) = args.iter().map(|a| a.width).max() else {
            self.fail("a math function of no arguments");
            return self.lit_u(0);
        };
        let args: Vec<Val> = args.iter().map(|a| self.splat(*a, width)).collect();
        let Some(first) = args.first().copied() else {
            self.fail("a math function of no arguments");
            return self.lit_u(0);
        };
        let kind = first.kind;
        let arg = |i: usize| args.get(i).map(|a| a.expr);
        let (width, kind) = match fun {
            MathFunction::Dot => (Width::One, kind),
            MathFunction::Pack2x16float => (Width::One, Kind::U),
            MathFunction::Unpack2x16float => (Width::Two, Kind::F),
            _ => (width, kind),
        };
        Val {
            expr: self.app(Expression::Math {
                fun,
                arg: first.expr,
                arg1: arg(1),
                arg2: arg(2),
                arg3: arg(3),
            }),
            kind,
            width,
        }
    }
    /// `select(reject, accept, cond)`: a scalar condition is splat to the operands' width.
    pub fn select(&mut self, cond: Val, accept: Val, reject: Val) -> Val {
        let width = accept.width.max(reject.width).max(cond.width);
        let (accept, reject) = (self.splat(accept, width), self.splat(reject, width));
        let cond = self.splat(cond, width);
        Val {
            expr: self.app(Expression::Select {
                condition: cond.expr,
                accept: accept.expr,
                reject: reject.expr,
            }),
            kind: accept.kind,
            width,
        }
    }
    /// A comparison result as D3D's mask: all ones where true, else zero.
    pub fn mask_of(&mut self, cond: Val) -> Val {
        let ones = self.lit_u(u32::MAX);
        let zero = self.lit_u(0);
        self.select(cond, ones, zero)
    }

    /// Close the function being built as the module's function `name` with these arguments and result, and start an
    /// empty one: a program of a stage with no entry point is several of them (see [`crate::stage_functions`]).
    pub fn end_function(&mut self, name: &str, decl: FunctionDecl) -> Result<Handle<naga::Function>, String> {
        self.finishable()?;
        let mut function = std::mem::take(&mut self.function);
        function.name = Some(name.to_owned());
        function.arguments = decl.arguments;
        function.result = decl.result.map(|ty| naga::FunctionResult { ty, binding: None });
        function.body = self.blocks.first_mut().map(std::mem::take).unwrap_or_default();
        Ok(self.module.functions.append(function, Span::UNDEFINED))
    }

    /// Declare a function with no body, the module's name for an operation the pipeline does (see [`crate::rt`]); one
    /// that returns a value returns `false`.
    pub fn declare_operation(
        &mut self,
        name: &str,
        arguments: Vec<naga::FunctionArgument>,
        returns_bool: bool,
    ) -> Handle<naga::Function> {
        let mut function = Function {
            name: Some(name.to_owned()),
            arguments,
            ..Function::default()
        };
        if returns_bool {
            let ty = self.ty_of(Kind::B, Width::One);
            function.result = Some(naga::FunctionResult { ty, binding: None });
            let value = function
                .expressions
                .append(Expression::Literal(naga::Literal::Bool(false)), Span::UNDEFINED);
            function
                .body
                .push(Statement::Return { value: Some(value) }, Span::UNDEFINED);
        }
        self.module.functions.append(function, Span::UNDEFINED)
    }

    /// Finish a module that has functions and no entry point; an error where the builder cannot be finished (a fault
    /// nobody checked, control flow left open).
    pub fn finish_functions(mut self) -> Result<Module, String> {
        self.finishable()?;
        Ok(self.module)
    }

    /// Finish a module whose one entry point `name` runs in `stage` with these `arguments` and `result`;
    /// `early_depth_test` forces early depth testing. The entry's workgroup size is [`Builder::workgroup_size`]. An
    /// error where the builder cannot be finished: a fault nobody checked ([`Builder::check`]) or control flow left open,
    /// both a lifter's own bugs.
    pub fn finish(
        mut self,
        name: &str,
        stage: naga::ShaderStage,
        arguments: Vec<naga::FunctionArgument>,
        result: Option<Handle<Type>>,
        early_depth_test: bool,
    ) -> Result<Module, String> {
        self.finishable()?;
        let mut function = self.function;
        function.name = Some(name.to_owned());
        function.arguments = arguments;
        function.result = result.map(|ty| naga::FunctionResult { ty, binding: None });
        function.body = self.blocks.pop().unwrap_or_default();
        self.module.entry_points.push(naga::EntryPoint {
            name: name.to_owned(),
            stage,
            early_depth_test: early_depth_test.then_some(naga::EarlyDepthTest::Force),
            workgroup_size: self.workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        Ok(self.module)
    }
}
