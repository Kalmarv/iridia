//! Operations. Each states what it gives for every input, including NaN, infinities, signed zero and out-of-range
//! values; where sources' rules differ the operation carries a mandatory semantic enum naming the behaviour (never a
//! source). Target legalization expands what a target lacks, in one place (`iridia-passes`).

use alloc::vec::Vec;

use crate::interface::ResourceId;

/// A value of a function: an index into its value arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ValueId(pub u32);

/// A mutable variable of a function (a source register, a local array), until `promote` turns it into values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct VarId(pub u32);

/// A label of a `Regions` construct (`Block`, `Loop`) that exits name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct LabelId(pub u32);

/// What `FMin` and `FMax` give when an operand is NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MinMaxNan {
    /// IEEE 754-2008 minNum/maxNum: one NaN operand gives the other operand; two give NaN (D3D10 and later).
    MinNum,
    /// Any NaN operand gives NaN.
    Propagate,
    /// Either operand or NaN: the source leaves it undefined (GLSL, WGSL).
    Any,
}

/// What `Saturate` gives for NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum SaturateNan {
    /// NaN gives 0 (D3D).
    Zero,
    /// Any value in 0..1, or NaN: undefined by the source.
    Any,
}

/// How `FMul` treats a zero operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MulZero {
    /// IEEE 754: `0 * inf` is NaN, signs multiply.
    Ieee,
    /// Zero times anything, infinity and NaN included, is +0 (D3D9's legacy multiply, a compatibility policy).
    ZeroWins,
}

/// What an integer division (`UDiv`, `SDiv`) and remainder (`URem`, `SRem`) give for a zero divisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum OnZero {
    /// All ones (D3D's `udiv`, for quotient and remainder).
    AllOnes,
    /// As if the divisor were one: the dividend for a quotient, 0 for a remainder (WGSL's rule).
    ByOne,
    /// Undefined (SPIR-V, GLSL).
    Undef,
}

/// How a float becomes an integer, out of range and for NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum FloatToInt {
    /// Truncate toward zero, saturate to the integer type's range, NaN gives 0 (D3D's `ftoi`/`ftou`).
    Saturate,
    /// Truncate toward zero; out of range and NaN undefined.
    Undef,
}

/// A float rounding to an integral value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Rounding {
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Toward zero.
    Trunc,
    /// To the nearest, ties to even.
    NearestEven,
    /// To the nearest, ties away from zero (D3D9 `mova`).
    NearestAway,
}

/// The screen axis a derivative is taken along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Axis {
    /// Along x (`ddx`, `dsx`, `dpdx`).
    X,
    /// Along y.
    Y,
}

/// Which neighbours a derivative is taken between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Fineness {
    /// Once per 2x2 quad: every invocation of the quad gets the same difference.
    Coarse,
    /// Per invocation, between it and its neighbour in its row or column of the quad.
    Fine,
    /// Either: the source does not say (D3D9's `dsx` and `dsy`, WGSL's `dpdx`).
    Any,
}

/// A comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Cmp {
    /// Equal (ordered for floats: false with a NaN).
    Eq,
    /// Not equal (unordered for floats: true with a NaN).
    Ne,
    /// Less than (ordered).
    Lt,
    /// At most (ordered).
    Le,
    /// Greater than (ordered).
    Gt,
    /// At least (ordered).
    Ge,
}

/// A one-operand operation; the result has the operand's type unless noted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Unary {
    /// Float negation (flips the sign bit, NaN included).
    FNeg,
    /// Float absolute value (clears the sign bit).
    FAbs,
    /// `x - floor(x)`.
    FFract,
    /// Rounding to an integral float.
    FRound(Rounding),
    /// `sqrt(x)`, correctly rounded is not required (WGSL's precision).
    FSqrt,
    /// `1 / sqrt(x)`, at the precision WGSL's `inverseSqrt` states.
    FInverseSqrt,
    /// `2^x`.
    FExp2,
    /// `log2(x)`; 0 gives -inf, negative values NaN.
    FLog2,
    /// `sin(x)`, at WGSL's precision.
    FSin,
    /// `cos(x)`, at WGSL's precision.
    FCos,
    /// `tan(x)`, at WGSL's precision.
    FTan,
    /// `asin(x)`; outside -1..1 NaN.
    FAsin,
    /// `acos(x)`; outside -1..1 NaN.
    FAcos,
    /// `atan(x)`.
    FAtan,
    /// `sinh(x)`.
    FSinh,
    /// `cosh(x)`.
    FCosh,
    /// `tanh(x)`.
    FTanh,
    /// -1, 0 or 1 by the sign of a float (signed zero gives itself, NaN gives 0).
    FSign,
    /// Clamp to 0..1, with the stated NaN rule; below zero, -0 among it, gives +0.
    Saturate(SaturateNan),
    /// Clamp to `lo..=hi`, the bounds' f32 bits (finite, `lo <= hi`), with the stated NaN rule; `Zero` needs 0 within
    /// the range. A source's register range (D3D9 `ps_1_x` constants within -1..1 and colours within 0..1).
    ClampRange {
        /// The lower bound's bits.
        lo: u32,
        /// The upper bound's bits.
        hi: u32,
        /// What NaN gives.
        nan: SaturateNan,
    },
    /// Fixed point of `frac_bits` fractional bits: `round(x * 2^frac_bits) / 2^frac_bits` with the rounding, NaN and
    /// infinities kept (early D3D9 `ps_1_x` hardware, as a compatibility policy).
    Quantize {
        /// The fractional bits kept, 0 to 23.
        frac_bits: u8,
        /// How the scaled value rounds.
        rounding: Rounding,
    },
    /// The derivative of a float along a screen axis, between the invocations of a 2x2 quad (pixel programs only).
    /// Where the quad's invocations diverged it is undefined.
    Derivative(Axis, Fineness),
    /// Boolean not, or the bitwise complement of an integer.
    Not,
    /// Integer negation (wrapping).
    INeg,
    /// Whether any lane of a bool vector is true; the result is a `bool`.
    Any,
    /// Whether every lane of a bool vector is true; the result is a `bool`.
    All,
    /// Whether a float is NaN, by its bits; the result is bool of the operand's lanes.
    IsNan,
    /// Whether a float is an infinity; the result is bool of the operand's lanes.
    IsInf,
    /// The number of bits set in an integer.
    CountOnes,
    /// An integer's bits in reverse order.
    BitReverse,
    /// The number of bits above the highest set bit of a 32-bit integer (`31 - index`), or all ones where none is set
    /// (D3D's `firstbit_hi`).
    FirstBitHigh,
    /// The number of bits above the highest bit of a 32-bit integer that differs from its sign bit, or all ones where
    /// none does (0 and -1; D3D's `firstbit_shi`).
    FirstBitHighSigned,
    /// The index of the lowest set bit of a 32-bit integer, or all ones where none is set (D3D's `firstbit_lo`).
    FirstBitLow,
    /// An `f32` as the bits of the nearest binary16 in the low 16 bits of a `u32`, the high bits zero. Which way a value
    /// between two halves rounds is the implementation's (to nearest even, or toward zero); NaN stays a NaN, a value
    /// beyond the range an infinity, and a binary16 denormal is kept.
    F32ToF16,
    /// The binary16 in the low 16 bits of a `u32` (the high bits ignored) as the `f32` of the same value; the result
    /// type is `f32` of the operand's lanes.
    F16ToF32,
}

/// A two-operand operation on operands of one type; the result has their type unless noted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Binary {
    /// Float addition.
    FAdd,
    /// Float subtraction.
    FSub,
    /// Float multiplication with the stated zero rule.
    FMul(MulZero),
    /// Float division (IEEE: `x / 0` is a signed infinity or NaN).
    FDiv,
    /// The float remainder of a division truncated toward zero, exact: `x - y * q` for the integer `q` of `x / y`
    /// truncated, with the sign of `x` (C's `fmod`, LLVM's and DXIL's `frem`, SPIR-V's `OpFRem`). `y = 0` or an
    /// infinite `x` gives NaN, an infinite `y` gives `x`.
    FRem,
    /// Float minimum with the stated NaN rule.
    FMin(MinMaxNan),
    /// Float maximum with the stated NaN rule.
    FMax(MinMaxNan),
    /// `x^y` for `x >= 0`; undefined for negative `x`, and for `x = 0` with `y <= 0`.
    FPow,
    /// Float comparison; the result is bool of the operands' lanes.
    FCmp(Cmp),
    /// Signed integer comparison; the result is bool.
    SCmp(Cmp),
    /// Unsigned integer comparison (also equality of any integers and bools); the result is bool.
    UCmp(Cmp),
    /// Integer addition, wrapping.
    IAdd,
    /// Integer subtraction, wrapping.
    ISub,
    /// Integer multiplication, wrapping: the low half of the product.
    IMul,
    /// The high half of the product of two unsigned integers.
    UMulHigh,
    /// The high half of the product of two signed integers.
    SMulHigh,
    /// The carry out of an unsigned addition: 1 where `a + b` wraps, else 0.
    UAddCarry,
    /// The borrow of an unsigned subtraction: 1 where `a < b`, else 0.
    USubBorrow,
    /// Unsigned division with the stated zero rule.
    UDiv(OnZero),
    /// Unsigned remainder with the stated zero rule.
    URem(OnZero),
    /// Signed division truncated toward zero with the stated zero rule (the most negative value divided by -1 wraps
    /// to itself).
    SDiv(OnZero),
    /// Signed remainder with the sign of the dividend, with the stated zero rule (the most negative value by -1 gives
    /// 0).
    SRem(OnZero),
    /// Signed minimum.
    SMin,
    /// Signed maximum.
    SMax,
    /// Unsigned minimum.
    UMin,
    /// Unsigned maximum.
    UMax,
    /// Bitwise and (logical and of bools).
    And,
    /// Bitwise or (logical or of bools).
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Shift left; the count (an unsigned integer of the operand's lanes) is taken modulo the bit width.
    Shl,
    /// Logical shift right, the count taken modulo the bit width.
    ShrU,
    /// Arithmetic shift right (the sign bit copied in), the count taken modulo the bit width.
    ShrS,
}

/// A three-operand operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Ternary {
    /// `a * b + c` rounded once (fused), of floats of one type.
    Fma,
    /// `(a, offset, count)`: `count` bits of the integer `a` from bit `offset`, both counts taken modulo 32; where
    /// `offset + count` reaches past bit 31 the field is the bits from `offset` up; a count of 0 gives 0. Unsigned
    /// zero-extends the field, signed extends its top bit (D3D's `ubfe`, `ibfe`). The counts are `u32` of `a`'s lanes.
    BitExtract {
        /// Whether the field is sign-extended.
        signed: bool,
    },
}

/// The level of detail a sample is taken at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Level {
    /// From the coordinate's derivatives (pixel programs only).
    Auto,
    /// From the derivatives, plus a bias (`f32`).
    Bias(ValueId),
    /// An explicit level (`f32`).
    Exact(ValueId),
    /// Level zero.
    Zero,
    /// From explicit derivatives of the coordinate along x and y, each of the coordinate's type (without an array
    /// layer).
    Grad(ValueId, ValueId),
}

/// What a read outside its memory gives, and what a write there does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum OutOfBounds {
    /// A read gives zero and a write has no effect (D3D9 relative constants outside the bank, D3D10+ buffer reads).
    Zero,
    /// Undefined.
    Undef,
}

/// A resource an operation names: one of the interface's, and for an array of resources the element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct Res {
    /// The resource.
    pub id: ResourceId,
    /// For an array of resources (`count` other than 1), the element: a `u32` or `i32` value; `None` is element 0.
    pub index: Option<ValueId>,
    /// The index may differ between the invocations of a wave or quad (D3D's `NonUniformResourceIndex`): a
    /// constraint on the lowering, which must not assume it uniform.
    pub nonuniform: bool,
}

impl Res {
    /// The resource itself, not an element of an array of them.
    pub const fn of(id: ResourceId) -> Res {
        Res {
            id,
            index: None,
            nonuniform: false,
        }
    }
}

/// A byte address in a buffer: `offset + stride * index`, in wrapping 32-bit arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Address {
    /// The dynamic index (`u32` or `i32`), or none for a fixed address.
    pub index: Option<ValueId>,
    /// The bytes between elements.
    pub stride: u32,
    /// The byte offset of element 0.
    pub offset: u32,
}

impl Address {
    /// A fixed byte address.
    pub const fn fixed(offset: u32) -> Address {
        Address {
            index: None,
            stride: 0,
            offset,
        }
    }
    /// A dynamic byte address: `index` bytes.
    pub const fn bytes(index: ValueId) -> Address {
        Address {
            index: Some(index),
            stride: 1,
            offset: 0,
        }
    }
}

/// The memory an atomic operation reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Memory {
    /// The word of a buffer at a byte address (its low two bits ignored).
    Buffer(Res, Address),
    /// The texel of a storage image at an integer coordinate (an array layer as its last lane), or the element of a
    /// typed storage buffer (the element as the coordinate): its first lane, a single-lane 32-bit integer view.
    Image(Res, ValueId),
    /// An element of a variable that is an array (`Some(index)`), or a variable that is not.
    Var(VarId, Option<ValueId>),
}

/// An atomic read-modify-write; each gives the value the memory held before.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum AtomicOp {
    /// Wrapping addition.
    Add,
    /// Wrapping subtraction.
    Sub,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Signed minimum.
    SMin,
    /// Signed maximum.
    SMax,
    /// Unsigned minimum.
    UMin,
    /// Unsigned maximum.
    UMax,
    /// Store the value.
    Exchange,
    /// Store the value where the memory equals the comparand.
    CompareExchange,
}

/// The hidden counter of a structured unordered-access buffer (D3D's append and consume buffers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum CounterOp {
    /// Add one; gives the count before (`IncrementCounter`, `imm_atomic_alloc`).
    Increment,
    /// Subtract one; gives the count after (`DecrementCounter`, `imm_atomic_consume`).
    Decrement,
}

/// Where the offset of a gather comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GatherOffset {
    /// A texel offset fixed in the program, per coordinate lane (-8 to 7).
    Fixed([i8; 3]),
    /// A texel offset computed by the program: an `i32` vector of the coordinate's lanes (without a layer), each lane
    /// taken in its low six bits as a signed number (-32 to 31).
    Dynamic(ValueId),
}

/// Direct3D's standard sample positions in sixteenths of a pixel from its centre (x right, y down), for 1, 2, 4, 8 and
/// 16 samples one after the other: the positions of a count start at `count - 1` (D3D11's
/// `D3D11_STANDARD_MULTISAMPLE_PATTERN`).
const STANDARD_POSITIONS: [(i8, i8); 31] = [
    (0, 0),
    (4, 4),
    (-4, -4),
    (-2, -6),
    (6, -2),
    (-6, 2),
    (2, 6),
    (1, -3),
    (-1, 3),
    (5, 1),
    (-3, -5),
    (-5, 5),
    (-7, -1),
    (3, 7),
    (7, -7),
    (1, 1),
    (-1, -3),
    (-3, 2),
    (4, -1),
    (-5, -2),
    (2, 5),
    (5, 3),
    (3, -5),
    (-2, 6),
    (0, -7),
    (-4, -6),
    (-6, 4),
    (-8, 0),
    (7, -4),
    (6, 7),
    (-7, -8),
];

/// The position of sample `index` of a texture of `count` samples in Direct3D's standard pattern, in pixels from the
/// pixel's centre (x right, y down): `None` for an index past the count, or a count other than 1, 2, 4, 8 or 16.
pub fn standard_sample_position(count: u32, index: u32) -> Option<[f32; 2]> {
    if !matches!(count, 1 | 2 | 4 | 8 | 16) || index >= count {
        return None;
    }
    let row = usize::try_from(count.checked_sub(1)?.checked_add(index)?).ok()?;
    let (x, y) = *STANDARD_POSITIONS.get(row)?;
    Some([f32::from(x) / 16.0, f32::from(y) / 16.0])
}

/// What a resource query asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Query {
    /// The width (and height, and depth) of a texture's level (a `u32` value; none for a multisampled texture or a
    /// storage image), as `u32` lanes (a cube's faces' width and height); 0 in every lane for a level the texture
    /// does not have.
    Size(Option<ValueId>),
    /// The array layers of an arrayed texture, `u32`.
    Layers,
    /// The mip levels of a texture, `u32`.
    Levels,
    /// The samples per texel of a multisampled texture, `u32`.
    Samples,
    /// The position of sample `index` (a `u32`) within the pixel, relative to its centre (x right, y down), in
    /// Direct3D's standard pattern for the texture's sample count ([`standard_sample_position`]), `vec2<f32>`; (0, 0)
    /// for an index past the texture's samples or a count with no standard pattern.
    SamplePosition(ValueId),
    /// The elements of a buffer (its bytes for a raw buffer), `u32`.
    Elements,
}

/// What an [`Op::Wave`] does across the active invocations of a wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Wave {
    /// Which active invocations' condition (a `bool`) holds: a `vec4<u32>` whose word `n / 32` holds invocation
    /// `n`'s bit `n % 32` (Direct3D's `WaveActiveBallot`).
    Ballot(ValueId),
    /// The value the active invocation of the lowest number has (`WaveReadLaneFirst`).
    First(ValueId),
    /// The value invocation `.1` (a `u32`, which may differ between invocations) has; undefined where that
    /// invocation is not active (`WaveReadLaneAt`).
    Lane(ValueId, ValueId),
    /// The value invocation `.1` (a `u32`, 0 to 3) of the invocation's 2x2 quad has (`QuadReadLaneAt`).
    QuadLane(ValueId, ValueId),
    /// The value the invocation's neighbour in its quad has (`QuadReadAcrossX`, `QuadReadAcrossY`,
    /// `QuadReadAcrossDiagonal`).
    QuadSwap(ValueId, QuadSwap),
    /// The operation over the active invocations' values: of all of them, or of those numbered up to this one
    /// (`WaveActiveSum`, `WavePrefixSum`, `WaveActiveBitAnd`, `WaveActiveAllTrue`).
    Collective {
        /// The operation.
        op: WaveOp,
        /// Which invocations' values it takes.
        scan: WaveScan,
        /// The value.
        value: ValueId,
    },
}

/// The neighbour a [`Wave::QuadSwap`] reads, by its place in the quad.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum QuadSwap {
    /// The invocation beside it, across x.
    X,
    /// The invocation above or below it, across y.
    Y,
    /// The invocation across the diagonal.
    Diagonal,
}

/// The operation of a [`Wave::Collective`]. A minimum or maximum compares as the value's type does (signed for
/// `i32`); `And`, `Or` and `Xor` take integers, `All` and `Any` bools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum WaveOp {
    /// The sum.
    Add,
    /// The product.
    Mul,
    /// The least.
    Min,
    /// The greatest.
    Max,
    /// The bitwise and.
    And,
    /// The bitwise or.
    Or,
    /// The bitwise exclusive or.
    Xor,
    /// Whether every value holds.
    All,
    /// Whether any value holds.
    Any,
}

/// The invocations whose values a [`Wave::Collective`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum WaveScan {
    /// Every active invocation's.
    Reduce,
    /// The active invocations' numbered up to this one, this one's too.
    Inclusive,
    /// The active invocations' numbered below this one (the operation's identity where there are none).
    Exclusive,
}

/// A value the pipeline gives a stage that is not one of its interface's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Builtin {
    /// The output control point a hull program's main function computes, `u32`.
    OutputControlPoint,
    /// The patch (hull, domain) or input primitive (geometry) being processed, `u32`.
    PrimitiveId,
    /// The instance of a geometry program, `u32`.
    GsInstance,
    /// The location of a domain program's point in the patch's domain, `vec3<f32>` (barycentric for triangles, `uv`
    /// and 0 for quads, `u` and 0 and 0 for isolines).
    DomainLocation,
    /// The phase instance of a hull program's patch constant phase (D3D's fork and join phase instance), `u32`.
    PhaseInstance,
}

/// What a barrier waits for and makes visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Barrier {
    /// Every invocation of the workgroup reaches it before any goes on.
    pub execution: bool,
    /// Writes to workgroup memory before it are visible to the workgroup after it.
    pub workgroup: bool,
    /// Writes to buffers and images before it are visible to the workgroup after it (`device` to every invocation).
    pub storage: Option<Scope>,
}

/// How far a memory operation's effect is ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Scope {
    /// The workgroup.
    Workgroup,
    /// The device: every invocation of the dispatch or draw.
    Device,
}

/// An operation that defines a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Op {
    /// A constant: each lane's exact bits, in the low bits of its word (a bool is 0 or 1).
    Literal([u64; 4]),
    /// An undefined value.
    Undef,
    /// A value given by its construct: a `Loop`'s parameter, a construct's result, a `Cfg` block's parameter.
    Param,
    /// A one-operand operation.
    Unary(Unary, ValueId),
    /// A two-operand operation.
    Binary(Binary, ValueId, ValueId),
    /// A three-operand operation.
    Ternary(Ternary, ValueId, ValueId, ValueId),
    /// `count` bits of `insert` placed into `base` from bit `offset` (both counts taken modulo 32), the field cut at
    /// bit 31: `(base & !field) | ((insert << offset) & field)` with `field = ((1 << count) - 1) << offset` (D3D's
    /// `bfi`). Integers of one type; the counts `u32` of their lanes.
    BitInsert {
        /// The value the field is put into.
        base: ValueId,
        /// The value whose low bits are the field.
        insert: ValueId,
        /// The field's lowest bit.
        offset: ValueId,
        /// The field's width.
        count: ValueId,
    },
    /// `cond ? accept : reject`, lane by lane when `cond` is a vector, else whole.
    Select {
        /// A bool, or a bool vector of the result's lanes.
        cond: ValueId,
        /// The value where `cond` holds.
        accept: ValueId,
        /// The value where it does not.
        reject: ValueId,
    },
    /// A vector of the lanes of the parts, in order (scalars and vectors of the result's scalar type).
    Compose(Vec<ValueId>),
    /// The result's lanes taken from the operand's lanes `lanes[0..result lanes]`.
    Swizzle(ValueId, [u8; 4]),
    /// Every lane of the result is the scalar operand.
    Splat(ValueId),
    /// The float dot product of two vectors of one type, a scalar: `a0*b0 + a1*b1 + ...`, left to right, each step
    /// rounded (no fused multiply-add).
    Dot(ValueId, ValueId),
    /// A numeric conversion between scalar types of the same lane count. Float to integer follows `FloatToInt`;
    /// integer to float rounds to nearest even.
    Convert(ValueId, FloatToInt),
    /// The same bits as another type of the same total size.
    Bitcast(ValueId),
    /// Lanes of an interface input (`interface.inputs[slot]`): the lanes of mask `.1` (bit n is lane n, at least one,
    /// within the varying's width), in ascending order, as a value of the varying's scalar type.
    LoadInput(u32, u8),
    /// Lanes of an input of one vertex or control point (`u32` or `i32` value `.2`) of a stage whose inputs are
    /// arrays (geometry, hull, domain); an index past the stage's points is undefined.
    LoadInputAt(u32, u8, ValueId),
    /// Lanes of output slot `.0` of one output control point (`.2`), read by a hull program's patch constant function.
    LoadOutputAt(u32, u8, ValueId),
    /// Lanes of patch constant slot `.0` (the domain facts' or hull facts' `patch_constants`): read by a domain
    /// program, and by a hull program's patch constant function after it stored them.
    LoadPatch(u32, u8),
    /// A value the pipeline gives that is not an interface input.
    Builtin(Builtin),
    /// The value a variable holds (a variable that is not an array).
    LoadVar(VarId),
    /// The value element `.1` (a `u32` or `i32`) of an array variable holds, with the stated rule for an index past
    /// its end.
    LoadElement(VarId, ValueId, OutOfBounds),
    /// The value of a pipeline state resource (a [`crate::interface::ResourceKind::PipelineState`]), as its
    /// [`crate::interface::StateParam::value`].
    LoadState(ResourceId),
    /// The result's lanes as consecutive 32-bit words of a buffer (a constant buffer, a raw, structured or storage
    /// buffer) from the byte address (its low two bits ignored); each word past the buffer's size by the rule.
    LoadBuffer {
        /// The buffer.
        resource: Res,
        /// The byte address of the first word.
        address: Address,
        /// What a read past the resource's size (or at a negative index) gives.
        oob: OutOfBounds,
    },
    /// A filtered sample of a texture through a sampler: `vec4` of the texture's sample type; one `f32` for a depth
    /// texture, and for a comparison (the fraction of the footprint that passed).
    Sample {
        /// The texture.
        texture: Res,
        /// The sampler.
        sampler: Res,
        /// The coordinate (`f32`): one lane for 1D, two for 2D, three for cube and volume, and the array layer after
        /// them for an arrayed texture.
        coord: ValueId,
        /// The level of detail.
        level: Level,
        /// The reference value (`f32`) of a depth comparison.
        compare: Option<ValueId>,
        /// A texel offset fixed in the program, per coordinate lane (-8 to 7; zero past the texture's dimensions).
        offset: [i8; 3],
        /// The lowest level of detail (the most detailed) the sample may use, `f32`.
        clamp: Option<ValueId>,
    },
    /// The four texels a bilinear sample would read, one lane of each (`component`), in D3D's order (the texel at
    /// (-, +), (+, +), (+, -), (-, -) of the footprint's centre): `vec4` of the sample type (`f32` for a depth texture),
    /// or each texel's comparison result (`vec4<f32>`).
    Gather {
        /// The texture.
        texture: Res,
        /// The sampler.
        sampler: Res,
        /// The coordinate, as for a sample.
        coord: ValueId,
        /// The lane gathered, 0 to 3.
        component: u8,
        /// The reference value of a depth comparison.
        compare: Option<ValueId>,
        /// The texel offset.
        offset: GatherOffset,
    },
    /// One texel by integer coordinate, no filtering: of a texture (`vec4` of its sample type, one `f32` for a depth
    /// texture), a storage image (`vec4` of its lanes' type) or a typed buffer (`vec4` of its element's type, the
    /// element as the coordinate). A texel outside the resource gives zero.
    LoadTexel {
        /// The texture, image or buffer.
        texture: Res,
        /// The coordinate (`i32` or `u32`): a lane per dimension and the array layer after them.
        coord: ValueId,
        /// The mip level (`i32` or `u32`) of a texture that has levels.
        level: Option<ValueId>,
        /// The sample (`i32` or `u32`) of a multisampled texture.
        sample: Option<ValueId>,
        /// A texel offset fixed in the program (-8 to 7).
        offset: [i8; 3],
    },
    /// A fact of a resource, as [`Query`] says.
    Query(Res, Query),
    /// An atomic read-modify-write of `value` (a 32- or 64-bit integer; of an image's or typed buffer's lane type) into
    /// the memory: the value it held before. Outside the memory (a word past a buffer's end, a texel outside an image,
    /// an element past an array's end) it changes nothing and gives zero. A demoted invocation's changes nothing (its
    /// result is then undefined). The operation is the instruction's own effect, kept even where nothing reads its
    /// result.
    Atomic {
        /// What it reaches.
        memory: Memory,
        /// The operation.
        op: AtomicOp,
        /// The operand.
        value: ValueId,
        /// The comparand of a compare-exchange.
        compare: Option<ValueId>,
    },
    /// An atomic change of a buffer's hidden counter (a `u32`), as [`CounterOp`] says.
    Counter(Res, CounterOp),
    /// An intersection program reports a hit of the ray at a distance (Direct3D's `ReportHit`), with the attributes the
    /// hit's programs read: whether the hit was accepted (a `bool`). A distance outside the ray's interval is not a hit;
    /// the any-hit program of the hit's group decides the rest. The report is the instruction's own effect, kept even
    /// where nothing reads its result.
    ReportHit {
        /// The distance along the ray (`f32`).
        t: ValueId,
        /// The hit kind the hit's programs read (`u32`, at most 127 for one the program states).
        kind: ValueId,
        /// The attributes: `u32` elements laid out as [`crate::Space::HitAttributes`] is.
        attributes: VarId,
    },
    /// An operation across the invocations of the invocation's wave (Direct3D's wave, a subgroup), as [`Wave`] says.
    /// Which invocations make a wave, and how many, is the device's. An invocation takes part where it is active:
    /// where its control flow reaches the operation with the others' (a demoted invocation, a helper, is not active);
    /// a quad's invocations are those of its 2x2 pixels in a pixel program, four consecutive invocations in a compute
    /// program.
    Wave(Wave),
}

impl Res {
    fn operands_mut<'a>(&'a mut self, out: &mut Vec<&'a mut ValueId>) {
        out.extend(self.index.as_mut());
    }
}

impl Op {
    /// The values the operation reads, each place it reads one, in operand order (an array element's index before
    /// the rest).
    pub fn operands_mut(&mut self) -> Vec<&mut ValueId> {
        let mut v: Vec<&mut ValueId> = Vec::new();
        match self {
            Op::Literal(_)
            | Op::Undef
            | Op::Param
            | Op::LoadInput(..)
            | Op::LoadPatch(..)
            | Op::Builtin(_)
            | Op::LoadVar(_)
            | Op::LoadState(_) => {}
            Op::Unary(_, a)
            | Op::Swizzle(a, _)
            | Op::Splat(a)
            | Op::Convert(a, _)
            | Op::Bitcast(a)
            | Op::LoadInputAt(_, _, a)
            | Op::LoadOutputAt(_, _, a)
            | Op::LoadElement(_, a, _) => v.push(a),
            Op::Binary(_, a, b) | Op::Dot(a, b) => {
                v.push(a);
                v.push(b);
            }
            Op::Ternary(_, a, b, c) => v.extend([a, b, c]),
            Op::BitInsert {
                base,
                insert,
                offset,
                count,
            } => v.extend([base, insert, offset, count]),
            Op::Select { cond, accept, reject } => v.extend([cond, accept, reject]),
            Op::Compose(parts) => v.extend(parts.iter_mut()),
            Op::LoadBuffer {
                resource, address, ..
            } => {
                resource.operands_mut(&mut v);
                v.extend(address.index.as_mut());
            }
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                clamp,
                ..
            } => {
                texture.operands_mut(&mut v);
                sampler.operands_mut(&mut v);
                v.push(coord);
                match level {
                    Level::Bias(x) | Level::Exact(x) => v.push(x),
                    Level::Grad(x, y) => v.extend([x, y]),
                    Level::Auto | Level::Zero => {}
                }
                v.extend(compare.as_mut());
                v.extend(clamp.as_mut());
            }
            Op::Gather {
                texture,
                sampler,
                coord,
                compare,
                offset,
                ..
            } => {
                texture.operands_mut(&mut v);
                sampler.operands_mut(&mut v);
                v.push(coord);
                v.extend(compare.as_mut());
                if let GatherOffset::Dynamic(o) = offset {
                    v.push(o);
                }
            }
            Op::LoadTexel {
                texture,
                coord,
                level,
                sample,
                ..
            } => {
                texture.operands_mut(&mut v);
                v.push(coord);
                v.extend(level.as_mut());
                v.extend(sample.as_mut());
            }
            Op::Query(r, q) => {
                r.operands_mut(&mut v);
                match q {
                    Query::Size(Some(l)) | Query::SamplePosition(l) => v.push(l),
                    _ => {}
                }
            }
            Op::Atomic {
                memory,
                value,
                compare,
                ..
            } => {
                memory.operands_mut(&mut v);
                v.push(value);
                v.extend(compare.as_mut());
            }
            Op::Counter(r, _) => r.operands_mut(&mut v),
            Op::ReportHit { t, kind, .. } => v.extend([t, kind]),
            Op::Wave(w) => match w {
                Wave::Ballot(a) | Wave::First(a) | Wave::QuadSwap(a, _) => v.push(a),
                Wave::Lane(a, l) | Wave::QuadLane(a, l) => v.extend([a, l]),
                Wave::Collective { value, .. } => v.push(value),
            },
        }
        v
    }

    /// The values the operation reads, in operand order.
    pub fn operands(&self) -> Vec<ValueId> {
        self.clone().operands_mut().into_iter().map(|v| *v).collect()
    }

    /// Replace each operand by `f` of it.
    pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        for v in self.operands_mut() {
            *v = f(*v);
        }
    }

    /// The variable the operation reads or changes: a load's, an atomic operation's on a variable.
    pub fn var(&self) -> Option<VarId> {
        self.clone().var_mut().copied()
    }

    /// The variable the operation reads or changes, to change.
    pub fn var_mut(&mut self) -> Option<&mut VarId> {
        match self {
            Op::LoadVar(var)
            | Op::LoadElement(var, ..)
            | Op::Atomic {
                memory: Memory::Var(var, _),
                ..
            }
            | Op::ReportHit { attributes: var, .. } => Some(var),
            _ => None,
        }
    }

    /// The resources the operation reaches, in operand order.
    pub fn resources(&self) -> Vec<ResourceId> {
        match self {
            Op::LoadState(r) => alloc::vec![*r],
            Op::LoadBuffer { resource: r, .. }
            | Op::LoadTexel { texture: r, .. }
            | Op::Query(r, _)
            | Op::Counter(r, _)
            | Op::Atomic {
                memory: Memory::Buffer(r, _) | Memory::Image(r, _),
                ..
            } => alloc::vec![r.id],
            Op::Sample { texture, sampler, .. } | Op::Gather { texture, sampler, .. } => {
                alloc::vec![texture.id, sampler.id]
            }
            _ => Vec::new(),
        }
    }

    /// Whether the operation reads state other than its operands (inputs, variables, resources, the quad, the wave);
    /// such a value is placed where its node is and is not moved across stores.
    pub fn reads_state(&self) -> bool {
        // a derivative reads the quad's other invocations, so it stays where its control flow puts it
        !matches!(
            self,
            Op::Literal(_)
                | Op::Undef
                | Op::Param
                | Op::Unary(..)
                | Op::Binary(..)
                | Op::Ternary(..)
                | Op::BitInsert { .. }
                | Op::Select { .. }
                | Op::Compose(_)
                | Op::Swizzle(..)
                | Op::Splat(_)
                | Op::Dot(..)
                | Op::Convert(..)
                | Op::Bitcast(_)
        ) || matches!(self, Op::Unary(Unary::Derivative(..), _))
    }

    /// Whether the operation changes state (memory, a counter): it is kept where it is and as often as it runs, even
    /// where nothing reads its value.
    pub fn has_effect(&self) -> bool {
        matches!(self, Op::Atomic { .. } | Op::Counter(..) | Op::ReportHit { .. })
    }
}

impl Memory {
    fn operands_mut<'a>(&'a mut self, out: &mut Vec<&'a mut ValueId>) {
        match self {
            Memory::Buffer(r, a) => {
                r.operands_mut(out);
                out.extend(a.index.as_mut());
            }
            Memory::Image(r, c) => {
                r.operands_mut(out);
                out.push(c);
            }
            Memory::Var(_, i) => out.extend(i.as_mut()),
        }
    }
}
