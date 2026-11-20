// A float compared with itself, the test for NaN (our own HLSL): fxc /T ps_5_0 /E main. The quotient is NaN where the
// constant buffer makes it one (0 / 0), which fxc cannot rule out, so it keeps the comparisons: `isnan(x)` and
// `x != x` are `ne x, x`. It packs comparisons of different pairs into one instruction, so an instruction compares a
// lane with itself beside a lane with another:
//   ne r1.xyzw, r0.xyzy, r0.xyzx
//   eq r3.xy, r0.yzyy, r0.yyyy
//   ge r3.xz, r0.zzzz, r0.zzwz
//   lt r3.w, r0.w, r0.x
// (`q.w < q.w` it writes as 0 itself.)
cbuffer C : register(b0) { float4 n; float4 d; };
struct O {
    float4 guarded : SV_Target0; // a NaN lane written as 0, as a filter guards what it writes
    uint4 same : SV_Target1;     // x != x, x == x, x >= x, x < x
    uint4 other : SV_Target2;    // each lane against another lane: comparisons of two values
};
O main() {
    O o;
    float4 q = n / d;
    o.guarded = float4(isnan(q.xyz) ? 0.0 : q.xyz, 1.0);
    o.same = uint4(q.x != q.x, q.y == q.y, q.z >= q.z, q.w < q.w);
    o.other = uint4(q.x != q.y, q.y == q.z, q.z >= q.w, q.w < q.x);
    return o;
}
