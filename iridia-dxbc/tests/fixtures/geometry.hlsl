// A geometry program (Shader Model 5): a triangle in, two triangle strips out. Compiled with
//   fxc /T gs_5_0 /E main /Fo geometry.dxbc geometry.hlsl   (and /Od for geometry.od.dxbc).
// Per invocation: the triangle's corners with x scaled by `scale` (the first strip), a restart, then the corners moved by
// the primitive number in x and one in y, uv.x increased by one (the second strip). The loops are unrolled: the
// container decoder reads a vertex of the input primitive by a constant index.
cbuffer Scale : register(b0) {
    float scale;
};

struct Vertex {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[maxvertexcount(6)]
void main(triangle Vertex input[3], uint primitive : SV_PrimitiveID, inout TriangleStream<Vertex> stream) {
    [unroll]
    for (uint i = 0; i < 3; i++) {
        Vertex v = input[i];
        v.position.x *= scale;
        stream.Append(v);
    }
    stream.RestartStrip();
    [unroll]
    for (uint j = 0; j < 3; j++) {
        Vertex v = input[j];
        v.position.x += primitive;
        v.position.y += 1;
        v.uv.x += 1;
        stream.Append(v);
    }
}
