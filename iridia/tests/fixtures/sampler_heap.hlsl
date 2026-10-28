// A pixel program (Shader Model 6.6) that samples a bound texture with a sampler from the sampler heap at an index from
// a constant buffer: the heap is its only array of descriptors. Compiled with dxc -T ps_6_6 -E main (sampler_heap.dxil).
cbuffer Indices : register(b0) {
    uint sampler_index;
};

Texture2D<float4> image : register(t0);

float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    SamplerState smp = SamplerDescriptorHeap[sampler_index];
    return image.Sample(smp, uv);
}
