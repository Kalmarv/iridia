// The reading of a pixel program's outputs as a surface (`iridia_interface::surface`), for a composite that
// decodes a G-buffer on the GPU as the CPU reading does. A program's surface is `iridia_surface_rows` rows of four
// words (`iridia::surface::rows`), one per term at the index its constant below names:
//
//   x  the render target the term is in, plus one; 0 where no output has a role for the term (the caller takes its
//      own default and calls none of these functions with the row)
//   y  bits 0 and 1 the term's first lane, 4 to 7 how many lanes it has, 8 to 15 its kind, 16 to 23 and 24 to 31 the
//      power and the root of its scalar encoding (`ScalarEncoding`: the value is `stored^power`, its root-th root,
//      times the scale)
//   z  the scale's f32 bits, or a normal encoding's first number
//   w  a normal encoding's second number
//
// Each function takes the row and the four values of the target it names (the caller reads them, at `row.x - 1`).

const iridia_surface_rows: u32 = 9u;
const iridia_surface_base: u32 = 0u;
const iridia_surface_normal: u32 = 1u;
const iridia_surface_roughness: u32 = 2u;
const iridia_surface_metallic: u32 = 3u;
const iridia_surface_reflectance: u32 = 4u;
const iridia_surface_specular: u32 = 5u;
const iridia_surface_emission: u32 = 6u;
const iridia_surface_occlusion: u32 = 7u;
// The occlusion of the environment's ambient light alone (`Surface::ambient_occlusion`): it scales the ambient light,
// not the other lights; a surface has this row or the occlusion row, not both.
const iridia_surface_ambient_occlusion: u32 = 8u;

fn iridia_kind(row: vec4<u32>) -> u32 {
    return (row.y >> 8u) & 0xffu;
}

// Lane `k` of the term (0 its first lane), 0 past the target's four.
fn iridia_lane(row: vec4<u32>, value: vec4<f32>, k: u32) -> f32 {
    let lane = (row.y & 0x3u) + k;
    if (lane > 3u) {
        return 0.0;
    }
    return value[lane];
}

// The term's lanes as the program stored them, undecoded (a role shown as written): three or more lanes their first
// three as a colour, one or two lanes the first as grey.
fn iridia_stored(row: vec4<u32>, value: vec4<f32>) -> vec3<f32> {
    if (((row.y >> 4u) & 0xfu) >= 3u) {
        return vec3<f32>(iridia_lane(row, value, 0u), iridia_lane(row, value, 1u), iridia_lane(row, value, 2u));
    }
    return vec3<f32>(iridia_lane(row, value, 0u));
}

// `ScalarEncoding::decode`: the stored value to the power, its root (of 0 where negative), times the scale.
fn iridia_decode(row: vec4<u32>, stored: f32) -> f32 {
    let power = (row.y >> 16u) & 0xffu;
    let root = (row.y >> 24u) & 0xffu;
    var raised = 1.0;
    for (var i = 0u; i < power; i = i + 1u) {
        raised = raised * stored;
    }
    var value = raised;
    if (root == 2u) {
        value = sqrt(max(raised, 0.0));
    } else if (root > 2u) {
        value = pow(max(raised, 0.0), 1.0 / f32(root));
    }
    return value * bitcast<f32>(row.z);
}

// A one-lane term, decoded (the metallic, the specular strength, either occlusion).
fn iridia_scalar(row: vec4<u32>, value: vec4<f32>) -> f32 {
    return iridia_decode(row, iridia_lane(row, value, 0u));
}

// A three-lane term, each lane decoded (the reflectance).
fn iridia_colour(row: vec4<u32>, value: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(
        iridia_decode(row, iridia_lane(row, value, 0u)),
        iridia_decode(row, iridia_lane(row, value, 1u)),
        iridia_decode(row, iridia_lane(row, value, 2u)),
    );
}

// The surface's colour (`Base`), its first three lanes as stored. Kind 1 an albedo, 2 a forward program's finished
// colour.
fn iridia_base(row: vec4<u32>, value: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(iridia_lane(row, value, 0u), iridia_lane(row, value, 1u), iridia_lane(row, value, 2u));
}

// The perceptual roughness (`Roughness::decode`), unclamped. Kind 1 the roughness stored, 2 a gloss (one minus it),
// 3 a specular exponent n, `(2 / (n + 2))^(1/4)`.
fn iridia_roughness(row: vec4<u32>, value: vec4<f32>) -> f32 {
    let v = iridia_scalar(row, value);
    let kind = iridia_kind(row);
    if (kind == 2u) {
        return 1.0 - v;
    }
    if (kind == 3u) {
        return pow(2.0 / (max(v, 0.0) + 2.0), 0.25);
    }
    return v;
}

// The light the surface gives off (`Emission::decode`). Kind 1 a colour in three lanes, 2 one lane scaling `albedo`.
fn iridia_emission(row: vec4<u32>, value: vec4<f32>, albedo: vec3<f32>) -> vec3<f32> {
    if (iridia_kind(row) == 2u) {
        return albedo * iridia_scalar(row, value);
    }
    return iridia_colour(row, value);
}

fn iridia_length_or_zero(n: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(n, n);
    if (length_squared > 1e-16) {
        return n * inverseSqrt(length_squared);
    }
    return n;
}

// The octahedral unfold: z = 1 - |x| - |y|, x and y folded back where z < 0, normalised.
fn iridia_unfold(x: f32, y: f32) -> vec3<f32> {
    let z = 1.0 - abs(x) - abs(y);
    var fx = x;
    var fy = y;
    if (z < 0.0) {
        fx = (1.0 - abs(y)) * select(1.0, -1.0, x < 0.0);
        fy = (1.0 - abs(x)) * select(1.0, -1.0, y < 0.0);
    }
    return iridia_length_or_zero(vec3<f32>(fx, fy, z));
}

// The diamond layout over three 8-bit lanes (`NormalEncoding::decode_octahedral_diamond`): lanes 0 and 1 the low eight
// bits of two 9-bit coordinates, lane 2's bits 1 and 2 their ninth bits and its bit 0 the sign of z.
fn iridia_diamond(lanes: vec3<f32>) -> vec3<f32> {
    let bytes = vec3<u32>(round(clamp(lanes, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0));
    let u9 = f32(((bytes.z >> 1u) & 1u) * 256u + bytes.x);
    let v9 = f32(((bytes.z >> 2u) & 1u) * 256u + bytes.y);
    let pu = clamp(u9 / 255.0 - 1.0, -1.0, 1.0);
    let pv = clamp(v9 / 255.0 - 1.0, -1.0, 1.0);
    let x = (pu + pv) * 0.5;
    let y = (pu - pv) * 0.5;
    let z = (1.0 - abs(x) - abs(y)) * select(-1.0, 1.0, (bytes.z & 1u) == 1u);
    return iridia_length_or_zero(vec3<f32>(x, y, z));
}

// The cube face (`NormalEncoding::CubeFace`): the nearest axis to lane 3 and the side of one half lane 2 is on; lanes 0
// and 1 the face's two minor components through the minor affine (z its scale, w its bias).
fn iridia_cube_face(row: vec4<u32>, value: vec4<f32>) -> vec3<f32> {
    let scale = bitcast<f32>(row.z);
    let bias = bitcast<f32>(row.w);
    let sc = (iridia_lane(row, value, 0u) - bias) / scale;
    let tc = (iridia_lane(row, value, 1u) - bias) / scale;
    let axis = u32(clamp(round(iridia_lane(row, value, 3u) * 3.0), 0.0, 2.0));
    let negative = iridia_lane(row, value, 2u) > 0.5;
    let major = sqrt(max(1.0 - sc * sc - tc * tc, 0.0)) * select(1.0, -1.0, negative);
    var n = vec3<f32>(0.0);
    switch (axis * 2u + select(0u, 1u, negative)) {
        case 0u: { n = vec3<f32>(major, -tc, -sc); }
        case 1u: { n = vec3<f32>(major, -tc, sc); }
        case 2u: { n = vec3<f32>(sc, major, tc); }
        case 3u: { n = vec3<f32>(sc, major, -tc); }
        case 4u: { n = vec3<f32>(sc, -tc, major); }
        default: { n = vec3<f32>(-sc, -tc, major); }
    }
    return iridia_length_or_zero(n);
}

// The normal the lanes store (`Normal::vector`): the three stored components unbiased (kinds 1 to 3, not normalised),
// the unit vector of the others. Kinds: 1 `Xyz`, 2 `XyzUnorm`, 3 `XyzBiased` (z the bias), 4 `Xy`, 5 `XyUnorm`,
// 6 `Octahedral` (z and w the read affine's scale and bias), 7 `OctahedralDiamond`, 8 `CubeFace` (z and w the minor
// affine's scale and bias).
fn iridia_normal(row: vec4<u32>, value: vec4<f32>) -> vec3<f32> {
    let l0 = iridia_lane(row, value, 0u);
    let l1 = iridia_lane(row, value, 1u);
    let l2 = iridia_lane(row, value, 2u);
    switch (iridia_kind(row)) {
        case 1u: { return vec3<f32>(l0, l1, l2); }
        case 2u: { return (vec3<f32>(l0, l1, l2) - vec3<f32>(0.5)) * 2.0; }
        case 3u: { return (vec3<f32>(l0, l1, l2) - vec3<f32>(bitcast<f32>(row.z))) * 2.0; }
        case 4u: { return vec3<f32>(l0, l1, sqrt(max(1.0 - l0 * l0 - l1 * l1, 0.0))); }
        case 5u: {
            let x = l0 * 2.0 - 1.0;
            let y = l1 * 2.0 - 1.0;
            return vec3<f32>(x, y, sqrt(max(1.0 - x * x - y * y, 0.0)));
        }
        case 6u: {
            let scale = bitcast<f32>(row.z);
            let bias = bitcast<f32>(row.w);
            return iridia_unfold(l0 * scale + bias, l1 * scale + bias);
        }
        case 7u: { return iridia_diamond(vec3<f32>(l0, l1, l2)); }
        case 8u: { return iridia_cube_face(row, value); }
        default: { return vec3<f32>(0.0); }
    }
}
