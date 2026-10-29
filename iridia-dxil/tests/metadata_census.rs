#![allow(clippy::unwrap_used)]
//! What the metadata accessors of `dxil_reader` find in the games' own Shader Model 6 programs: the attribute groups and
//! the string attributes of the functions, the metadata attached to instructions by kind, the entry point's properties
//! (tags and the values of the ray tracing and library ones), the trailing tags of the signature elements, and the
//! optional fields of the resource records. The expected totals are those of a second reader of the same containers (a
//! Python walker of the bitcode written for the census, sharing no code with `dxil-reader`); they pin down what the
//! stock programs use, and in particular what they never use: no `fp32-denorm-mode` attribute, no element tag but the
//! usage mask, no resource record with coherence, rasterizer ordering, sampler feedback or 64-bit atomics. Needs the
//! corpus oracle set's `skate/shaders` or `elden/shaders` (each a folder, searched recursively, of containers as the
//! game ships them).
mod inputs;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dxil_reader::module::{Attribute, Value};
use dxil_reader::reflection;
use inputs::{Game, OracleSet, Query, expect_found, resolve};

fn containers(root: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            containers(&path, out);
        } else if std::fs::read(&path).is_ok_and(|b| b.starts_with(b"DXBC")) {
            out.push(path);
        }
    }
}

#[derive(Default)]
struct Census {
    modules: usize,
    attribute_groups: usize,
    enum_attributes: usize,
    integer_attributes: usize,
    string_attributes: usize,
    /// Functions with at least one string attribute on themselves, and the string attributes they carry.
    functions_with_strings: usize,
    function_strings: usize,
    denorm_modes: usize,
    /// Attachments on instructions by kind name.
    attachments: BTreeMap<String, usize>,
    function_attachments: usize,
    /// Entry point properties by tag.
    properties: BTreeMap<i64, usize>,
    /// The values of the ray payload size, the ray attribute size, the library kind and the auto binding space.
    payload_sizes: BTreeMap<u32, usize>,
    attribute_sizes: BTreeMap<u32, usize>,
    library_kinds: BTreeMap<u32, usize>,
    binding_spaces: BTreeMap<u32, usize>,
    /// Signature elements, those with a usage mask, and those with any other tag.
    elements: usize,
    usage_masks: usize,
    other_element_tags: usize,
    /// Resource records by class (0 SRV, 1 UAV, 2 constant buffer, 3 sampler).
    resources: BTreeMap<u8, usize>,
    sample_counts: BTreeMap<u32, usize>,
    counters: usize,
    coherent: usize,
    rasterizer_ordered: usize,
    feedback: usize,
    atomic64: usize,
    resource_tags: BTreeMap<i64, usize>,
}

fn census(game: Game) -> Option<(String, Census)> {
    let query = Query::Oracle(OracleSet::Corpus, game, "shaders");
    let root = resolve(query)?;
    let mut paths = Vec::new();
    containers(&root, &mut paths);
    expect_found(query, "DXBC containers", paths.len());
    let mut c = Census::default();
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let name = path.display();
        let chunks = dxil_reader::parts(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let m = dxil_reader::module(&chunks, b"DXIL")
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .unwrap();
        c.modules += 1;

        // attributes
        c.attribute_groups += m.attribute_groups.len();
        for g in &m.attribute_groups {
            for a in &g.attributes {
                match a {
                    Attribute::Enum(_) => c.enum_attributes += 1,
                    Attribute::Integer(..) => c.integer_attributes += 1,
                    Attribute::String(..) => c.string_attributes += 1,
                }
            }
        }
        for (id, v) in m.values.iter().enumerate() {
            if !matches!(v, Value::Function { .. }) {
                continue;
            }
            let strings = m
                .function_attributes(id)
                .into_iter()
                .filter(|a| matches!(a, Attribute::String(..)))
                .count();
            c.function_strings += strings;
            c.functions_with_strings += usize::from(strings > 0);
        }

        // attachments
        for f in &m.functions {
            c.function_attachments += f.attachments.len();
            for i in &f.instructions {
                for a in &i.attachments {
                    let kind = m.kind_name(a.kind).unwrap_or("?").to_owned();
                    *c.attachments.entry(kind).or_default() += 1;
                }
            }
        }

        // the entry point
        c.denorm_modes += usize::from(
            reflection::fp32_denorm_mode(&m)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .is_some(),
        );
        for p in reflection::entry_properties(&m).unwrap_or_else(|e| panic!("{name}: {e}")) {
            *c.properties.entry(p.tag).or_default() += 1;
        }
        let tally = |map: &mut BTreeMap<u32, usize>, value: Option<u32>| {
            if let Some(v) = value {
                *map.entry(v).or_default() += 1;
            }
        };
        tally(&mut c.payload_sizes, reflection::ray_payload_size(&m).unwrap());
        tally(
            &mut c.attribute_sizes,
            reflection::ray_attribute_size(&m).unwrap(),
        );
        tally(&mut c.binding_spaces, reflection::auto_binding_space(&m).unwrap());
        if reflection::entry_properties(&m)
            .unwrap()
            .iter()
            .any(|p| p.tag == 8)
        {
            tally(&mut c.library_kinds, Some(reflection::library_kind(&m).unwrap()));
        }
        for list in reflection::element_tags(&m).unwrap_or_else(|e| panic!("{name}: {e}")) {
            for tags in list {
                c.elements += 1;
                c.usage_masks += usize::from(tags.usage_mask.is_some());
                c.other_element_tags += usize::from(
                    tags.stream.is_some() || tags.dynamic_index_mask.is_some() || !tags.other.is_empty(),
                );
            }
        }

        // the resources
        let ranges = reflection::ranges(&m).unwrap_or_else(|e| panic!("{name}: {e}"));
        let extras = reflection::resource_extras(&m).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(ranges.len(), extras.len(), "{name}");
        for ((range, _, _), extra) in ranges.iter().zip(&extras) {
            assert_eq!((range.class, range.id), (extra.class, extra.id), "{name}");
            *c.resources.entry(range.class).or_default() += 1;
            c.counters += usize::from(range.counter);
            if let Some(n) = extra.sample_count.filter(|n| *n != 0) {
                *c.sample_counts.entry(n).or_default() += 1;
            }
            c.coherent += usize::from(extra.globally_coherent);
            c.rasterizer_ordered += usize::from(extra.rasterizer_ordered);
            c.feedback += usize::from(extra.sampler_feedback.is_some());
            c.atomic64 += usize::from(extra.atomic64);
            for (tag, _) in &extra.tags {
                *c.resource_tags.entry(*tag).or_default() += 1;
            }
        }
    }
    println!("{}: {} modules", query.key(), c.modules);
    Some((query.key().to_string(), c))
}

fn counts<K: Ord + Clone, const N: usize>(pairs: [(K, usize); N]) -> BTreeMap<K, usize> {
    pairs.into_iter().collect()
}

fn named<const N: usize>(pairs: [(&str, usize); N]) -> BTreeMap<String, usize> {
    pairs.into_iter().map(|(k, n)| (k.to_owned(), n)).collect()
}

#[test]
fn skate_programs() {
    let Some((name, c)) = census(Game::Skate) else {
        return;
    };
    assert_eq!(c.modules, 19_445, "{name}");
    // 60,793 attribute groups of flags only: no integer attribute, no string attribute, so no denormal mode
    assert_eq!(c.attribute_groups, 60_793, "{name}");
    assert_eq!(c.enum_attributes, 101_896, "{name}");
    assert_eq!((c.integer_attributes, c.string_attributes), (0, 0), "{name}");
    assert_eq!(
        (c.functions_with_strings, c.function_strings, c.denorm_modes),
        (0, 0, 0),
        "{name}"
    );
    assert_eq!(
        c.attachments,
        named([
            ("alias.scope", 3_683),
            ("dx.controlflow.hints", 121_111),
            ("dx.precise", 107_469),
            ("llvm.loop", 1_651),
            ("noalias", 13_060),
            ("range", 7),
            ("tbaa", 42_568),
        ]),
        "{name}"
    );
    assert_eq!(c.function_attachments, 0, "{name}");
    // pixel and vertex programs state flags (14,369); compute programs a thread group (5,780); 205 libraries state the
    // auto binding space (0) and their kind; 27 and 19 of them state a payload and an attribute size; one program states
    // mesh state
    assert_eq!(
        c.properties,
        counts([
            (0, 14_369),
            (4, 5_780),
            (5, 205),
            (6, 27),
            (7, 19),
            (8, 205),
            (9, 1)
        ]),
        "{name}"
    );
    assert_eq!(c.binding_spaces, counts([(0, 205)]), "{name}");
    assert_eq!(c.payload_sizes, counts([(4, 4), (20, 17), (32, 6)]), "{name}");
    assert_eq!(c.attribute_sizes, counts([(4, 5), (8, 14)]), "{name}");
    assert_eq!(
        c.library_kinds,
        counts([(7, 172), (8, 6), (9, 3), (10, 16), (11, 8)]),
        "{name}"
    );
    // the usage mask is the only element tag
    assert_eq!(c.elements, 145_462, "{name}");
    assert_eq!((c.usage_masks, c.other_element_tags), (138_044, 0), "{name}");
    assert_eq!(
        c.resources,
        counts([(0, 121_032), (1, 13_176), (2, 46_350), (3, 21_633)]),
        "{name}"
    );
    assert_eq!(c.sample_counts, counts([(2, 1)]), "{name}");
    assert_eq!(c.counters, 147, "{name}");
    assert_eq!(
        (c.coherent, c.rasterizer_ordered, c.feedback, c.atomic64),
        (0, 0, 0, 0),
        "{name}"
    );
    assert_eq!(c.resource_tags, counts([(0, 95_873), (1, 32_753)]), "{name}");
}

#[test]
fn elden_ring_programs() {
    let Some((name, c)) = census(Game::Elden) else {
        return;
    };
    assert_eq!(c.modules, 50_283, "{name}");
    // 4,281 library functions carry nine string attributes each (the C++ front end's: `no-frame-pointer-elim`,
    // `less-precise-fpmad`, ...), none of them `fp32-denorm-mode`
    assert_eq!(c.attribute_groups, 160_520, "{name}");
    assert_eq!(c.enum_attributes, 268_620, "{name}");
    assert_eq!((c.integer_attributes, c.string_attributes), (0, 38_529), "{name}");
    assert_eq!(
        (c.functions_with_strings, c.function_strings, c.denorm_modes),
        (4_281, 38_529, 0),
        "{name}"
    );
    assert_eq!(
        c.attachments,
        named([
            ("dx.controlflow.hints", 88_988),
            ("llvm.loop", 48),
            ("noalias", 15_817),
            ("range", 2_137),
            ("tbaa", 4_449),
        ]),
        "{name}"
    );
    assert_eq!(c.function_attachments, 0, "{name}");
    assert_eq!(
        c.properties,
        counts([
            (0, 31_265),
            (1, 33),
            (2, 150),
            (3, 136),
            (4, 1_731),
            (5, 4_281),
            (6, 4_278),
            (7, 4_276),
            (8, 4_281),
        ]),
        "{name}"
    );
    assert_eq!(c.binding_spaces, counts([(0, 4_281)]), "{name}");
    assert_eq!(c.payload_sizes, counts([(4, 2_139), (12, 2_139)]), "{name}");
    assert_eq!(c.attribute_sizes, counts([(8, 4_276)]), "{name}");
    assert_eq!(
        c.library_kinds,
        counts([(7, 3), (9, 2_138), (10, 2_138), (11, 2)]),
        "{name}"
    );
    assert_eq!(c.elements, 545_303, "{name}");
    assert_eq!((c.usage_masks, c.other_element_tags), (459_288, 0), "{name}");
    assert_eq!(
        c.resources,
        counts([(0, 392_694), (1, 1_985), (2, 280_916), (3, 102_908)]),
        "{name}"
    );
    assert_eq!(c.sample_counts, counts([(2, 3), (4, 4)]), "{name}");
    assert_eq!(c.counters, 0, "{name}");
    assert_eq!(
        (c.coherent, c.rasterizer_ordered, c.feedback, c.atomic64),
        (0, 0, 0, 0),
        "{name}"
    );
    assert_eq!(c.resource_tags, counts([(0, 321_451), (1, 44_394)]), "{name}");
}
