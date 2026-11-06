//! The version gate of a lifter on the IR. A lifter's `LIFTER_VERSION` and the IR's [`crate::VERSION`] are what a cache
//! keyed by a bytecode's hash trusts to know that a program it holds is stale, so a change to what a lifter makes of a
//! bytecode must come with a bump. The gate makes that checkable: a lifter's tests lift its committed fixtures, [`table`]
//! hashes each outcome (the whole program as [`lifted`] writes it, or the refusal) under the versions the lifter has,
//! the table is committed, and [`check`] fails when the outcomes changed and the versions did not (bump the one that
//! names the change), or the versions changed and the committed table is not that version's.

use alloc::{borrow::ToOwned, format, string::String, vec::Vec};
use core::fmt::Write;

use sha2::{Digest, Sha256};

use crate::interface::{Interface, Stage};
use crate::{Conventions, FloatEnv, Program, Provenance};

/// One fixture's outcome: the program lifted as [`lifted`] writes it, or the refusal's reason.
pub type Outcome = Result<String, String>;

/// A lifted program as the gate hashes it: the entry function's [`crate::text`] form, then every other part of the
/// program (its stage, interface, conventions, float controls, provenance and notes) as JSON, so a change a lifter makes
/// to its bindings, layouts or policies alone is a changed outcome too. The lifter's version is left out of the
/// provenance: the table's header carries it.
pub fn lifted(program: &Program) -> String {
    // destructured whole, so a part added to programs is a part the gate hashes or a compile error here
    let Program {
        stage,
        entry,
        patch,
        workgroup,
        interface,
        conventions,
        float,
        provenance,
        notes,
    } = program;
    #[derive(serde::Serialize)]
    struct Rest<'a> {
        stage: &'a Stage,
        // left out where there is none, so the outcomes of stages without one are what they were before it
        #[serde(skip_serializing_if = "Option::is_none")]
        workgroup: &'a Option<[u32; 3]>,
        interface: &'a Interface,
        conventions: &'a Conventions,
        float: &'a FloatEnv,
        provenance: Provenance,
        notes: &'a [String],
    }
    let rest = Rest {
        stage,
        workgroup,
        interface,
        conventions,
        float,
        provenance: Provenance {
            lifter_version: 0,
            ..provenance.clone()
        },
        notes,
    };
    let json = serde_json::to_string(&rest).unwrap_or_else(|e| format!("not serialised: {e}"));
    let patch = patch.as_ref().map(crate::text::print).unwrap_or_default();
    format!("{}{patch}\n{json}\n", crate::text::print(entry))
}

/// The table of `outcomes` (fixture name, outcome) for `lifter` at these versions: a header of the versions, then a line
/// per fixture in the order given.
pub fn table(lifter: &str, lifter_version: u32, outcomes: &[(String, Outcome)]) -> String {
    let mut out = format!(
        "# lifted fixtures of {lifter}\n# versions lifter={lifter_version} ir={}\n",
        crate::VERSION
    );
    for (name, outcome) in outcomes {
        let _ = match outcome {
            Ok(text) => writeln!(out, "{name}\tlifted\t{}", hex(&Sha256::digest(text.as_bytes()))),
            Err(reason) => writeln!(out, "{name}\trefused\t{reason}"),
        };
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The first names of a list, and how many more there are.
fn some(names: &[impl AsRef<str>]) -> String {
    const SHOWN: usize = 12;
    let first: Vec<&str> = names.iter().take(SHOWN).map(AsRef::as_ref).collect();
    match names.len().checked_sub(SHOWN) {
        Some(more) if more > 0 => format!("{} and {more} more", first.join(", ")),
        _ => first.join(", "),
    }
}

fn versions(table: &str) -> Option<&str> {
    table.lines().find(|l| l.starts_with("# versions "))
}

fn entries(table: &str) -> impl Iterator<Item = &str> {
    table.lines().filter(|l| !l.starts_with('#'))
}

/// Whether the `current` table is the committed one; if not, what to do about it. A fixture whose outcome differs from
/// the committed one's needs a bump of the version that names the change; one added or taken away says nothing of what
/// the lifter makes of the others, so the committed table is replaced without a bump.
pub fn check(committed: &str, current: &str) -> Result<(), String> {
    if committed == current {
        return Ok(());
    }
    let name = |line: &str| line.split('\t').next().unwrap_or_default().to_owned();
    let committed_names: Vec<String> = entries(committed).map(name).collect();
    let current_names: Vec<String> = entries(current).map(name).collect();
    let changed: Vec<String> = entries(current)
        .filter(|line| !entries(committed).any(|c| c == *line))
        .map(name)
        .filter(|n| committed_names.contains(n))
        .collect();
    let added: Vec<&str> = current_names
        .iter()
        .filter(|n| !committed_names.contains(n))
        .map(String::as_str)
        .collect();
    let removed: Vec<&str> = committed_names
        .iter()
        .filter(|n| !current_names.contains(n))
        .map(String::as_str)
        .collect();
    let same_versions = versions(committed) == versions(current);
    let mut what = Vec::new();
    if !changed.is_empty() {
        what.push(format!(
            "the outcome of {} fixtures differs from the committed one: {}",
            changed.len(),
            some(&changed)
        ));
    }
    for (list, how) in [(&added, "added"), (&removed, "taken away")] {
        if !list.is_empty() {
            what.push(format!("{} fixtures {how}: {}", list.len(), some(list)));
        }
    }
    let what = if what.is_empty() {
        "the fixtures' outcomes are the committed ones".to_owned()
    } else {
        what.join("; ")
    };
    Err(if same_versions && !changed.is_empty() {
        format!(
            "what the lifter makes of its fixtures changed and its versions did not ({what}): bump LIFTER_VERSION (a \
             change of the lifter) or iridia_ir::VERSION (a change of the IR's meaning or text form), then replace the \
             committed table by the current one"
        )
    } else if same_versions {
        format!(
            "the fixtures are not the committed table's ({what}): replace the committed table by the current one"
        )
    } else {
        format!(
            "the versions are not the committed table's ({} now, {} committed; {what}): replace the committed table by \
             the current one",
            versions(current).unwrap_or("none"),
            versions(committed).unwrap_or("none"),
        )
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "tests")]
    use super::*;

    const HEAD: &str = "# lifted fixtures of l\n# versions lifter=1 ir=2\n";

    /// A changed outcome asks for a bump; an added or removed fixture only for the table; a bumped version for the
    /// table.
    #[test]
    fn a_changed_outcome_needs_a_bump_and_an_added_fixture_only_the_table() {
        let committed = format!("{HEAD}a\tlifted\t00\nb\trefused\tno\n");
        assert_eq!(check(&committed, &committed), Ok(()));
        let changed = check(&committed, &format!("{HEAD}a\tlifted\t01\nb\trefused\tno\n")).unwrap_err();
        assert!(
            changed.contains("bump") && changed.contains("1 fixtures differs") && changed.contains(": a"),
            "{changed}"
        );
        let added = check(
            &committed,
            &format!("{HEAD}a\tlifted\t00\nb\trefused\tno\nc\tlifted\t02\n"),
        )
        .unwrap_err();
        assert!(
            !added.contains("bump") && added.contains("1 fixtures added: c"),
            "{added}"
        );
        let removed = check(&committed, &format!("{HEAD}a\tlifted\t00\n")).unwrap_err();
        assert!(
            !removed.contains("bump") && removed.contains("1 fixtures taken away: b"),
            "{removed}"
        );
        let bumped = "# lifted fixtures of l\n# versions lifter=2 ir=2\na\tlifted\t01\nb\trefused\tno\n";
        let e = check(&committed, bumped).unwrap_err();
        assert!(!e.contains("bump LIFTER") && e.contains("lifter=2"), "{e}");
    }

    #[test]
    fn a_table_hashes_the_text_and_keeps_the_refusal() {
        let t = table(
            "l",
            3,
            &[("a".into(), Ok("x".into())), ("b".into(), Err("why".into()))],
        );
        let want = format!(
            "# lifted fixtures of l\n# versions lifter=3 ir={}\n\
             a\tlifted\t2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881\nb\trefused\twhy\n",
            crate::VERSION
        );
        assert_eq!(t, want);
    }
}
