// SPDX-License-Identifier: AGPL-3.0-only
//! Cross-check references for the TypeScript circuit port and its keygen:
//! for each key shape, the verifying key's pinned representation (the text
//! the verifying key's transcript representative hashes: constraint system,
//! fixed and permutation commitments), and the SRS the keys are built over.
//!
//! `cargo run --release -p sid-pake-core --example circuit_vectors -- <dir>`
//! writes `<dir>/pinned-<policy>-<domains>.txt` and `<dir>/srs-k11.bin`.

use sid_pake_core::circuit::{CircuitShape, ZKPP_K};
use sid_pake_core::keygen::{generate_params, generate_vk, write_params};
use sid_pake_core::types::CE_DEFAULT_POLICY;

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: circuit_vectors <output-dir>"),
    );
    std::fs::create_dir_all(&dir).expect("create the output directory");
    let params = generate_params(ZKPP_K);
    let mut srs = Vec::new();
    write_params(&params, &mut srs).expect("serialize the SRS");
    std::fs::write(dir.join("srs-k11.bin"), srs).expect("write the SRS");
    for domains in [1, 2] {
        let shape = CircuitShape {
            policy: CE_DEFAULT_POLICY,
            history_domains: domains,
        };
        let vk = generate_vk(&params, shape).expect("keygen");
        std::fs::write(
            dir.join(format!("pinned-ce-{domains}.txt")),
            format!("{:?}", vk.pinned()),
        )
        .expect("write the pinned key");
        std::fs::write(
            dir.join(format!("pinned-ce-{domains}.pretty.txt")),
            format!("{:#?}", vk.pinned()),
        )
        .expect("write the readable pinned key");
    }
}
