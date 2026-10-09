// SPDX-License-Identifier: AGPL-3.0-only
//! Cross-check references for the TypeScript circuit port and its keygen:
//! for each key shape, the verifying key's pinned representation (the text
//! the verifying key's transcript representative hashes: constraint system,
//! fixed and permutation commitments), and the SRS the keys are built over.
//!
//! `cargo run --release -p sid-pake-core --example circuit_vectors -- <dir>`
//! writes `<dir>/pinned-<policy>-<domains>.txt`, `<dir>/srs-k11.bin`, and the
//! OPAQUE element mapping vectors `<dir>/hash-to-curve.json` and
//! `<dir>/gadget-c.json`.

use ff::PrimeField;
use group::Curve;
use pasta_curves::{arithmetic::CurveAffine, pallas};
use sid_pake_core::circuit::gadget_c::hash_to_curve_outside;
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

    // The OPAQUE element mapping (gadget C's witness): `H_p` for two passwords
    // as written, and `M = 7·H_p`, field elements little-endian.
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let fe = |v: pallas::Base| hex(&v.to_repr());
    let (h, u, offset) = hash_to_curve_outside(b"Str0ngP@ssword!");
    let h = h.coordinates().unwrap();
    std::fs::write(
        dir.join("hash-to-curve.json"),
        format!(
            r#"{{"password":"Str0ngP@ssword!","u":"{}","offset":{},"hpx":"{}","hpy":"{}"}}"#,
            fe(u),
            offset_u64(offset),
            fe(*h.x()),
            fe(*h.y())
        ),
    )
    .expect("write the hash-to-curve vector");
    let (h, u, _) = hash_to_curve_outside(b"Str0ngP@ss");
    let blind = pallas::Scalar::from(7u64);
    let m = (pallas::Point::from(h) * blind).to_affine();
    let (h, m) = (h.coordinates().unwrap(), m.coordinates().unwrap());
    std::fs::write(
        dir.join("gadget-c.json"),
        format!(
            r#"{{"u":"{}","hpx":"{}","hpy":"{}","blind":"{}","mx":"{}","my":"{}"}}"#,
            fe(u),
            fe(*h.x()),
            fe(*h.y()),
            hex(&blind.to_repr()),
            fe(*m.x()),
            fe(*m.y())
        ),
    )
    .expect("write the gadget C vector");
}

/// The offset as an integer: it is below 2^8 by construction.
fn offset_u64(offset: pallas::Base) -> u64 {
    let repr = offset.to_repr();
    u64::from_le_bytes(repr[..8].try_into().unwrap())
}
