// SPDX-License-Identifier: AGPL-3.0-only
//! Cross-check vectors for the TypeScript history client: the domain
//! elements, the history input `u`, its canonical point `H`, the blinded
//! request `B = r·H`, the evaluator's answer with its proof, and the tag, for
//! a spread of passwords.
//!
//! `cargo run -p sid-pake-core --example history_vectors -- history-vectors.json`

use ff::{Field, PrimeField};
use group::{Group, GroupEncoding};
use pasta_curves::pallas;
use rand::{SeedableRng, rngs::StdRng};
use sid_pake_core::history::{
    blind_request, canonical_point, domain_element, evaluate_with_proof, finalize_tag,
    history_input, random_blind,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn vector(case: u64, owner: &[u8], password: &[u8]) -> String {
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", owner]);
    let c = domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-1", b"format-1"]);
    let u = history_input(d, password);
    let (h, offset) = canonical_point(u);
    let mut rng = StdRng::seed_from_u64(4_000 + case);
    let r = random_blind(&mut rng);
    let b = blind_request(u, r);
    let k = pallas::Scalar::random(&mut rng);
    let pk = group::Curve::to_affine(&(pallas::Point::generator() * k));
    let operation_id = [case as u8; 16];
    let (z, proof) = evaluate_with_proof(k, b, &operation_id, &mut rng).expect("nonidentity");
    let t = finalize_tag(c, u, r, z);
    format!(
        "{{\"owner\":\"{}\",\"password\":\"{}\",\"ownerDomain\":\"{}\",\"comparisonDomain\":\"{}\",\
         \"u\":\"{}\",\"h\":\"{}\",\"offset\":{offset},\"blind\":\"{}\",\"blinded\":\"{}\",\
         \"publicKey\":\"{}\",\"operationId\":\"{}\",\"evaluated\":\"{}\",\"challenge\":\"{}\",\
         \"response\":\"{}\",\"tag\":\"{}\"}}",
        hex(owner),
        hex(password),
        hex(d.to_repr().as_ref()),
        hex(c.to_repr().as_ref()),
        hex(u.to_repr().as_ref()),
        hex(h.to_bytes().as_ref()),
        hex(r.to_repr().as_ref()),
        hex(b.to_bytes().as_ref()),
        hex(pk.to_bytes().as_ref()),
        hex(&operation_id),
        hex(z.to_bytes().as_ref()),
        hex(proof.c.to_repr().as_ref()),
        hex(proof.s.to_repr().as_ref()),
        hex(t.to_repr().as_ref()),
    )
}

fn main() {
    let max = vec![b'Z'; 128];
    let cases: Vec<String> = [
        (&b"alice"[..], &b"Str0ngP@ssword!"[..]),
        (b"bob", b"Str0ngP@ssword!"),
        (b"alice", "пароль-Ünïcode-1".as_bytes()),
        (b"alice", b"x"),
        (b"alice", &max),
        (b"carol", b"Aa1!Aa1!"),
    ]
    .iter()
    .enumerate()
    .map(|(i, (owner, password))| vector(i as u64, owner, password))
    .collect();
    let out = std::env::args()
        .nth(1)
        .expect("usage: history_vectors <output.json>");
    std::fs::write(&out, format!("[{}]\n", cases.join(",\n"))).expect("write the vectors");
}
