// SPDX-License-Identifier: AGPL-3.0-only
//! Verify registration proofs made by the clients (the TypeScript prover and
//! the WASM kernel) with the reference verifier, as the server does.
//!
//! `cargo run --release -p sid-pake-core --example verify_proofs -- <file.jsonl>`
//!
//! Each line is `{"label", "policyVersion", "operationId", "request",
//! "proof", "instances": [..]}`, bytes in hex: the operation id (16 bytes),
//! the OPAQUE registration request (the element M, 32 bytes), the proof and
//! its public instances (32 bytes each). Prints one verdict per line and
//! exits non-zero if any proof fails.

use std::collections::HashMap;

use ff::PrimeField;
use group::GroupEncoding;
use pasta_curves::pallas;
use sid_pake_core::binding::operation_context;
use sid_pake_core::circuit::{CircuitShape, ZKPP_K};
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::prover::BoundProof;
use sid_pake_core::types::{PolicyVersion, ZkppProof};
use sid_pake_core::verifier::ZkppVerifier;

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd hex length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn field(v: &serde_json::Value, what: &str) -> Result<String, String> {
    v[what]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{what}: expected a hex string"))
}

fn base(hex: &str) -> Result<pallas::Base, String> {
    let bytes: [u8; 32] = unhex(hex)?
        .try_into()
        .map_err(|_| "instance must be 32 bytes".to_string())?;
    Option::from(pallas::Base::from_repr(bytes)).ok_or_else(|| "not a field element".into())
}

fn check(
    line: &serde_json::Value,
    verifiers: &mut HashMap<(u32, usize), ZkppVerifier>,
) -> Result<(), String> {
    let policy_version = line["policyVersion"]
        .as_u64()
        .ok_or("policyVersion: expected a number")? as u32;
    let operation_id: [u8; 16] = unhex(&field(line, "operationId")?)?
        .try_into()
        .map_err(|_| "operationId must be 16 bytes".to_string())?;
    let request = unhex(&field(line, "request")?)?;
    let request_bytes: [u8; 32] = request
        .clone()
        .try_into()
        .map_err(|_| "request must be 32 bytes".to_string())?;
    let m = Option::<pallas::Affine>::from(pallas::Affine::from_bytes(&request_bytes))
        .ok_or("request: not a curve point")?;
    let instances: Vec<pallas::Base> = line["instances"]
        .as_array()
        .ok_or("instances: expected an array")?
        .iter()
        .map(|v| base(v.as_str().ok_or("instance: expected a hex string")?))
        .collect::<Result<_, _>>()?;
    // instances = M (2) + d + c_j (D) + B (2) + (Z_j (2) + t_j)·D
    let domains = instances
        .len()
        .checked_sub(5)
        .filter(|n| n % 4 == 0)
        .ok_or("instance count fits no domain count")?
        / 4;
    let verifier = match verifiers.entry((policy_version, domains)) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let policy = sid_pake_core::policy::get_policy(PolicyVersion(policy_version))
                .ok_or("unknown policy version")?;
            let shape = CircuitShape {
                policy,
                history_domains: domains,
            };
            let params = generate_params(ZKPP_K);
            let pk = generate_pk(&params, shape).map_err(|e| format!("keygen: {e:?}"))?;
            let vk = pk.get_vk().clone();
            e.insert(ZkppVerifier::new(params, vk, shape))
        }
    };
    let bound = BoundProof {
        snark_proof: ZkppProof(unhex(&field(line, "proof")?)?),
        instances,
    };
    verifier
        .verify(&bound, &operation_context(&operation_id, &request), m)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Variants of a valid line an attacker could send instead, from cheapest to
/// reject to costliest: a wrong length, random bytes of the right length,
/// and the valid proof with its last byte flipped (fails only at the end).
fn adversarial(line: &serde_json::Value) -> Vec<(String, serde_json::Value)> {
    let proof = line["proof"].as_str().unwrap_or_default().to_owned();
    let mut short = line.clone();
    short["proof"] = serde_json::Value::from(&proof[..proof.len() / 2]);
    let mut random = line.clone();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    random["proof"] = serde_json::Value::from(
        (0..proof.len() / 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                format!("{:02x}", state as u8)
            })
            .collect::<String>(),
    );
    let mut flipped = line.clone();
    let last = u8::from_str_radix(&proof[proof.len() - 2..], 16).unwrap_or(0) ^ 1;
    flipped["proof"] = serde_json::Value::from(format!("{}{last:02x}", &proof[..proof.len() - 2]));
    vec![
        ("half-length proof".into(), short),
        ("random bytes, right length".into(), random),
        ("last byte flipped".into(), flipped),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .expect("usage: verify_proofs <file.jsonl> [--adversarial]");
    let text = std::fs::read_to_string(path).expect("read the proofs file");
    let mut verifiers = HashMap::new();
    let mut failed = 0;
    let mut first: Option<serde_json::Value> = None;
    for (n, raw) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let line: serde_json::Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => {
                println!("line {}: not JSON: {e}", n + 1);
                failed += 1;
                continue;
            }
        };
        let label = line["label"].as_str().unwrap_or("?").to_owned();
        let t0 = std::time::Instant::now();
        let verdict = check(&line, &mut verifiers);
        // The first proof of a shape also builds its verifying key.
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        match verdict {
            Ok(()) => println!("{label}: VERIFIED in {ms:.1} ms"),
            Err(e) => {
                println!("{label}: FAILED ({e}) in {ms:.1} ms");
                failed += 1;
            }
        }
        first.get_or_insert(line);
    }
    if args.iter().any(|a| a == "--adversarial")
        && let Some(line) = &first
    {
        for (what, bad) in adversarial(line) {
            let t0 = std::time::Instant::now();
            let verdict = check(&bad, &mut verifiers);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "adversarial, {what}: {} in {ms:.2} ms",
                match verdict {
                    Ok(()) => "ACCEPTED (bug)".to_string(),
                    Err(e) => format!("rejected ({e})"),
                }
            );
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
