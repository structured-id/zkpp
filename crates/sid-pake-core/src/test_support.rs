// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers for the crate's unit tests.

use group::Curve;
use pasta_curves::pallas;
use rand::rngs::OsRng;

use crate::circuit::HistoryInputs;
use crate::circuit::gadget_c::hash_to_curve_outside;
use crate::circuit::gadget_h::HistoryTagWitness;
use crate::history::{blind_request, domain_element, evaluate, history_input, random_blind};
use crate::prover::HistoryEvaluation;
use crate::types::MAX_PASSWORD_LEN;

/// The OPRF element an OPAQUE client sends for `password` under `blind`:
/// `blind·H_p` of the padded password, as the Pallas OPAQUE group hashes it.
pub(crate) fn oprf_element(password: &[u8], blind: pallas::Scalar) -> pallas::Affine {
    let mut padded = [0u8; MAX_PASSWORD_LEN];
    padded[..password.len()].copy_from_slice(password);
    let (h_p, _, _) = hash_to_curve_outside(&padded);
    (pallas::Point::from(h_p) * blind).to_affine()
}

/// Owner domain of the test owner.
pub(crate) fn test_owner() -> pallas::Base {
    domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"])
}

/// Comparison domains `epoch-0 ..` of the test manifest.
pub(crate) fn test_domains(n: usize) -> Vec<pallas::Base> {
    (0..n)
        .map(|j| domain_element(b"SID-HISTORY-TAG-v1", &[format!("epoch-{j}").as_bytes()]))
        .collect()
}

/// The evaluator's answers to a fresh request for `password` under `keys`,
/// as the prover receives them.
pub(crate) fn history_evaluation(password: &[u8], keys: &[pallas::Scalar]) -> HistoryEvaluation {
    let d = test_owner();
    let r = random_blind(OsRng);
    let b = blind_request(history_input(d, password), r);
    HistoryEvaluation {
        d,
        domains: test_domains(keys.len()),
        r,
        evaluations: keys
            .iter()
            .map(|k| evaluate(*k, b).expect("nonidentity"))
            .collect(),
    }
}

/// Circuit history inputs for `password` over `domains` fresh keys.
pub(crate) fn history_inputs(password: &[u8], domains: usize) -> HistoryInputs {
    let keys: Vec<_> = (0..domains)
        .map(|_| <pallas::Scalar as ff::Field>::random(OsRng))
        .collect();
    let eval = history_evaluation(password, &keys);
    let u = history_input(eval.d, password);
    HistoryInputs {
        d: eval.d,
        domains: eval.domains,
        witness: HistoryTagWitness::from_evaluations(u, eval.r, &eval.evaluations),
    }
}
