// SPDX-License-Identifier: AGPL-3.0-only
//! History fixture shared by the integration tests: the evaluator's proven
//! answers to a fresh request, taken through the client's own checks.

use ff::Field;
use group::{Curve, Group};
use pasta_curves::pallas;
use sid_pake_core::client::{ComparisonDomain, EvaluatorAnswer, history_request, verified_history};
use sid_pake_core::history::{domain_element, evaluate_with_proof};
use sid_pake_core::prover::HistoryEvaluation;

/// The operation the fixture's evaluations are proven for.
pub const OPERATION: [u8; 16] = [0x5a; 16];

pub fn history_evaluation(password: &[u8]) -> HistoryEvaluation {
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"]);
    let request = history_request(&mut rand::rng(), d, password).expect("provable password");
    let k = pallas::Scalar::random(&mut rand::rng());
    let domain = ComparisonDomain {
        domain: domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-0"]),
        evaluator_public_key: (pallas::Point::generator() * k).to_affine(),
    };
    let (evaluation, proof) =
        evaluate_with_proof(k, request.blinded, &OPERATION, rand::rng()).expect("nonidentity");
    verified_history(
        password,
        &OPERATION,
        &request,
        &[domain],
        &[EvaluatorAnswer { evaluation, proof }],
    )
    .expect("the evaluator's answer verifies")
}
