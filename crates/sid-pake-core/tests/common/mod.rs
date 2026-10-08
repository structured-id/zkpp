// SPDX-License-Identifier: AGPL-3.0-only
//! History fixture shared by the integration tests: the evaluator's answers
//! to a fresh request, as the prover receives them.

use ff::Field;
use pasta_curves::pallas;
use sid_pake_core::history::{
    blind_request, domain_element, evaluate, history_input, random_blind,
};
use sid_pake_core::prover::HistoryEvaluation;

pub fn history_evaluation(password: &[u8]) -> HistoryEvaluation {
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"]);
    let r = random_blind(rand::rng());
    let b = blind_request(history_input(d, password), r);
    let k = pallas::Scalar::random(&mut rand::rng());
    HistoryEvaluation {
        d,
        domains: vec![domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-0"])],
        r,
        evaluations: vec![evaluate(k, b).expect("nonidentity")],
    }
}
