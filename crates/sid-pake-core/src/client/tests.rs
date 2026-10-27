use super::*;
use crate::circuit::gadget_c::hash_to_curve_outside;
use crate::history::{domain_element, evaluate_with_proof};
use ff::Field;
use group::{Curve, Group, GroupEncoding};

const OPERATION: [u8; 16] = [7; 16];
const PASSWORD: &[u8] = b"Str0ngP@ssword!";

/// A fresh evaluator key.
fn key() -> pallas::Scalar {
    pallas::Scalar::random(&mut rand::rng())
}

fn owner() -> pallas::Base {
    domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"])
}

/// One comparison domain under the evaluator key `k`.
fn domain(k: pallas::Scalar, epoch: &[u8]) -> ComparisonDomain {
    ComparisonDomain {
        domain: domain_element(b"SID-HISTORY-TAG-v1", &[epoch]),
        evaluator_public_key: (pallas::Point::generator() * k).to_affine(),
    }
}

/// The evaluator's proven answer under `k` to `blinded` for `operation`.
fn answer(k: pallas::Scalar, blinded: pallas::Affine, operation: &[u8]) -> EvaluatorAnswer {
    let (evaluation, proof) =
        evaluate_with_proof(k, blinded, operation, rand::rng()).expect("nonidentity");
    EvaluatorAnswer { evaluation, proof }
}

// The request a start sends is the OPRF element of exactly the blind it
// reports, over the password padded as the circuit hashes it, and the circuit
// can hold that blind: a proof made with it binds this request.
#[test]
fn a_start_reports_the_blind_of_its_own_request() {
    let start = start_registration(&mut rand::rng(), PASSWORD).unwrap();
    let mut padded = PASSWORD.to_vec();
    padded.resize(MAX_PASSWORD_LEN, 0);
    let (h, _, _) = hash_to_curve_outside(&padded);
    let element = (pallas::Point::from(h) * start.blind()).to_affine();
    assert_eq!(
        start.request.serialize().as_slice(),
        element.to_bytes().as_ref()
    );
    assert!(bool::from(
        pallas::Base::from_repr(start.blind().to_repr()).is_some()
    ));
}

// A state kept and restored between calls still yields the blind of its
// request.
#[test]
fn a_restored_state_yields_the_same_blind() {
    let start = start_registration(&mut rand::rng(), PASSWORD).unwrap();
    let restored =
        ClientRegistration::<PallasCipherSuite>::deserialize(&start.state.serialize()).unwrap();
    assert_eq!(registration_blind(&restored), start.blind());
}

// Two starts of one password draw independent blinds.
#[test]
fn each_start_draws_a_fresh_blind() {
    let a = start_registration(&mut rand::rng(), PASSWORD).unwrap();
    let b = start_registration(&mut rand::rng(), PASSWORD).unwrap();
    assert_ne!(a.blind(), b.blind());
}

// An empty password is refused before OPAQUE runs.
#[test]
fn an_empty_password_cannot_start() {
    assert!(matches!(
        start_registration(&mut rand::rng(), b""),
        Err(StartError::EmptyPassword)
    ));
}

// A password longer than the circuit holds has no history request: it cannot
// be proven, so nothing is evaluated for it.
#[test]
fn a_password_the_circuit_cannot_hold_has_no_history_request() {
    let long = [b'a'; MAX_PASSWORD_LEN + 1];
    assert!(history_request(&mut rand::rng(), owner(), &long).is_none());
    assert!(history_request(&mut rand::rng(), owner(), &long[..MAX_PASSWORD_LEN]).is_some());
}

// The history request is r·H for the password's canonical point under the
// owner domain, with the blind it reports.
#[test]
fn the_history_request_is_blinded_by_its_own_blind() {
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    assert_eq!(
        request.blinded,
        crate::history::blind_request(history_input(owner(), PASSWORD), request.blind)
    );
}

// Answers whose proofs verify for this operation become the prover's history
// inputs, in the order of the server's domains.
#[test]
fn verified_answers_are_the_provers_history_inputs() {
    let (k0, k1) = (key(), key());
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let domains = [domain(k0, b"epoch-0"), domain(k1, b"epoch-1")];
    let answers = [
        answer(k0, request.blinded, &OPERATION),
        answer(k1, request.blinded, &OPERATION),
    ];
    let history = verified_history(PASSWORD, &OPERATION, &request, &domains, &answers).unwrap();
    assert_eq!(history.d, owner());
    assert_eq!(history.r, request.blind);
    assert_eq!(history.domains, vec![domains[0].domain, domains[1].domain]);
    assert_eq!(
        history.evaluations,
        vec![answers[0].evaluation, answers[1].evaluation]
    );
}

// An answer proven for another operation is refused: an evaluation cannot be
// replayed from one operation into another.
#[test]
fn an_answer_for_another_operation_is_refused() {
    let k = key();
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let answers = [answer(k, request.blinded, &[8; 16])];
    assert!(matches!(
        verified_history(PASSWORD, &OPERATION, &request, &[domain(k, b"e")], &answers),
        Err(HistoryAnswerError::Proof { index: 0 })
    ));
}

// An answer under another key than the domain's evaluator key is refused.
#[test]
fn an_answer_under_another_key_is_refused() {
    let (k, other) = (key(), key());
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let domains = [domain(k, b"e0"), domain(k, b"e1")];
    let answers = [
        answer(k, request.blinded, &OPERATION),
        answer(other, request.blinded, &OPERATION),
    ];
    assert!(matches!(
        verified_history(PASSWORD, &OPERATION, &request, &domains, &answers),
        Err(HistoryAnswerError::Proof { index: 1 })
    ));
}

// An answer to another blinded request (another password, or another blind)
// is refused for this one.
#[test]
fn an_answer_to_another_request_is_refused() {
    let k = key();
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let other = history_request(&mut rand::rng(), owner(), b"An0therP@ss!").unwrap();
    let answers = [answer(k, other.blinded, &OPERATION)];
    assert!(matches!(
        verified_history(PASSWORD, &OPERATION, &request, &[domain(k, b"e")], &answers),
        Err(HistoryAnswerError::Proof { index: 0 })
    ));
}

// The request must be the one this password makes: answers proven for its
// blinded element do not cover another password proven over them.
#[test]
fn answers_for_one_password_do_not_cover_another() {
    let k = key();
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let answers = [answer(k, request.blinded, &OPERATION)];
    assert!(matches!(
        verified_history(
            b"An0therP@ss!",
            &OPERATION,
            &request,
            &[domain(k, b"e")],
            &answers
        ),
        Err(HistoryAnswerError::Proof { index: 0 })
    ));
}

// One answer per domain: a missing or an extra answer is refused, never
// proven over a shorter list.
#[test]
fn one_answer_per_domain_is_required() {
    let k = key();
    let request = history_request(&mut rand::rng(), owner(), PASSWORD).unwrap();
    let domains = [domain(k, b"e0"), domain(k, b"e1")];
    let one = [answer(k, request.blinded, &OPERATION)];
    assert!(matches!(
        verified_history(PASSWORD, &OPERATION, &request, &domains, &one),
        Err(HistoryAnswerError::Count {
            domains: 2,
            answers: 1
        })
    ));
    let three = [one[0], one[0], one[0]];
    assert!(matches!(
        verified_history(PASSWORD, &OPERATION, &request, &domains, &three),
        Err(HistoryAnswerError::Count {
            domains: 2,
            answers: 3
        })
    ));
}
