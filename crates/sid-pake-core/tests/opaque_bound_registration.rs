// SPDX-License-Identifier: AGPL-3.0-only
//! A bound proof must verify against the element of the OPAQUE registration
//! request that the same client sends: the server checks the proof's `M`
//! against the request, never against a value the proof carries.

mod common;

use common::history_evaluation;
use ff::PrimeField;
use group::GroupEncoding;
use pasta_curves::pallas;
use sid_opaque_ke::{ClientRegistration, ClientRegistrationStartResult, RegistrationRequest};
use sid_pake_core::circuit::CircuitShape;
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::pallas_opaque::PallasCipherSuite;
use sid_pake_core::prover::ZkppProver;
use sid_pake_core::types::CE_DEFAULT_POLICY;
use sid_pake_core::verifier::ZkppVerifier;

const CE: CircuitShape = CircuitShape::single_domain(CE_DEFAULT_POLICY);

/// The OPAQUE registration element `M`: the blinded element the request carries.
fn request_element(request: &RegistrationRequest<PallasCipherSuite>) -> pallas::Affine {
    let bytes: [u8; 32] = request
        .serialize()
        .as_slice()
        .try_into()
        .expect("a Pallas registration request is one compressed point");
    Option::from(pallas::Affine::from_bytes(&bytes)).expect("the request element is a point")
}

/// The OPRF blind the client drew, read from its registration state
/// (`OprfClient` = blind scalar, then the blinded element).
fn client_blind(state: &ClientRegistration<PallasCipherSuite>) -> pallas::Scalar {
    let mut repr = <pallas::Scalar as PrimeField>::Repr::default();
    repr.copy_from_slice(&state.serialize()[..32]);
    Option::from(pallas::Scalar::from_repr(repr)).expect("the state starts with the blind")
}

fn keys() -> (ZkppProver, ZkppVerifier) {
    let params = generate_params(sid_pake_core::circuit::ZKPP_K);
    let pk = generate_pk(&params, CE).expect("keygen");
    let verifier = ZkppVerifier::new(params.clone(), pk.get_vk().clone(), CE);
    (ZkppProver::new(params, pk, CE), verifier)
}

fn start(password: &[u8]) -> ClientRegistrationStartResult<PallasCipherSuite> {
    ClientRegistration::<PallasCipherSuite>::start(&mut rand::rng(), password)
        .expect("registration start")
}

/// A client proves its policy-valid password for the request it actually sends,
/// and the server accepts that proof against the request's own element.
#[test]
fn a_proof_for_the_registered_password_verifies_against_its_request() {
    let (prover, verifier) = keys();
    let password = b"Str0ngP@ssword!";
    let start = start(password);
    let request_bytes = start.message.serialize();
    let bound = prover
        .prove(
            password,
            client_blind(&start.state),
            &request_bytes,
            &history_evaluation(password),
        )
        .expect("prove");

    let public = verifier
        .verify(&bound, &request_bytes, request_element(&start.message))
        .expect("the proof binds the element the request carries");
    assert_eq!(
        public.domains.len(),
        1,
        "one history tag for the one domain"
    );
}

/// A proof made for a strong password does not certify a request for another one.
#[test]
fn a_proof_for_another_password_does_not_verify_against_the_request() {
    let (prover, verifier) = keys();
    let weak = start(b"weak");
    let request_bytes = weak.message.serialize();
    let strong = b"Str0ngP@ssword!";
    let bound = prover
        .prove(
            strong,
            client_blind(&weak.state),
            &request_bytes,
            &history_evaluation(strong),
        )
        .expect("prove");

    assert!(
        verifier
            .verify(&bound, &request_bytes, request_element(&weak.message))
            .is_err()
    );
}

/// The proof is bound to one request, not to the password: a second
/// registration of the same password cannot reuse it.
#[test]
fn a_proof_is_bound_to_one_request_not_to_the_password() {
    let (prover, verifier) = keys();
    let password = b"N3wP@ssw0rd!xY";
    let first = start(password);
    let request_bytes = first.message.serialize();
    let bound = prover
        .prove(
            password,
            client_blind(&first.state),
            &request_bytes,
            &history_evaluation(password),
        )
        .expect("prove");
    verifier
        .verify(&bound, &request_bytes, request_element(&first.message))
        .expect("the proof binds its own request");

    let other = start(password);
    assert!(
        verifier
            .verify(
                &bound,
                &other.message.serialize(),
                request_element(&other.message)
            )
            .is_err(),
        "the proof is bound to one request, not to the password"
    );
}
