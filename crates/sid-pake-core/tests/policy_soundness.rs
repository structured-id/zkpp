// SPDX-License-Identifier: AGPL-3.0-only
//! The policy verdict must come from the server's verifying key, not from the
//! prover: a client that proves under a weaker policy of its own choosing must
//! not obtain a proof the server accepts.

mod common;

use common::history_evaluation;
use ff::PrimeField;
use group::GroupEncoding;
use opaque_ke::rand::rngs::OsRng;
use opaque_ke::{ClientRegistration, ClientRegistrationStartResult};
use pasta_curves::pallas;
use sid_pake_core::circuit::CircuitShape;
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::pallas_opaque::PallasCipherSuite;
use sid_pake_core::prover::ZkppProver;
use sid_pake_core::types::{CE_DEFAULT_POLICY, PolicyParams};
use sid_pake_core::verifier::ZkppVerifier;

const LAX: PolicyParams = PolicyParams {
    min_length: 0,
    min_upper: 0,
    min_lower: 0,
    min_digit: 0,
    min_symbol: 0,
};
const CE: CircuitShape = CircuitShape::single_domain(CE_DEFAULT_POLICY);
const LAX_SHAPE: CircuitShape = CircuitShape::single_domain(LAX);

fn start(password: &[u8]) -> ClientRegistrationStartResult<PallasCipherSuite> {
    ClientRegistration::<PallasCipherSuite>::start(&mut OsRng, password)
        .expect("registration start")
}

fn blind(start: &ClientRegistrationStartResult<PallasCipherSuite>) -> pallas::Scalar {
    let mut repr = <pallas::Scalar as PrimeField>::Repr::default();
    repr.copy_from_slice(&start.state.serialize()[..32]);
    pallas::Scalar::from_repr(repr).unwrap()
}

fn element(start: &ClientRegistrationStartResult<PallasCipherSuite>) -> pallas::Affine {
    let request = start.message.serialize();
    pallas::Affine::from_bytes(request.as_slice().try_into().unwrap()).unwrap()
}

/// A client builds its own keys for an all-zero policy and proves a weak
/// password with them; the server's verifier, built for its CE policy, refuses.
#[test]
fn a_prover_chosen_policy_does_not_make_a_weak_password_compliant() {
    let params = generate_params(sid_pake_core::circuit::ZKPP_K);
    let server_pk = generate_pk(&params, CE).expect("keygen");
    let verifier = ZkppVerifier::new(params.clone(), server_pk.get_vk().clone(), CE);
    let lax_pk = generate_pk(&params, LAX_SHAPE).expect("keygen");
    let prover = ZkppProver::new(params, lax_pk, LAX_SHAPE);

    let password = b"abc";
    let start = start(password);
    let request = start.message.serialize();
    let bound = prover
        .prove(
            password,
            blind(&start),
            &request,
            &history_evaluation(password),
        )
        .expect("prove");

    assert!(
        verifier.verify(&bound, &request, element(&start)).is_err(),
        "a weak password must not verify as policy-compliant"
    );
}

/// The same holds for a strong password: a proof made under another policy's
/// key is not a proof under the server's, whatever the password.
#[test]
fn a_proof_under_another_policy_key_is_refused() {
    let params = generate_params(sid_pake_core::circuit::ZKPP_K);
    let server_pk = generate_pk(&params, CE).expect("keygen");
    let verifier = ZkppVerifier::new(params.clone(), server_pk.get_vk().clone(), CE);
    let lax_pk = generate_pk(&params, LAX_SHAPE).expect("keygen");
    let prover = ZkppProver::new(params, lax_pk, LAX_SHAPE);

    let password = b"Str0ngP@ssword!";
    let start = start(password);
    let request = start.message.serialize();
    let bound = prover
        .prove(
            password,
            blind(&start),
            &request,
            &history_evaluation(password),
        )
        .expect("prove");

    assert!(verifier.verify(&bound, &request, element(&start)).is_err());
}

/// With the server's own key a weak password has no proof at all: the prover
/// finds no witness meeting the key's policy minimums.
#[test]
fn the_servers_key_does_not_prove_a_weak_password() {
    let params = generate_params(sid_pake_core::circuit::ZKPP_K);
    let pk = generate_pk(&params, CE).expect("keygen");
    let prover = ZkppProver::new(params, pk, CE);

    let password = b"abc";
    let start = start(password);
    let request = start.message.serialize();

    assert!(
        prover
            .prove(
                password,
                blind(&start),
                &request,
                &history_evaluation(password)
            )
            .is_err()
    );
}
