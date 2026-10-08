// SPDX-License-Identifier: AGPL-3.0-only
//! A registration proof must bind the OPAQUE element `M` of the request to the
//! proved password: the circuit computes `M = blind·H(P)` and exposes it as
//! its first instances, and the operation context is absorbed into the
//! transcript. A strong password's valid proof must not carry any other
//! element (an unrelated point, a shifted one, the element of a weak
//! password) nor verify under another operation.

mod common;

use common::history_evaluation;
use ff::PrimeField;
use group::{Curve, Group};
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::pallas;
use sid_opaque_ke::ClientRegistration;
use sid_pake_core::binding::operation_context;
use sid_pake_core::circuit::CircuitShape;
use sid_pake_core::circuit::gadget_c::hash_to_curve_outside;
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::pallas_opaque::PallasCipherSuite;
use sid_pake_core::prover::{BoundProof, ZkppProveError, ZkppProver};
use sid_pake_core::types::{CE_DEFAULT_POLICY, MAX_PASSWORD_LEN};
use sid_pake_core::verifier::{ZkppVerifier, ZkppVerifyError};

const CE: CircuitShape = CircuitShape::single_domain(CE_DEFAULT_POLICY);
const PASSWORD: &[u8] = b"Str0ngP@ssword!";
const OPERATION: [u8; 16] = [0x42; 16];

fn keys() -> (ZkppProver, ZkppVerifier) {
    let params = generate_params(sid_pake_core::circuit::ZKPP_K);
    let pk = generate_pk(&params, CE).expect("keygen");
    let verifier = ZkppVerifier::new(params.clone(), pk.get_vk().clone(), CE);
    (ZkppProver::new(params, pk, CE), verifier)
}

/// An honest registration: the OPAQUE request element, its blind, the
/// operation context and the proof for them.
struct Honest {
    m: pallas::Affine,
    blind: pallas::Scalar,
    context: Vec<u8>,
    proof: BoundProof,
}

fn honest(prover: &ZkppProver) -> Honest {
    loop {
        let start =
            ClientRegistration::<PallasCipherSuite>::start(&mut rand::rng(), PASSWORD).unwrap();
        let mut repr = <pallas::Scalar as PrimeField>::Repr::default();
        repr.copy_from_slice(&start.state.serialize()[..32]);
        let blind = pallas::Scalar::from_repr(repr).unwrap();
        let request = start.message.serialize();
        let context = operation_context(&OPERATION, &request);
        match prover.prove(PASSWORD, blind, &context, &history_evaluation(PASSWORD)) {
            Ok(proof) => {
                let m = element(PASSWORD, blind);
                assert_eq!(
                    request.as_slice(),
                    group::GroupEncoding::to_bytes(&m).as_ref()
                );
                return Honest {
                    m,
                    blind,
                    context,
                    proof,
                };
            }
            // The client redraws a blind outside the base field.
            Err(ZkppProveError::BlindOutOfRange) => continue,
            Err(e) => panic!("prove: {e}"),
        }
    }
}

/// `blind·H(P)` over the zero-padded password, as the circuit and the OPAQUE
/// suite hash it.
fn element(password: &[u8], blind: pallas::Scalar) -> pallas::Affine {
    let mut padded = [0u8; MAX_PASSWORD_LEN];
    padded[..password.len()].copy_from_slice(password);
    let (h, _, _) = hash_to_curve_outside(&padded);
    (pallas::Point::from(h) * blind).to_affine()
}

/// The proof's instances rewritten to name `m`, as a forger keeping the
/// strong password's SNARK would send them.
fn claiming(mut proof: BoundProof, m: pallas::Affine) -> BoundProof {
    let c = m.coordinates().unwrap();
    proof.instances[0] = *c.x();
    proof.instances[1] = *c.y();
    proof
}

/// Positive control: the honest proof verifies for its request and operation.
#[test]
fn the_honest_proof_verifies() {
    let (prover, verifier) = keys();
    let h = honest(&prover);
    verifier
        .verify(&h.proof, &h.context, h.m)
        .expect("honest proof");
}

/// A request carrying an unrelated element is refused whether the instances
/// keep the honest `M` (mismatch) or are rewritten to the new one (the SNARK
/// no longer holds).
#[test]
fn an_unrelated_element_is_refused() {
    let (prover, verifier) = keys();
    let h = honest(&prover);
    let other = pallas::Point::hash_to_curve("SID_TEST_UNRELATED")(b"").to_affine();
    assert!(matches!(
        verifier.verify(&h.proof, &h.context, other),
        Err(ZkppVerifyError::ElementMismatch)
    ));
    assert!(matches!(
        verifier.verify(&claiming(h.proof, other), &h.context, other),
        Err(ZkppVerifyError::ProofInvalid)
    ));
}

/// The honest element shifted by a known point: nonzero relation to `M`, still
/// not the blinded hash of the proved password.
#[test]
fn a_shifted_element_is_refused() {
    let (prover, verifier) = keys();
    let h = honest(&prover);
    let shifted = (pallas::Point::from(h.m) + pallas::Point::generator()).to_affine();
    assert!(
        verifier
            .verify(&claiming(h.proof, shifted), &h.context, shifted)
            .is_err()
    );
}

/// The element of a weak password under the same blind: registering it behind
/// the strong password's proof is the substitution the binding exists to stop.
#[test]
fn a_weak_passwords_element_is_refused() {
    let (prover, verifier) = keys();
    let h = honest(&prover);
    let weak = element(b"abc", h.blind);
    assert!(
        verifier
            .verify(&claiming(h.proof, weak), &h.context, weak)
            .is_err()
    );
}

/// A proof made for one operation does not verify under another operation id
/// or another request with the same element.
#[test]
fn a_proof_does_not_verify_under_another_operation() {
    let (prover, verifier) = keys();
    let h = honest(&prover);
    let request = group::GroupEncoding::to_bytes(&h.m);
    let other_op = operation_context(&[0x43; 16], request.as_ref());
    assert!(matches!(
        verifier.verify(&h.proof, &other_op, h.m),
        Err(ZkppVerifyError::ProofInvalid)
    ));
}
