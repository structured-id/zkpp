//! The verifier refuses malformed submissions before the SNARK: a wrong
//! length, a wrong instance count, a foreign request element and any proof
//! element that is not the point or scalar its position holds. A well-formed
//! proof with a wrong value still reaches the SNARK and fails there.

use super::*;
use crate::circuit::ZKPP_K;
use crate::keygen::{generate_params, generate_pk};
use crate::prover::ZkppProver;
use crate::test_support::{history_evaluation, oprf_element};
use crate::types::{CE_DEFAULT_POLICY, ZkppProof};
use ff::Field;
use group::{Curve, Group};

const PASSWORD: &[u8] = b"Str0ngP@ssword!";
const CONTEXT: &[u8] = b"sid-op-context";

struct Fixture {
    verifier: ZkppVerifier,
    bound: BoundProof,
    m: pallas::Affine,
}

fn fixture(domains: usize) -> Fixture {
    let shape = CircuitShape {
        policy: CE_DEFAULT_POLICY,
        history_domains: domains,
    };
    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, shape).unwrap();
    let verifier = ZkppVerifier::new(params.clone(), pk.get_vk().clone(), shape);
    let prover = ZkppProver::new(params, pk, shape);
    let keys: Vec<_> = (0..domains)
        .map(|_| pallas::Scalar::random(&mut rand::rng()))
        .collect();
    let blind = pallas::Scalar::random(&mut rand::rng());
    let bound = prover
        .prove(
            PASSWORD,
            blind,
            CONTEXT,
            &history_evaluation(PASSWORD, &keys),
        )
        .unwrap();
    Fixture {
        verifier,
        bound,
        m: oprf_element(PASSWORD, blind),
    }
}

fn with_proof(bound: &BoundProof, bytes: Vec<u8>) -> BoundProof {
    BoundProof {
        snark_proof: ZkppProof(bytes),
        instances: bound.instances.clone(),
    }
}

/// The dry-run layout gives exactly the length real proofs have, for one
/// and for two comparison domains.
#[test]
fn proof_len_is_the_length_of_real_proofs() {
    for domains in [1, 2] {
        let f = fixture(domains);
        assert_eq!(f.verifier.proof_len(), f.bound.snark_proof.0.len());
        f.verifier
            .verify(&f.bound, CONTEXT, f.m)
            .expect("the honest proof verifies");
    }
}

/// A proof of any other length is refused by length, before anything is read.
#[test]
fn a_wrong_length_is_refused_by_length() {
    let f = fixture(1);
    let len = f.verifier.proof_len();
    for bytes in [
        vec![],
        f.bound.snark_proof.0[..len / 2].to_vec(),
        [f.bound.snark_proof.0.clone(), vec![0u8; 32]].concat(),
    ] {
        let got = bytes.len();
        assert!(matches!(
            f.verifier.verify(&with_proof(&f.bound, bytes), CONTEXT, f.m),
            Err(ZkppVerifyError::ProofLength { expected, got: g }) if expected == len && g == got
        ));
    }
}

/// Bytes that are no point where a point belongs, or no canonical scalar
/// where a scalar belongs, are refused at that element.
#[test]
fn a_bad_encoding_is_refused_at_its_element() {
    let f = fixture(1);
    // Element 0 is a point (the first advice commitment): 0xff.. decodes to
    // no point (x is not below the modulus).
    let mut bytes = f.bound.snark_proof.0.clone();
    bytes[..32].fill(0xff);
    assert!(matches!(
        f.verifier
            .verify(&with_proof(&f.bound, bytes), CONTEXT, f.m),
        Err(ZkppVerifyError::ProofEncoding { at: 0 })
    ));
    // The first scalar position: the modulus itself is not canonical.
    let at = f
        .verifier
        .layout
        .iter()
        .position(|e| *e == Element::Scalar)
        .unwrap();
    let mut bytes = f.bound.snark_proof.0.clone();
    let mut modulus = (-vesta::Scalar::ONE).to_repr();
    modulus[0] = modulus[0].wrapping_add(1); // p - 1 + 1 = p, little-endian
    bytes[32 * at..32 * at + 32].copy_from_slice(&modulus);
    assert!(matches!(
        f.verifier.verify(&with_proof(&f.bound, bytes), CONTEXT, f.m),
        Err(ZkppVerifyError::ProofEncoding { at: a }) if a == at
    ));
}

/// The instance count and the request element are checked before the SNARK.
#[test]
fn instances_and_element_are_checked_first() {
    let f = fixture(1);
    let mut short = f.bound.instances.clone();
    short.pop();
    let short = BoundProof {
        snark_proof: f.bound.snark_proof.clone(),
        instances: short,
    };
    assert!(matches!(
        f.verifier.verify(&short, CONTEXT, f.m),
        Err(ZkppVerifyError::InstanceCountMismatch { .. })
    ));
    let other = (pallas::Point::generator() * pallas::Scalar::random(&mut rand::rng())).to_affine();
    assert!(matches!(
        f.verifier.verify(&f.bound, CONTEXT, other),
        Err(ZkppVerifyError::ElementMismatch)
    ));
}

/// A well-formed proof with one wrong value passes the cheap checks and
/// fails the SNARK.
#[test]
fn a_wrong_value_fails_the_snark() {
    let f = fixture(1);
    let mut bytes = f.bound.snark_proof.0.clone();
    let last = bytes.len() - 32;
    // Replace the final scalar with another canonical one.
    bytes[last..].copy_from_slice(&vesta::Scalar::from(7u64).to_repr());
    let bad = with_proof(&f.bound, bytes);
    f.verifier.check_form(&bad).expect("well-formed");
    assert!(matches!(
        f.verifier.verify(&bad, CONTEXT, f.m),
        Err(ZkppVerifyError::ProofInvalid)
    ));
}

/// The claimed inputs are the verified ones for an honest proof, and
/// malformed proofs get no claimed inputs.
#[test]
fn claimed_inputs_match_the_verified_ones() {
    let f = fixture(2);
    let claimed = f.verifier.claimed_inputs(&f.bound).unwrap();
    let verified = f.verifier.verify(&f.bound, CONTEXT, f.m).unwrap();
    assert_eq!(claimed.owner_domain, verified.owner_domain);
    assert_eq!(claimed.blinded, verified.blinded);
    assert_eq!(claimed.domains.len(), 2);
    for (c, v) in claimed.domains.iter().zip(&verified.domains) {
        assert_eq!(c.comparison_domain, v.comparison_domain);
        assert_eq!(c.evaluated, v.evaluated);
        assert_eq!(c.tag.expose(), v.tag.expose());
    }
    let truncated = with_proof(&f.bound, f.bound.snark_proof.0[..64].to_vec());
    assert!(f.verifier.claimed_inputs(&truncated).is_err());
}
