// SPDX-License-Identifier: AGPL-3.0-only
//! Verifier for OPAQUE-ZKPP registration proofs.
//!
//! A registration proof is accepted only if both hold:
//! 1. The SNARK verifies under the key of this verifier's shape (its fixed
//!    columns carry the policy minimums: a password below them has no valid
//!    proof), with the operation context absorbed into its transcript.
//! 2. Its first public instances are exactly the OPAQUE element `M` of the
//!    registration request the server received. The circuit computes
//!    `M = blind·H(P)` from the proved password, so the request is the
//!    blinded hash of that password.
//!
//! The history domains, request, evaluations and tags are returned for the
//! history checker.
//!
//! The SNARK check costs tens of milliseconds, so everything cheaper runs
//! first and a malformed submission is refused in microseconds: the exact
//! proof length of the shape, the instance count, the request element, then
//! every 32-byte element of the proof decoded as the point or scalar its
//! position holds. The positions are recorded once per verifier by a dry
//! run of the verifier over a counting transcript: halo2 reads a proof by
//! the key's structure alone, never by its values.

use ff::{Field, PrimeField};
use group::{CurveAffine as _, GroupEncoding};
use halo2_proofs::{
    plonk::{SingleVerifier, VerifyingKey, verify_proof},
    poly::commitment::Params,
    transcript::{Blake2bRead, Challenge255, EncodedChallenge, Transcript, TranscriptRead},
};
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::{pallas, vesta};
use std::io;

use crate::circuit::{CircuitShape, instance_count};
use crate::prover::BoundProof;
use crate::types::{DomainPublicInputs, HistoryTag, ZkppPublicInputs};

/// Error during proof verification.
#[derive(Debug, thiserror::Error)]
pub enum ZkppVerifyError {
    #[error("proof verification failed")]
    ProofInvalid,
    #[error("proof length mismatch: expected {expected} bytes, got {got}")]
    ProofLength { expected: usize, got: usize },
    #[error("proof element {at} is not a valid encoding")]
    ProofEncoding { at: usize },
    #[error("instance count mismatch: expected {expected}, got {got}")]
    InstanceCountMismatch { expected: usize, got: usize },
    #[error("the proof's OPAQUE element is not the request's")]
    ElementMismatch,
}

/// What each 32-byte element of a proof is, in reading order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Element {
    Point,
    Scalar,
}

/// A transcript that answers every read with a fixed valid value and records
/// what was read, so a dry run of the verifier yields the proof's layout.
struct LayoutRecorder {
    layout: Vec<Element>,
    squeezes: u8,
}

impl Transcript<vesta::Affine, Challenge255<vesta::Affine>> for LayoutRecorder {
    fn squeeze_challenge(&mut self) -> Challenge255<vesta::Affine> {
        // Distinct challenges, so no derived point coincides with another.
        self.squeezes = self.squeezes.wrapping_add(1);
        Challenge255::new(&[self.squeezes; 64])
    }
    fn common_point(&mut self, _point: vesta::Affine) -> io::Result<()> {
        Ok(())
    }
    fn common_scalar(&mut self, _scalar: vesta::Scalar) -> io::Result<()> {
        Ok(())
    }
}

impl TranscriptRead<vesta::Affine, Challenge255<vesta::Affine>> for LayoutRecorder {
    fn read_point(&mut self) -> io::Result<vesta::Affine> {
        self.layout.push(Element::Point);
        Ok(vesta::Affine::generator())
    }
    fn read_scalar(&mut self) -> io::Result<vesta::Scalar> {
        self.layout.push(Element::Scalar);
        Ok(vesta::Scalar::ONE + vesta::Scalar::from(self.layout.len() as u64))
    }
}

/// The element kinds of a proof under `vk`, in order.
fn proof_layout(
    params: &Params<vesta::Affine>,
    vk: &VerifyingKey<vesta::Affine>,
    shape: CircuitShape,
) -> Vec<Element> {
    let instances = vec![pallas::Base::ZERO; instance_count(shape.history_domains)];
    let mut recorder = LayoutRecorder {
        layout: Vec::new(),
        squeezes: 0,
    };
    let verdict = verify_proof(
        params,
        vk,
        SingleVerifier::new(params),
        &[&[&instances]],
        &mut recorder,
    );
    // The dry run reads a fixed point and fixed scalars: it never verifies,
    // and its verdict carries nothing but the reads it made.
    debug_assert!(verdict.is_err(), "a dry-run proof verified");
    recorder.layout
}

/// Domain separation of [`ZkppVerifier::artifact`].
const ARTIFACT_PURPOSE: &[u8] = b"SID-ZKPP-ARTIFACT-v2";

/// The identity of the verifier `params` and `vk` make: SHA-256 over the
/// serialized parameters and halo2's pinned verifying key (constraint system,
/// fixed commitments, which carry the policy constants, and permutation).
/// The parameters count on their own: verification uses their IPA generator
/// `u`, which no key commits to. Verifiers that accept different proofs
/// differ in it, so evidence naming it names exactly what verified.
fn artifact_of(params: &Params<vesta::Affine>, vk: &VerifyingKey<vesta::Affine>) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut serialized = Vec::new();
    params
        .write(&mut serialized)
        .expect("writing parameters to memory cannot fail");
    let pinned = format!("{:?}", vk.pinned());
    let mut hash = Sha256::new();
    hash.update(ARTIFACT_PURPOSE);
    for part in [serialized.as_slice(), pinned.as_bytes()] {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    hash.finalize().into()
}

/// ZKPP Verifier — verifies proofs and extracts public inputs.
pub struct ZkppVerifier {
    params: Params<vesta::Affine>,
    vk: VerifyingKey<vesta::Affine>,
    shape: CircuitShape,
    layout: Vec<Element>,
    artifact: [u8; 32],
}

impl ZkppVerifier {
    /// `vk` must be the key generated for `shape`, the policy and domain count
    /// this verifier enforces.
    pub fn new(
        params: Params<vesta::Affine>,
        vk: VerifyingKey<vesta::Affine>,
        shape: CircuitShape,
    ) -> Self {
        let layout = proof_layout(&params, &vk, shape);
        let artifact = artifact_of(&params, &vk);
        Self {
            params,
            vk,
            shape,
            layout,
            artifact,
        }
    }

    pub fn shape(&self) -> CircuitShape {
        self.shape
    }

    /// The identity of this verifier's artifact, recorded as the evidence of
    /// every proof it accepts so a retired artifact's verdicts can be found.
    pub fn artifact(&self) -> [u8; 32] {
        self.artifact
    }

    /// The exact length in bytes of every proof under this key.
    pub fn proof_len(&self) -> usize {
        self.layout.len() * 32
    }

    /// The length and instance count, then each element decoded as its
    /// position requires: the checks that cost microseconds.
    fn check_form(&self, bound: &BoundProof) -> Result<(), ZkppVerifyError> {
        let bytes = &bound.snark_proof.0;
        if bytes.len() != self.proof_len() {
            return Err(ZkppVerifyError::ProofLength {
                expected: self.proof_len(),
                got: bytes.len(),
            });
        }
        let expected = instance_count(self.shape.history_domains);
        if bound.instances.len() != expected {
            return Err(ZkppVerifyError::InstanceCountMismatch {
                expected,
                got: bound.instances.len(),
            });
        }
        // The length is a multiple of 32 (checked above), so nothing remains.
        let (elements, _) = bytes.as_chunks::<32>();
        for (at, (kind, repr)) in self.layout.iter().zip(elements).enumerate() {
            let valid = match kind {
                Element::Scalar => bool::from(vesta::Scalar::from_repr(*repr).is_some()),
                Element::Point => bool::from(vesta::Affine::from_bytes(repr).is_some()),
            };
            if !valid {
                return Err(ZkppVerifyError::ProofEncoding { at });
            }
        }
        Ok(())
    }

    /// The public inputs `bound` claims, after the cheap form checks and
    /// before the SNARK: for comparing with what the server already knows
    /// (the operation's domains, its request and the evaluator's answers),
    /// so a submission for another operation is refused without the SNARK.
    /// Nothing returned here is verified.
    pub fn claimed_inputs(&self, bound: &BoundProof) -> Result<ZkppPublicInputs, ZkppVerifyError> {
        self.check_form(bound)?;
        self.extract_public_inputs(&bound.instances)
    }

    /// Verify `bound` for the operation `context`
    /// ([`crate::binding::operation_context`]) and the request element
    /// `request_m`. Returns the history inputs for the checker.
    pub fn verify(
        &self,
        bound: &BoundProof,
        context: &[u8],
        request_m: pallas::Affine,
    ) -> Result<ZkppPublicInputs, ZkppVerifyError> {
        self.check_form(bound)?;
        // The element next: the request fixes M, a proof for any other M is
        // refused before the costly SNARK check.
        let m = request_m
            .coordinates()
            .into_option()
            .ok_or(ZkppVerifyError::ElementMismatch)?;
        if bound.instances[0] != *m.x() || bound.instances[1] != *m.y() {
            return Err(ZkppVerifyError::ElementMismatch);
        }

        let mut transcript = Blake2bRead::<&[u8], vesta::Affine, Challenge255<vesta::Affine>>::init(
            &bound.snark_proof.0[..],
        );
        transcript
            .common_scalar(crate::binding::transcript_context(context))
            .map_err(|_| ZkppVerifyError::ProofInvalid)?;
        verify_proof(
            &self.params,
            &self.vk,
            SingleVerifier::new(&self.params),
            &[&[&bound.instances]],
            &mut transcript,
        )
        .map_err(|_| ZkppVerifyError::ProofInvalid)?;

        self.extract_public_inputs(&bound.instances)
    }

    /// The history public inputs of verified instances (layout in
    /// [`crate::circuit::ZkppCircuit`]).
    fn extract_public_inputs(
        &self,
        instances: &[pallas::Base],
    ) -> Result<ZkppPublicInputs, ZkppVerifyError> {
        let domains = self.shape.history_domains;
        let b_at = 3 + domains;
        let blinded = point(instances[b_at], instances[b_at + 1])?.to_bytes();
        let domains = (0..domains)
            .map(|j| {
                let at = b_at + 2 + 3 * j;
                Ok(DomainPublicInputs {
                    comparison_domain: instances[3 + j].to_repr(),
                    evaluated: point(instances[at], instances[at + 1])?.to_bytes(),
                    tag: HistoryTag::new(instances[at + 2].to_repr()),
                })
            })
            .collect::<Result<_, ZkppVerifyError>>()?;
        Ok(ZkppPublicInputs {
            owner_domain: instances[2].to_repr(),
            blinded,
            domains,
        })
    }
}

#[cfg(test)]
mod tests;

/// The affine point with these coordinates; the identity (the ECC chip's
/// `(0, 0)`) and off-curve pairs are refused.
fn point(x: pallas::Base, y: pallas::Base) -> Result<pallas::Affine, ZkppVerifyError> {
    Option::<pallas::Affine>::from(pallas::Affine::from_xy(x, y))
        .filter(|p| !bool::from(group::CurveAffine::is_identity(p)))
        .ok_or(ZkppVerifyError::ProofInvalid)
}
