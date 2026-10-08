// SPDX-License-Identifier: AGPL-3.0-only
//! Prover for OPAQUE-ZKPP proofs.
//!
//! Creates Halo2 proofs for the combined ZKPP circuit.
//! Used server-side for testing and client-side (WASM) for production.

use ff::PrimeField;
use halo2_proofs::{
    plonk::{ProvingKey, create_proof},
    poly::commitment::Params,
    transcript::{Blake2bWrite, Challenge255, Transcript},
};
use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;

use crate::circuit::{
    BREACH_PARAMS, CircuitShape, HistoryInputs, ZkppCircuit, gadget_d::BloomFilter,
    gadget_h::HistoryTagWitness,
};
use crate::types::{MAX_PASSWORD_LEN, ZkppProof};

/// Error during proof generation.
#[derive(Debug, thiserror::Error)]
pub enum ZkppProveError {
    #[error("proof generation failed: {0}")]
    ProofFailed(String),
    #[error("password too long (max {MAX_PASSWORD_LEN} bytes)")]
    PasswordTooLong,
    #[error("history evaluations for {got} domains, the key is for {expected}")]
    HistoryShape { expected: usize, got: usize },
    /// The OPRF blind is not below the base-field modulus (probability about
    /// 2^-126); the client draws a new one.
    #[error("OPRF blind outside the base field; start the registration again")]
    BlindOutOfRange,
}

/// A registration proof: the SNARK and its public instances, `M` (the OPAQUE
/// element it computed) first. It is bound to one operation and request by
/// the context absorbed into its transcript; the verifier compares `M` with
/// the element of that request.
pub struct BoundProof {
    pub snark_proof: ZkppProof,
    pub instances: Vec<pasta_curves::pallas::Base>,
}

/// The history half of a proof's inputs: the owner domain `d` and comparison
/// domains `c_j` the server named in its preparation, the blind `r` of the
/// request `B = r·H` the client sent, and the evaluator's answers `Z_j` in the
/// order of `domains`.
pub struct HistoryEvaluation {
    pub d: pallas::Base,
    pub domains: Vec<pallas::Base>,
    pub r: pallas::Base,
    pub evaluations: Vec<pallas::Affine>,
}

/// ZKPP Prover — generates proofs for the combined circuit.
pub struct ZkppProver {
    params: Params<vesta::Affine>,
    pk: ProvingKey<vesta::Affine>,
    shape: CircuitShape,
}

impl ZkppProver {
    pub fn new(
        params: Params<vesta::Affine>,
        pk: ProvingKey<vesta::Affine>,
        shape: CircuitShape,
    ) -> Self {
        Self { params, pk, shape }
    }

    pub fn shape(&self) -> CircuitShape {
        self.shape
    }

    pub fn params(&self) -> &Params<vesta::Affine> {
        &self.params
    }

    pub fn pk(&self) -> &ProvingKey<vesta::Affine> {
        &self.pk
    }

    /// Prove a new password (registration, change or authorized reset): it
    /// satisfies the key's policy, is not breached, the OPAQUE element is
    /// `M = blind·H(P)` of it, and its history tags derive from the
    /// evaluator's answers. `context` is the operation context
    /// ([`crate::binding::operation_context`]). Whether the tags repeat a
    /// retained password is the history checker's decision.
    pub fn prove(
        &self,
        password: &[u8],
        blind: pallas::Scalar,
        context: &[u8],
        history: &HistoryEvaluation,
    ) -> Result<BoundProof, ZkppProveError> {
        if history.domains.len() != self.shape.history_domains
            || history.evaluations.len() != self.shape.history_domains
        {
            return Err(ZkppProveError::HistoryShape {
                expected: self.shape.history_domains,
                got: history.evaluations.len(),
            });
        }
        let mut pw_buf = [0u8; MAX_PASSWORD_LEN];
        pw_buf
            .get_mut(..password.len())
            .ok_or(ZkppProveError::PasswordTooLong)?
            .copy_from_slice(password);

        let blind = Option::<pallas::Base>::from(pallas::Base::from_repr(blind.to_repr()))
            .ok_or(ZkppProveError::BlindOutOfRange)?;
        let u = crate::history::history_input(history.d, &pw_buf);
        let circuit = ZkppCircuit {
            password: pw_buf,
            // Bounded by MAX_PASSWORD_LEN, checked above.
            password_len: password.len() as u32,
            blind,
            policy: self.shape.policy,
            breach_filter: BloomFilter::new(BREACH_PARAMS),
            history_domains: self.shape.history_domains,
            history: Some(HistoryInputs {
                d: history.d,
                domains: history.domains.clone(),
                witness: HistoryTagWitness::from_evaluations(u, history.r, &history.evaluations),
            }),
        };
        let (snark_proof, instances) = self.create_proof_for(circuit, context)?;
        Ok(BoundProof {
            snark_proof,
            instances,
        })
    }

    fn create_proof_for(
        &self,
        circuit: ZkppCircuit,
        context: &[u8],
    ) -> Result<(ZkppProof, Vec<pallas::Base>), ZkppProveError> {
        let instances = circuit.instance_values();

        let mut transcript =
            Blake2bWrite::<Vec<u8>, vesta::Affine, Challenge255<vesta::Affine>>::init(vec![]);
        // The operation context before any proof message: every challenge
        // depends on it, so the proof verifies only under the same context.
        transcript
            .common_scalar(crate::binding::transcript_context(context))
            .map_err(|e| ZkppProveError::ProofFailed(format!("{e:?}")))?;

        create_proof(
            &self.params,
            &self.pk,
            &[circuit],
            &[&[&instances]],
            OsRng,
            &mut transcript,
        )
        .map_err(|e| ZkppProveError::ProofFailed(format!("{e:?}")))?;

        let proof_bytes = transcript.finalize();
        Ok((ZkppProof(proof_bytes), instances))
    }
}
