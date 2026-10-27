// SPDX-License-Identifier: AGPL-3.0-only
//! The client's side of a ZKPP-bound registration, around the prover: the
//! OPAQUE start whose OPRF blind the proof reuses, and the proof's history
//! inputs from the evaluator's verified answers. Every client of the proof
//! takes them from here instead of reading the blind out of the OPAQUE state
//! or checking the answers itself.

use ff::PrimeField;
use pasta_curves::pallas;
use rand_core::CryptoRng;
use sid_opaque_ke::errors::ProtocolError;
use sid_opaque_ke::{ClientRegistration, RegistrationRequest};

use crate::history::{
    EvaluationProof, blind_request, history_input, random_blind, verify_evaluation,
};
use crate::pallas_opaque::PallasCipherSuite;
use crate::prover::HistoryEvaluation;
use crate::types::MAX_PASSWORD_LEN;

/// OPAQUE starts drawn for a blind the circuit can hold. A blind at or above
/// the base-field modulus has probability about 2^-126 per start, so a second
/// failure means a broken random source, not bad luck.
pub const BLIND_ATTEMPTS: usize = 4;

/// A registration started for a proof: the OPAQUE client state for the
/// finish, the request to send, and the OPRF blind of that request.
pub struct RegistrationStart {
    pub state: ClientRegistration<PallasCipherSuite>,
    pub request: RegistrationRequest<PallasCipherSuite>,
    blind: pallas::Scalar,
}

impl RegistrationStart {
    /// The OPRF blind of [`Self::request`], below the base-field modulus: the
    /// blind [`crate::prover::ZkppProver::prove`] binds the proof to.
    pub fn blind(&self) -> pallas::Scalar {
        self.blind
    }
}

/// Why a registration could not start.
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("password cannot be empty")]
    EmptyPassword,
    #[error("registration start: {0}")]
    Opaque(ProtocolError),
    #[error("no usable OPRF blind drawn; the random source is broken")]
    NoUsableBlind,
}

/// Start the OPAQUE registration of `password` (a first password, a change or
/// an authorized reset), drawing until the circuit can hold the blind, so the
/// proof never has to start over.
pub fn start_registration<R: CryptoRng>(
    rng: &mut R,
    password: &[u8],
) -> Result<RegistrationStart, StartError> {
    if password.is_empty() {
        return Err(StartError::EmptyPassword);
    }
    for _ in 0..BLIND_ATTEMPTS {
        let started = ClientRegistration::<PallasCipherSuite>::start(rng, password)
            .map_err(StartError::Opaque)?;
        let blind = state_blind(&started.state);
        if bool::from(pallas::Base::from_repr(blind.to_repr()).is_some()) {
            return Ok(RegistrationStart {
                state: started.state,
                request: started.message,
                blind,
            });
        }
    }
    Err(StartError::NoUsableBlind)
}

/// The OPRF blind of a registration state: its serialization starts with it
/// (`OprfClient` = blind scalar, then the blinded element).
fn state_blind(state: &ClientRegistration<PallasCipherSuite>) -> pallas::Scalar {
    let mut repr = <pallas::Scalar as PrimeField>::Repr::default();
    let len = repr.as_ref().len();
    repr.as_mut().copy_from_slice(&state.serialize()[..len]);
    Option::from(pallas::Scalar::from_repr(repr)).expect("the state starts with a canonical scalar")
}

/// The history request of a password: the blinded element `B = r·H` sent to
/// the evaluator and the blind `r` the proof uses, for one owner domain.
pub struct HistoryRequest {
    pub owner_domain: pallas::Base,
    pub blind: pallas::Base,
    pub blinded: pallas::Affine,
}

/// The history request of `password` under the owner domain the server named;
/// `None` for a password longer than the circuit holds, which has no proof
/// and so nothing to evaluate.
pub fn history_request<R: CryptoRng>(
    rng: &mut R,
    owner_domain: pallas::Base,
    password: &[u8],
) -> Option<HistoryRequest> {
    if password.len() > MAX_PASSWORD_LEN {
        return None;
    }
    let blind = random_blind(rng);
    Some(HistoryRequest {
        owner_domain,
        blind,
        blinded: blind_request(history_input(owner_domain, password), blind),
    })
}

/// One comparison domain of the server's preparation: the domain element and
/// the public key of the evaluator that answers for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComparisonDomain {
    pub domain: pallas::Base,
    pub evaluator_public_key: pallas::Affine,
}

/// The evaluator's answer for one domain: `Z = k·B` and its proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluatorAnswer {
    pub evaluation: pallas::Affine,
    pub proof: EvaluationProof,
}

/// Why the evaluator's answers cannot be proven over.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HistoryAnswerError {
    #[error("{answers} evaluator answers for {domains} comparison domains")]
    Count { domains: usize, answers: usize },
    #[error("the evaluator's proof for domain {index} does not verify")]
    Proof { index: usize },
}

/// The proof's history inputs: one answer per domain of the server's
/// preparation, in its order, each accepted only with a proof under that
/// domain's key, for this `operation_id` and for the request this `password`
/// makes under `request`'s owner domain and blind.
pub fn verified_history(
    password: &[u8],
    operation_id: &[u8; 16],
    request: &HistoryRequest,
    domains: &[ComparisonDomain],
    answers: &[EvaluatorAnswer],
) -> Result<HistoryEvaluation, HistoryAnswerError> {
    if answers.len() != domains.len() {
        return Err(HistoryAnswerError::Count {
            domains: domains.len(),
            answers: answers.len(),
        });
    }
    // Recomputed from the password, never taken from the request: the
    // answers must be for the element this password makes.
    let blinded = blind_request(history_input(request.owner_domain, password), request.blind);
    for (index, (domain, answer)) in domains.iter().zip(answers).enumerate() {
        if !verify_evaluation(
            domain.evaluator_public_key,
            blinded,
            answer.evaluation,
            operation_id,
            &answer.proof,
        ) {
            return Err(HistoryAnswerError::Proof { index });
        }
    }
    Ok(HistoryEvaluation {
        d: request.owner_domain,
        domains: domains.iter().map(|d| d.domain).collect(),
        r: request.blind,
        evaluations: answers.iter().map(|a| a.evaluation).collect(),
    })
}

#[cfg(test)]
mod tests;
