// SPDX-License-Identifier: AGPL-3.0-only
//! Binding of a registration proof to its password operation.
//!
//! The circuit computes the OPAQUE registration element `M = b·H(P)` itself
//! and exposes it, so the proof shows that the element the client sends is
//! the blinded hash of the proved password. What remains outside the circuit
//! is which operation the proof belongs to: [`operation_context`] names the
//! operation and its registration request, and the prover and verifier both
//! absorb [`transcript_context`] of it into the proof transcript before any
//! other message. A proof made for one operation or request fails for any
//! other.

use ff::FromUniformBytes;
use pasta_curves::pallas;
use sha2::{Digest, Sha512};

/// Domain separator of a password operation's binding context.
const OPERATION_CONTEXT_DOMAIN: &[u8] = b"SID_ZKPP_PASSWORD_OPERATION_v1";
/// Domain separator of the transcript-context hash.
const TRANSCRIPT_CONTEXT_DOMAIN: &[u8] = b"SID_ZKPP_TRANSCRIPT_CONTEXT_v1";

/// The binding context of a password operation: its 16-byte operation ID and
/// the OPAQUE registration request the proof is for. The server keeps the rest
/// of what the proof must be bound to (purpose, owner, expected revisions,
/// policy) in that operation's state. Every field before the request has a
/// fixed length, so the encoding is unambiguous.
pub fn operation_context(operation_id: &[u8; 16], registration_request: &[u8]) -> Vec<u8> {
    let mut context =
        Vec::with_capacity(OPERATION_CONTEXT_DOMAIN.len() + 16 + registration_request.len());
    context.extend_from_slice(OPERATION_CONTEXT_DOMAIN);
    context.extend_from_slice(operation_id);
    context.extend_from_slice(registration_request);
    context
}

/// The scalar the prover and verifier absorb into the transcript for
/// `context`: a uniform reduction of its domain-separated SHA-512, so every
/// Fiat-Shamir challenge of the proof depends on it.
pub fn transcript_context(context: &[u8]) -> pallas::Base {
    let mut h = Sha512::new();
    h.update(TRANSCRIPT_CONTEXT_DOMAIN);
    h.update((context.len() as u64).to_le_bytes());
    h.update(context);
    let mut wide = [0u8; 64];
    wide.copy_from_slice(&h.finalize());
    pallas::Base::from_uniform_bytes(&wide)
}

#[cfg(test)]
mod tests;
