// SPDX-License-Identifier: AGPL-3.0-only
//! StructuredID OPAQUE-ZKPP: Zero-Knowledge Password Policy circuits.
//!
//! Implements the OPAQUE-ZKPP extension (draft-structured-opaque-zkpp) using
//! Halo2 proof system on Pallas curve. Gadgets:
//! - Gadget A: Policy Engine (ASCII lookup tables for character classification)
//! - Gadget C: Opaque-Binder (HashToCurve + the OPAQUE element M = blind·H_p)
//! - Gadget D: breach Bloom non-membership
//! - Gadget H: history tags for the history checker
//!
//! Reference: Prudnikov, D. (2026). "Zero-Knowledge Proof System for Password Policy
//! Verification in Asymmetric Password-Authenticated Key Exchange."
//! doi:[10.5281/zenodo.19387561](https://doi.org/10.5281/zenodo.19387561)

pub mod binding;
pub mod circuit;
pub mod history;
pub mod keygen;
pub mod pallas_opaque;
pub mod policy;
pub mod poseidon;
pub mod prover;
pub mod types;
pub mod verifier;

#[cfg(test)]
mod test_support;
