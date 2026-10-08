// SPDX-License-Identifier: AGPL-3.0-only
//! Core types for OPAQUE-ZKPP (Zero-Knowledge Password Policy): the password
//! policy constraints and proof structures, which must match the Halo2
//! circuits exactly.

use serde::{Deserialize, Serialize};

/// Maximum password length in bytes (zero-padded to fixed size in circuit).
/// Per OPAQUE-ZKPP spec section 9.1.
pub const MAX_PASSWORD_LEN: usize = 128;

/// Policy version identifier.
/// Each version corresponds to a compiled circuit with specific policy constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolicyVersion(pub u32);

/// Password policy parameters baked into the circuit at compile time.
/// Changing policy = recompiling circuit + new proving/verification keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyParams {
    pub min_length: u32,
    pub min_upper: u32,
    pub min_lower: u32,
    pub min_digit: u32,
    pub min_symbol: u32,
}

/// CE default policy: NIST SP 800-63B baseline.
pub const CE_DEFAULT_POLICY: PolicyParams = PolicyParams {
    min_length: 8,
    min_upper: 1,
    min_lower: 1,
    min_digit: 1,
    min_symbol: 0,
};

/// Serialized ZK proof bytes (Halo2 IPA proof).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkppProof(pub Vec<u8>);

/// The history tag `t` of one comparison domain.
/// Together with that domain's VOPRF key it tests password guesses without the
/// KSF, so it is only a transient input of the history checker: zeroized on
/// drop, never printed, serialized, persisted or logged.
pub struct HistoryTag([u8; 32]);

impl HistoryTag {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The tag bytes, for the checker's KSF input only.
    pub fn expose(&self) -> &[u8; 32] {
        &self.0
    }
}

impl zeroize::Zeroize for HistoryTag {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl Drop for HistoryTag {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(self);
    }
}

impl zeroize::ZeroizeOnDrop for HistoryTag {}

impl core::fmt::Debug for HistoryTag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("HistoryTag(<redacted>)")
    }
}

/// One comparison domain of a verified proof: which domain, the evaluator's
/// answer the proof used, and the tag for the checker.
#[derive(Debug)]
pub struct DomainPublicInputs {
    /// Comparison-domain element `c_j` (epoch, format, KSF configuration).
    pub comparison_domain: [u8; 32],
    /// Evaluated element `Z_j`, compressed.
    pub evaluated: [u8; 32],
    pub tag: HistoryTag,
}

/// Public inputs extracted from a verified proof. Policy compliance is not
/// among them: a proof verifies only for a password meeting the policy of the
/// verifying key.
#[derive(Debug)]
pub struct ZkppPublicInputs {
    /// Owner-domain element `d` the history input was hashed under.
    pub owner_domain: [u8; 32],
    /// Blinded history request `B`, compressed.
    pub blinded: [u8; 32],
    /// One entry per comparison domain, in the verifying key's order.
    pub domains: Vec<DomainPublicInputs>,
}
