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

/// What a policy's minimum length counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LengthUnit {
    /// The password's bytes.
    Bytes,
    /// Unicode code points: the bytes that start one in UTF-8, so a
    /// multibyte character counts once.
    CodePoints,
}

/// Password policy parameters baked into the circuit at compile time.
/// Changing policy = recompiling circuit + new proving/verification keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyParams {
    /// The least length, in `length_unit`.
    pub min_length: u32,
    pub min_upper: u32,
    pub min_lower: u32,
    pub min_digit: u32,
    /// The least count of ASCII symbols. Zero under
    /// [`LengthUnit::CodePoints`]: the circuit's fourth class count carries
    /// the code points there (see [`PolicyParams::check`]).
    pub min_symbol: u32,
    pub length_unit: LengthUnit,
}

/// Why a policy cannot be compiled into a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PolicyError {
    /// A minimum no password of [`MAX_PASSWORD_LEN`] bytes can meet.
    #[error("a policy minimum exceeds the longest password")]
    MinimumTooLarge,
    /// A length in code points together with a symbol minimum.
    #[error("a policy counted in code points cannot require symbols")]
    SymbolsWithCodePoints,
}

impl PolicyParams {
    /// A policy of `min_length` Unicode code points with no required
    /// character class.
    pub const fn code_points(min_length: u32) -> Self {
        Self {
            min_length,
            min_upper: 0,
            min_lower: 0,
            min_digit: 0,
            min_symbol: 0,
            length_unit: LengthUnit::CodePoints,
        }
    }

    /// Whether the circuit can hold this policy.
    pub fn check(&self) -> Result<(), PolicyError> {
        let longest = MAX_PASSWORD_LEN as u32;
        if [
            self.min_length,
            self.min_upper,
            self.min_lower,
            self.min_digit,
            self.min_symbol,
        ]
        .iter()
        .any(|min| *min > longest)
        {
            return Err(PolicyError::MinimumTooLarge);
        }
        if self.length_unit == LengthUnit::CodePoints && self.min_symbol != 0 {
            return Err(PolicyError::SymbolsWithCodePoints);
        }
        Ok(())
    }

    /// The circuit's five fixed minimums, in the order of its comparison
    /// rows: active bytes, uppercase, lowercase, digits and the fourth class
    /// (ASCII symbols, or code points under [`LengthUnit::CodePoints`]).
    pub fn circuit_minimums(&self) -> [u32; 5] {
        match self.length_unit {
            LengthUnit::Bytes => [
                self.min_length,
                self.min_upper,
                self.min_lower,
                self.min_digit,
                self.min_symbol,
            ],
            // Every code point has at least one byte, so the byte minimum
            // adds nothing and stays zero.
            LengthUnit::CodePoints => [
                0,
                self.min_upper,
                self.min_lower,
                self.min_digit,
                self.min_length,
            ],
        }
    }
}

/// Whether `byte` starts a code point in well-formed UTF-8 (RFC 3629
/// section 3): an ASCII byte other than NUL, which pads the password, or a
/// lead byte of a two- to four-byte sequence.
pub const fn starts_code_point(byte: u8) -> bool {
    matches!(byte, 0x01..=0x7F | 0xC2..=0xF4)
}

/// Built-in policy 1: 8 bytes with an ASCII uppercase letter, a lowercase
/// letter and a digit. Released; its meaning and keys do not change.
pub const CE_DEFAULT_POLICY: PolicyParams = PolicyParams {
    min_length: 8,
    min_upper: 1,
    min_lower: 1,
    min_digit: 1,
    min_symbol: 0,
    length_unit: LengthUnit::Bytes,
};

/// Built-in policy 2: 12 code points, no required character class.
pub const STANDARD_POLICY: PolicyParams = PolicyParams::code_points(12);

/// Built-in policy 3: 8 code points, no required character class. For an
/// issuer that requires a second factor at every sign-in with the password.
pub const SHORT_POLICY: PolicyParams = PolicyParams::code_points(8);

/// Built-in policy 4: 15 code points, no required character class.
pub const LONG_POLICY: PolicyParams = PolicyParams::code_points(15);

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
