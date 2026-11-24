// SPDX-License-Identifier: AGPL-3.0-only
//! The built-in password policies and their versions.
//!
//! A version names one compiled policy: its minimums and length unit are
//! fixed columns and a lookup table of the proving and verifying keys, so a
//! proof verifies only under the keys of the version it was made for. A
//! released version never changes its meaning; a different policy is a new
//! version. The credential issuer selects the version in force.

use crate::types::{
    CE_DEFAULT_POLICY, LONG_POLICY, LengthUnit, PolicyParams, PolicyVersion, SHORT_POLICY,
    STANDARD_POLICY, starts_code_point,
};

/// Version 1: 8 bytes with an ASCII uppercase letter, a lowercase letter and
/// a digit.
pub const CE_POLICY_VERSION: PolicyVersion = PolicyVersion(1);

/// Version 2: 12 Unicode code points, no required character class.
pub const STANDARD_POLICY_VERSION: PolicyVersion = PolicyVersion(2);

/// Version 3: 8 Unicode code points, no required character class.
pub const SHORT_POLICY_VERSION: PolicyVersion = PolicyVersion(3);

/// Version 4: 15 Unicode code points, no required character class.
pub const LONG_POLICY_VERSION: PolicyVersion = PolicyVersion(4);

/// Every built-in version, in order.
pub const BUILT_IN_VERSIONS: [PolicyVersion; 4] = [
    CE_POLICY_VERSION,
    STANDARD_POLICY_VERSION,
    SHORT_POLICY_VERSION,
    LONG_POLICY_VERSION,
];

/// The policy of a built-in `version`; none for a version this build does
/// not carry.
pub fn get_policy(version: PolicyVersion) -> Option<PolicyParams> {
    match version.0 {
        1 => Some(CE_DEFAULT_POLICY),
        2 => Some(STANDARD_POLICY),
        3 => Some(SHORT_POLICY),
        4 => Some(LONG_POLICY),
        _ => None,
    }
}

/// The length of `password` in the policy's `unit`, as the circuit counts
/// it.
pub fn length_in(unit: LengthUnit, password: &[u8]) -> usize {
    match unit {
        LengthUnit::Bytes => password.len(),
        LengthUnit::CodePoints => password.iter().filter(|b| starts_code_point(**b)).count(),
    }
}

/// Check `password` against `policy` where the password is at hand, for the
/// user's feedback: what a proof would establish, with the reasons. The
/// proof, not this check, is what a server accepts.
pub fn validate_password_client_side(password: &str, policy: &PolicyParams) -> Vec<String> {
    let mut errors = vec![];
    let len = length_in(policy.length_unit, password.as_bytes());

    if len < policy.min_length as usize {
        errors.push(format!(
            "Password too short: minimum {} characters, got {}",
            policy.min_length, len
        ));
    }

    let mut upper_count = 0u32;
    let mut lower_count = 0u32;
    let mut digit_count = 0u32;
    let mut symbol_count = 0u32;

    for ch in password.chars() {
        if ch.is_ascii_uppercase() {
            upper_count += 1;
        } else if ch.is_ascii_lowercase() {
            lower_count += 1;
        } else if ch.is_ascii_digit() {
            digit_count += 1;
        } else if ch.is_ascii_punctuation() {
            symbol_count += 1;
        }
    }

    if upper_count < policy.min_upper {
        errors.push(format!(
            "Need at least {} uppercase letters, got {}",
            policy.min_upper, upper_count
        ));
    }
    if lower_count < policy.min_lower {
        errors.push(format!(
            "Need at least {} lowercase letters, got {}",
            policy.min_lower, lower_count
        ));
    }
    if digit_count < policy.min_digit {
        errors.push(format!(
            "Need at least {} digits, got {}",
            policy.min_digit, digit_count
        ));
    }
    if symbol_count < policy.min_symbol {
        errors.push(format!(
            "Need at least {} symbols, got {}",
            policy.min_symbol, symbol_count
        ));
    }

    errors
}

#[cfg(test)]
mod tests;
