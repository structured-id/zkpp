// SPDX-License-Identifier: AGPL-3.0-only
//! Password policy versioning and lookup.
//!
//! CE ships with a single hardcoded policy (NIST SP 800-63B baseline).
//! EE extends with circuit compiler + DSL for custom policies.
//!
//! Policy version must match the Halo2 circuit version exactly.
//! Mismatches result in proof verification failure.

use crate::types::{CE_DEFAULT_POLICY, PolicyParams, PolicyVersion};

/// CE policy version: v1 (NIST SP 800-63B baseline).
pub const CE_POLICY_VERSION: PolicyVersion = PolicyVersion(1);

/// Get policy parameters by version.
///
/// CE: only v1 exists. EE would load from compiled circuit configs.
/// Returns None if version is not available.
pub fn get_policy(version: PolicyVersion) -> Option<PolicyParams> {
    match version.0 {
        1 => Some(CE_DEFAULT_POLICY),
        _ => None,
    }
}

/// Validate password against a policy (client-side check).
///
/// This is a simple check to give user feedback. Real validation happens in the ZK circuit.
/// Returns list of errors if validation fails, empty if valid.
pub fn validate_password_client_side(password: &str, policy: &PolicyParams) -> Vec<String> {
    let mut errors = vec![];
    let len = password.len();

    // Length check
    if len < policy.min_length as usize {
        errors.push(format!(
            "Password too short: minimum {} characters, got {}",
            policy.min_length, len
        ));
    }

    // Character class counts
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
        } else if !ch.is_ascii_whitespace() {
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
