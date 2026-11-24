use super::*;

#[test]
fn test_get_ce_policy() {
    let policy = get_policy(CE_POLICY_VERSION);
    assert!(policy.is_some());
    assert_eq!(policy.unwrap().min_length, 8);
}

/// Every built-in version answers its own policy: version 1 as released,
/// versions 2 to 4 their code-point minimums with no required class.
/// Bug prevented: a version answering a neighbour's policy, so the issuer's
/// selection enforces another length.
#[test]
fn every_built_in_version_has_its_policy() {
    assert_eq!(get_policy(CE_POLICY_VERSION), Some(CE_DEFAULT_POLICY));
    for (version, min_length) in [
        (STANDARD_POLICY_VERSION, 12),
        (SHORT_POLICY_VERSION, 8),
        (LONG_POLICY_VERSION, 15),
    ] {
        assert_eq!(
            get_policy(version),
            Some(PolicyParams::code_points(min_length))
        );
    }
    for version in BUILT_IN_VERSIONS {
        assert_eq!(get_policy(version).unwrap().check(), Ok(()));
    }
}

/// The length a policy counts is in its unit: bytes, or the bytes that start
/// a code point, so the user's feedback agrees with what a proof establishes.
/// Bug prevented: the form accepting five emoji as twenty characters and the
/// proof then failing.
#[test]
fn the_client_check_counts_in_the_policys_unit() {
    let emoji = "\u{1F600}";
    assert_eq!(length_in(LengthUnit::Bytes, emoji.as_bytes()), 4);
    assert_eq!(length_in(LengthUnit::CodePoints, emoji.as_bytes()), 1);
    assert_eq!(
        length_in(LengthUnit::CodePoints, "пароль 日本語".as_bytes()),
        10
    );
    assert!(!validate_password_client_side(&emoji.repeat(5), &STANDARD_POLICY).is_empty());
    assert!(!validate_password_client_side(&emoji.repeat(11), &STANDARD_POLICY).is_empty());
    assert!(validate_password_client_side(&emoji.repeat(12), &STANDARD_POLICY).is_empty());
    // No class is required: twelve lowercase letters are enough.
    assert!(validate_password_client_side("abcdefghijkl", &STANDARD_POLICY).is_empty());
}

/// A symbol minimum counts ASCII punctuation, as the circuit's table does.
/// Bug prevented: a space or a non-ASCII letter counted as a symbol by the
/// form and refused by the proof.
#[test]
fn the_client_check_counts_ascii_punctuation_as_symbols() {
    let policy = PolicyParams {
        min_symbol: 1,
        ..CE_DEFAULT_POLICY
    };
    let symbols = |password| {
        validate_password_client_side(password, &policy)
            .iter()
            .any(|e| e.contains("symbols"))
    };
    assert!(!symbols("ValidPass123!"));
    assert!(symbols("ValidPass123 "));
    assert!(symbols("ValidPass123п"));
}

#[test]
fn test_invalid_policy_version() {
    let policy = get_policy(PolicyVersion(999));
    assert!(policy.is_none());
}

#[test]
fn test_validate_valid_password() {
    let policy = CE_DEFAULT_POLICY;
    let errors = validate_password_client_side("ValidPass123", &policy);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_password_too_short() {
    let policy = CE_DEFAULT_POLICY;
    let errors = validate_password_client_side("short", &policy);
    assert!(!errors.is_empty());
}

#[test]
fn test_validate_missing_uppercase() {
    let policy = CE_DEFAULT_POLICY;
    let errors = validate_password_client_side("validpass123", &policy);
    assert!(errors.iter().any(|e| e.contains("uppercase")));
}
