use super::*;

#[test]
fn test_get_ce_policy() {
    let policy = get_policy(CE_POLICY_VERSION);
    assert!(policy.is_some());
    assert_eq!(policy.unwrap().min_length, 8);
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
