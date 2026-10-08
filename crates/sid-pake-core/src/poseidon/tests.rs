use super::*;
use pasta_curves::pallas;

#[test]
fn test_poseidon_hash_deterministic() {
    let a = pallas::Base::from(42u64);
    let b = pallas::Base::from(99u64);
    let h1 = poseidon_hash_2(&[a, b]);
    let h2 = poseidon_hash_2(&[a, b]);
    assert_eq!(h1, h2);
}

#[test]
fn test_poseidon_hash_different_inputs() {
    let a = pallas::Base::from(42u64);
    let b = pallas::Base::from(99u64);
    let c = pallas::Base::from(100u64);
    let h1 = poseidon_hash_2(&[a, b]);
    let h2 = poseidon_hash_2(&[a, c]);
    assert_ne!(h1, h2);
}

#[test]
fn test_bytes_to_field_elements() {
    let bytes = b"hello world";
    let fes = bytes_to_field_elements(bytes);
    assert_eq!(fes.len(), 1); // 11 bytes < 31 bytes per FE
}

#[test]
fn test_bytes_to_field_elements_long() {
    let bytes = [0u8; 128]; // MAX_PASSWORD_LEN
    let fes = bytes_to_field_elements(&bytes);
    assert_eq!(fes.len(), 5); // ceil(128/31) = 5
}

#[test]
fn test_legacy_digest_deterministic() {
    let password = b"Str0ngP@ssword!";
    let salt = [42u8; 32];
    assert_eq!(
        legacy_history_digest(password, &salt),
        legacy_history_digest(password, &salt)
    );
}

#[test]
fn test_legacy_digest_different_passwords() {
    let salt = [42u8; 32];
    assert_ne!(
        legacy_history_digest(b"password1", &salt),
        legacy_history_digest(b"password2", &salt)
    );
}

#[test]
fn test_legacy_digest_different_salts() {
    let password = b"same_password";
    assert_ne!(
        legacy_history_digest(password, &[1u8; 32]),
        legacy_history_digest(password, &[2u8; 32])
    );
}

/// The legacy digest is over the zero-padded buffer, the form the stored
/// legacy entries were computed from; migration must reproduce it exactly.
#[test]
fn test_legacy_digest_is_over_the_padded_password() {
    let salt = [42u8; 32];
    let mut padded = [0u8; crate::types::MAX_PASSWORD_LEN];
    padded[..15].copy_from_slice(b"Str0ngP@ssword!");
    assert_eq!(
        legacy_history_digest(b"Str0ngP@ssword!", &salt),
        legacy_history_digest(&padded, &salt)
    );
}
