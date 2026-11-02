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

/// The password hash is the length step followed by the padded elements, as
/// the circuit computes it.
#[test]
fn password_hash_is_the_length_step_then_the_elements() {
    let password = b"Str0ngP@ssword!";
    let mut acc = poseidon_hash_2(&[password_hash_tag(), pallas::Base::from(15u64)]);
    for element in password_elements(password) {
        acc = poseidon_hash_2(&[acc, element]);
    }
    assert_eq!(password_hash(password), acc);
    assert_eq!(password_elements(password).len(), 5);
    assert_eq!(password_elements(&[1u8; 129]).len(), 5);
    assert_eq!(password_elements(&[1u8; 156]).len(), 6);
}

/// The tag is the ASCII domain name packed as one element.
#[test]
fn password_hash_tag_is_its_name() {
    let mut repr = [0u8; 32];
    repr[..22].copy_from_slice(b"SID-PALLAS-PASSWORD-v2");
    assert_eq!(
        password_hash_tag(),
        <pallas::Base as ff::PrimeField>::from_repr(repr).unwrap()
    );
}
