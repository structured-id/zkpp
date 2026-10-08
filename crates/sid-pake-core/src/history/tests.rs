use super::*;
use rand::rngs::OsRng;

/// Small Argon2id cost for tests; production parameters live in the manifest.
const TEST_KSF: KsfParams = KsfParams {
    memory_kib: 1024,
    passes: 1,
    lanes: 1,
};
const EPOCH_SALT: &[u8] = b"test-epoch-salt-0001";

fn owner_domain(owner: &str) -> pallas::Base {
    domain_element(
        b"SID-HISTORY-INPUT-v1",
        &[b"test-installation", owner.as_bytes()],
    )
}

fn comparison_domain() -> pallas::Base {
    domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-1", b"format-1"])
}

/// The whole client/evaluator exchange for one password, with a fresh blind.
fn tag(k: pallas::Scalar, owner: &str, password: &[u8]) -> pallas::Base {
    let u = history_input(owner_domain(owner), password);
    let r = random_blind(OsRng);
    let z = evaluate(k, blind_request(u, r)).expect("nonidentity");
    finalize_tag(comparison_domain(), u, r, z)
}

/// Synthetic history of one owner: `depth` distinct passwords that all meet
/// the CE policy (upper, lower, digit, length >= 8), oldest first.
fn synthetic_history(owner: &str, depth: usize) -> Vec<String> {
    (0..depth)
        .map(|i| format!("Hist{owner}#{i:02}pwQ7"))
        .collect()
}

fn key() -> pallas::Scalar {
    pallas::Scalar::random(OsRng)
}

/// The tag depends on the password, not on the blind: two operations with
/// fresh blinds give the same tag, or history could never match.
#[test]
fn the_tag_does_not_depend_on_the_blind() {
    let k = key();
    assert_eq!(
        tag(k, "alice", b"Str0ngP@ss"),
        tag(k, "alice", b"Str0ngP@ss")
    );
}

#[test]
fn the_tag_separates_passwords_owners_and_domains() {
    let k = key();
    let base = tag(k, "alice", b"Str0ngP@ss");
    assert_ne!(base, tag(k, "alice", b"Str0ngP@sz"));
    assert_ne!(base, tag(k, "bob", b"Str0ngP@ss"));

    let u = history_input(owner_domain("alice"), b"Str0ngP@ss");
    let r = random_blind(OsRng);
    let z = evaluate(k, blind_request(u, r)).unwrap();
    let other_domain = domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-2", b"format-1"]);
    assert_ne!(base, finalize_tag(other_domain, u, r, z));
}

/// Another history key gives another tag: a key is a comparison domain.
#[test]
fn the_tag_depends_on_the_key() {
    assert_ne!(
        tag(key(), "alice", b"Str0ngP@ss"),
        tag(key(), "alice", b"Str0ngP@ss")
    );
}

/// Blinding `-H` instead of the prescribed `H` (which the circuit refuses)
/// still gives the same tag: the tag uses only x(N), so a wrong sign is not a
/// second tag for the same password even where the sign is not enforced.
#[test]
fn the_other_sign_of_the_point_gives_the_same_tag() {
    let k = key();
    let u = history_input(owner_domain("alice"), b"Str0ngP@ss");
    let (h, _) = canonical_point(u);
    let r = random_blind(OsRng);
    let b_neg = (-pallas::Point::from(h) * blind_scalar(r)).to_affine();
    let z_neg = evaluate(k, b_neg).unwrap();
    assert_eq!(
        finalize_tag(comparison_domain(), u, r, z_neg),
        tag(k, "alice", b"Str0ngP@ss")
    );
}

/// The prescribed sign is the even root, whatever root the square root returns.
#[test]
fn the_canonical_point_has_the_even_root() {
    for i in 0..64u8 {
        let u = history_input(
            owner_domain("alice"),
            &[b'A', b'a', b'1', i, b'x', b'y', b'z', b'w'],
        );
        let (h, _) = canonical_point(u);
        let y = *Option::<pasta_curves::arithmetic::Coordinates<_>>::from(h.coordinates())
            .unwrap()
            .y();
        assert_eq!(y.to_repr()[0] & 1, 0);
    }
}

/// The canonical point is the first one: every smaller offset has no point.
#[test]
fn the_canonical_point_is_the_first_on_the_curve() {
    for owner in ["alice", "bob", "carol", "dave"] {
        let u = history_input(owner_domain(owner), b"Str0ngP@ss");
        let (h, offset) = canonical_point(u);
        let x = *Option::<pasta_curves::arithmetic::Coordinates<_>>::from(h.coordinates())
            .unwrap()
            .x();
        assert_eq!(x, u + pallas::Base::from(offset));
        for i in 0..offset {
            let xi = u + pallas::Base::from(i);
            assert!(bool::from(
                (xi.square() * xi + pallas::Base::from(5u64))
                    .sqrt()
                    .is_none()
            ));
        }
    }
}

#[test]
fn an_honest_evaluation_verifies() {
    let k = key();
    let pk = (pallas::Point::generator() * k).to_affine();
    let b = blind_request(
        history_input(owner_domain("alice"), b"Str0ngP@ss"),
        random_blind(OsRng),
    );
    let (z, proof) = evaluate_with_proof(k, b, b"op-1", OsRng).unwrap();
    assert!(verify_evaluation(pk, b, z, b"op-1", &proof));
}

/// The checker accepts an evaluation only under the owner's key, for the
/// request it was made for, in the operation it was made in.
#[test]
fn a_substituted_evaluation_is_refused() {
    let k = key();
    let pk = (pallas::Point::generator() * k).to_affine();
    let b = blind_request(
        history_input(owner_domain("alice"), b"Str0ngP@ss"),
        random_blind(OsRng),
    );
    let (z, proof) = evaluate_with_proof(k, b, b"op-1", OsRng).unwrap();

    let other_pk = (pallas::Point::generator() * key()).to_affine();
    assert!(
        !verify_evaluation(other_pk, b, z, b"op-1", &proof),
        "another key"
    );
    assert!(
        !verify_evaluation(pk, b, z, b"op-2", &proof),
        "another operation"
    );
    let other_z = (pallas::Point::from(z) + pallas::Point::generator()).to_affine();
    assert!(
        !verify_evaluation(pk, b, other_z, b"op-1", &proof),
        "another result"
    );
    let other_b = blind_request(
        history_input(owner_domain("alice"), b"0therP@ss1"),
        random_blind(OsRng),
    );
    assert!(
        !verify_evaluation(pk, other_b, z, b"op-1", &proof),
        "another request"
    );
}

/// Against a retained history of `depth` passwords, each of them is refused
/// and a new password is accepted, returning the entry to store.
fn history_of_depth_refuses_every_retained_password(depth: usize) {
    let k = key();
    let owner = "alice";
    let old = synthetic_history(owner, depth);
    let retained: Vec<HistoryEntry> = old
        .iter()
        .map(|p| ksf(tag(k, owner, p.as_bytes()), EPOCH_SALT, TEST_KSF).unwrap())
        .collect();

    for p in &old {
        assert_eq!(
            check_candidate(tag(k, owner, p.as_bytes()), EPOCH_SALT, TEST_KSF, &retained),
            Err(HistoryError::Reused),
            "{p} is in the last {depth}"
        );
    }

    let new = b"FreshN3wPassw0rd";
    let entry = check_candidate(tag(k, owner, new), EPOCH_SALT, TEST_KSF, &retained)
        .expect("a new password passes");
    assert_eq!(
        entry,
        ksf(tag(k, owner, new), EPOCH_SALT, TEST_KSF).unwrap()
    );
}

#[test]
fn a_history_of_one_refuses_its_password() {
    history_of_depth_refuses_every_retained_password(1);
}

#[test]
fn a_history_of_ten_refuses_each_of_its_passwords() {
    history_of_depth_refuses_every_retained_password(10);
}

#[test]
fn a_history_of_twenty_four_refuses_each_of_its_passwords() {
    history_of_depth_refuses_every_retained_password(24);
}

/// Another owner's history never matches: owners are separate domains even
/// under one key.
#[test]
fn another_owners_history_does_not_match() {
    let k = key();
    let retained: Vec<HistoryEntry> = synthetic_history("alice", 10)
        .iter()
        .map(|p| ksf(tag(k, "alice", p.as_bytes()), EPOCH_SALT, TEST_KSF).unwrap())
        .collect();
    for p in synthetic_history("alice", 10) {
        assert!(
            check_candidate(tag(k, "bob", p.as_bytes()), EPOCH_SALT, TEST_KSF, &retained).is_ok()
        );
    }
}

/// The retained entry is not the tag: a stored value alone tests nothing
/// without the KSF.
#[test]
fn the_stored_entry_is_the_ksf_of_the_tag() {
    let t = tag(key(), "alice", b"Str0ngP@ss");
    let s = ksf(t, EPOCH_SALT, TEST_KSF).unwrap();
    assert_ne!(s.as_slice(), t.to_repr().as_ref());
    assert_ne!(s, ksf(t, b"another-epoch-salt!!", TEST_KSF).unwrap());
}
