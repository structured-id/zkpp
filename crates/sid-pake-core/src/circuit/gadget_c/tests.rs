use super::*;
use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};
use pasta_curves::arithmetic::CurveAffine;

/// The binder over `password`, with the hash-to-curve witnesses (`h_p`,
/// `offset`) either honest or chosen by a malicious prover.
#[derive(Clone)]
struct BinderTestCircuit {
    password: [u8; MAX_PASSWORD_LEN],
    witness: Option<(pallas::Affine, pallas::Base)>,
}

impl BinderTestCircuit {
    fn honest(password: [u8; MAX_PASSWORD_LEN]) -> Self {
        Self {
            password,
            witness: None,
        }
    }
}

impl Circuit<pallas::Base> for BinderTestCircuit {
    type Config = OpaqueBinderConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        OpaqueBinderChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = OpaqueBinderChip::construct(config);

        let password_values: [Value<pallas::Base>; MAX_PASSWORD_LEN] =
            core::array::from_fn(|i| Value::known(pallas::Base::from(self.password[i] as u64)));

        let (h_p, offset) = self.witness.unwrap_or_else(|| {
            let (h_p, _u, offset) = hash_to_curve_outside(&self.password);
            (h_p, offset)
        });

        let _output = chip.synthesize(
            &mut layouter,
            &password_values,
            Value::known(pallas::Base::from(7u64)),
            Value::known(h_p),
            Value::known(offset),
        )?;

        Ok(())
    }
}

fn make_password(s: &str) -> [u8; MAX_PASSWORD_LEN] {
    let mut buf = [0u8; MAX_PASSWORD_LEN];
    let bytes = s.as_bytes();
    buf[..bytes.len()].copy_from_slice(bytes);
    buf
}

/// MockProver k for the binder alone (ECC chip rows).
const K: u32 = 13;

#[test]
fn test_hash_to_curve_outside() {
    let password = b"Str0ngP@ssword!";
    let (point, _u, _offset) = hash_to_curve_outside(password);
    let coords = point.coordinates().unwrap();
    let x = *coords.x();
    let y = *coords.y();
    assert_eq!(y * y, x * x * x + pallas::Base::from(5u64));
}

#[test]
fn test_hash_to_curve_deterministic() {
    let (p1, _, _) = hash_to_curve_outside(b"Str0ngP@ssword!");
    let (p2, _, _) = hash_to_curve_outside(b"Str0ngP@ssword!");
    assert_eq!(p1, p2);
}

#[test]
fn test_hash_to_curve_different_passwords() {
    let (p1, _, _) = hash_to_curve_outside(b"Str0ngP@ssword!");
    let (p2, _, _) = hash_to_curve_outside(b"Differ3ntP@ss!");
    assert_ne!(p1, p2);
}

#[test]
fn test_opaque_binder_circuit() {
    let circuit = BinderTestCircuit::honest(make_password("Str0ngP@ssword!"));
    let prover = MockProver::run(K, &circuit, vec![]).unwrap();
    prover.assert_satisfied();
}

/// The committed point must be the hash of the circuit's own password bytes.
/// A prover keeping the strong password's bytes (and thus its policy proof)
/// but committing the point of a weak password would register the weak one
/// behind a valid proof.
#[test]
fn a_point_hashed_from_other_bytes_is_refused() {
    let strong = make_password("Str0ngP@ssword!");
    let (weak_point, _, _) = hash_to_curve_outside(&make_password("abc"));
    let (_, _, strong_offset) = hash_to_curve_outside(&strong);
    let circuit = BinderTestCircuit {
        password: strong,
        witness: Some((weak_point, strong_offset)),
    };
    let failures = format!(
        "{:?}",
        MockProver::run(K, &circuit, vec![]).unwrap().verify()
    );
    assert!(
        failures.contains("htc_bind"),
        "a point not hashed from the password bytes must fail x = u + offset: {failures}"
    );
}

/// The hash-to-curve offset is a small try counter, not a free field element:
/// otherwise `offset = x(H(weak)) - u(strong)` moves the strong password's
/// hash onto the weak password's point.
#[test]
fn an_offset_steering_to_another_point_is_refused() {
    let strong = make_password("Str0ngP@ssword!");
    let (weak_point, _, _) = hash_to_curve_outside(&make_password("abc"));
    let (_, strong_u, _) = hash_to_curve_outside(&strong);
    let weak_x = *weak_point.coordinates().unwrap().x();
    let circuit = BinderTestCircuit {
        password: strong,
        witness: Some((weak_point, weak_x - strong_u)),
    };
    let failures = format!(
        "{:?}",
        MockProver::run(K, &circuit, vec![]).unwrap().verify()
    );
    assert!(
        failures.contains("Lookup") && !failures.contains("htc_bind"),
        "an offset outside the try range must fail only its range check: {failures}"
    );
}
