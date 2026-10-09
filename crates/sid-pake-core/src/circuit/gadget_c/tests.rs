use super::*;
use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};
use pasta_curves::arithmetic::CurveAffine;

/// The shared offset bound alone, over one assigned `offset`.
#[derive(Clone)]
struct OffsetBoundCircuit {
    offset: u64,
}

#[derive(Clone, Debug)]
struct OffsetBoundConfig {
    lookup: LookupRangeCheckConfig<pallas::Base, 10>,
    value: Column<Advice>,
    table: TableColumn,
}

impl Circuit<pallas::Base> for OffsetBoundCircuit {
    type Config = OffsetBoundConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let running = meta.advice_column();
        let value = meta.advice_column();
        meta.enable_equality(running);
        meta.enable_equality(value);
        let constants = meta.fixed_column();
        meta.enable_constant(constants);
        let table = meta.lookup_table_column();
        let lookup = LookupRangeCheckConfig::configure(meta, running, table);
        OffsetBoundConfig {
            lookup,
            value,
            table,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "table",
            |mut table| {
                for index in 0..(1u64 << 10) {
                    table.assign_cell(
                        || "idx",
                        config.table,
                        index as usize,
                        || Value::known(pallas::Base::from(index)),
                    )?;
                }
                Ok(())
            },
        )?;
        let cell = layouter.assign_region(
            || "offset",
            |mut region| {
                region.assign_advice(
                    || "offset",
                    config.value,
                    0,
                    || Value::known(pallas::Base::from(self.offset)),
                )
            },
        )?;
        check_htc_offset(&config.lookup, &mut layouter, cell)
    }
}

/// The bound both gadgets share admits exactly the tries of the native
/// mappings: 0 and 2^8 - 1 pass, 2^8 fails. This checks the bound, not an
/// attack: a full Gadget H witness at 2^8 needs 2^8 consecutive non-squares.
#[test]
fn the_offset_bound_admits_exactly_the_tries() {
    let run = |offset| {
        MockProver::run(11, &OffsetBoundCircuit { offset }, vec![])
            .unwrap()
            .verify()
    };
    assert_eq!(run(0), Ok(()));
    assert_eq!(run((1 << HTC_TRY_BITS) - 1), Ok(()));
    assert!(run(1 << HTC_TRY_BITS).is_err());
}

/// The OPAQUE element mapping fixes the sign of `y` to the even root
/// (sgn0 of RFC 9380 §4.1), as the history mapping does, so the OPRF is
/// specified by a rule rather than by one `sqrt` implementation.
#[test]
fn the_opaque_mapping_takes_the_even_root() {
    for i in 0..64 {
        let mut password = [0u8; MAX_PASSWORD_LEN];
        let text = format!("pw-{i}");
        password[..text.len()].copy_from_slice(text.as_bytes());
        let (point, _, _) = hash_to_curve_outside(&password);
        let y = *point.coordinates().unwrap().y();
        assert_eq!(y.to_repr()[0] & 1, 0, "{text}");
    }
}

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
