use super::*;
use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};

use crate::types::CE_DEFAULT_POLICY;

/// Gadget A alone over an arbitrary witness: `bytes` are raw field values
/// (not necessarily bytes) and `active` the flags the prover claims, so a
/// test can hand it what a malicious client would.
#[derive(Clone)]
struct PolicyTestCircuit {
    bytes: [u64; MAX_PASSWORD_LEN],
    active: [bool; MAX_PASSWORD_LEN],
    policy: PolicyParams,
}

impl PolicyTestCircuit {
    /// The honest witness for `password`: its bytes, active exactly on them.
    fn honest(password: &str, policy: PolicyParams) -> Self {
        let mut bytes = [0u64; MAX_PASSWORD_LEN];
        let mut active = [false; MAX_PASSWORD_LEN];
        for (i, b) in password.bytes().enumerate() {
            bytes[i] = u64::from(b);
            active[i] = true;
        }
        Self {
            bytes,
            active,
            policy,
        }
    }

    fn satisfied(&self) -> bool {
        // k=11 gives 2048 rows, enough for the password rows, the comparison
        // rows and the lookup tables.
        MockProver::run(11, self, vec![]).unwrap().verify().is_ok()
    }
}

impl Circuit<pallas::Base> for PolicyTestCircuit {
    type Config = PolicyEngineConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        PolicyEngineChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = PolicyEngineChip::construct(config, self.policy);
        let password: [Value<pallas::Base>; MAX_PASSWORD_LEN] =
            core::array::from_fn(|i| Value::known(pallas::Base::from(self.bytes[i])));
        let active: [Value<pallas::Base>; MAX_PASSWORD_LEN] = core::array::from_fn(|i| {
            Value::known(if self.active[i] {
                pallas::Base::one()
            } else {
                pallas::Base::zero()
            })
        });
        chip.synthesize(&mut layouter, &password, &active)?;
        Ok(())
    }
}

#[test]
fn test_valid_password() {
    assert!(PolicyTestCircuit::honest("Str0ngPwd", CE_DEFAULT_POLICY).satisfied());
}

#[test]
fn test_minimum_valid_password() {
    // Exactly meets CE policy: 8 chars, 1 upper, 1 lower, 1 digit
    assert!(PolicyTestCircuit::honest("Aa1bbbbb", CE_DEFAULT_POLICY).satisfied());
}

#[test]
fn test_password_with_symbols() {
    assert!(PolicyTestCircuit::honest("P@ssw0rd!", CE_DEFAULT_POLICY).satisfied());
}

/// A password below the policy has no satisfying witness: the verdict is a
/// constraint, not a value the prover writes.
#[test]
fn a_password_below_the_policy_has_no_witness() {
    for weak in ["abc", "Aa1bbbb", "aa1bbbbb", "AA1BBBBB", "Aabbbbbb"] {
        assert!(
            !PolicyTestCircuit::honest(weak, CE_DEFAULT_POLICY).satisfied(),
            "{weak} must not satisfy the CE policy"
        );
    }
}

/// Every class minimum binds, symbols included when the policy asks for one.
#[test]
fn a_symbol_minimum_binds() {
    let policy = PolicyParams {
        min_symbol: 1,
        ..CE_DEFAULT_POLICY
    };
    assert!(!PolicyTestCircuit::honest("Str0ngPwd", policy).satisfied());
    assert!(PolicyTestCircuit::honest("Str0ng!Pwd", policy).satisfied());
}

/// Marking padding as password does not lengthen it: an active byte must be
/// non-zero.
#[test]
fn padding_marked_active_does_not_count_as_length() {
    let mut circuit = PolicyTestCircuit::honest("Aa1", CE_DEFAULT_POLICY);
    for flag in circuit.active.iter_mut().take(8) {
        *flag = true;
    }
    assert!(!circuit.satisfied());
}

/// A password byte cannot be hidden from the counts by marking it inactive:
/// an inactive position must be padding.
#[test]
fn a_password_byte_marked_inactive_is_refused() {
    let mut circuit = PolicyTestCircuit::honest("Str0ngPwd", CE_DEFAULT_POLICY);
    circuit.active[8] = false;
    assert!(!circuit.satisfied());
}

/// A position holds a byte, never a wider field value: otherwise the packing
/// into the hashed field elements could equal another byte string's.
#[test]
fn a_value_outside_a_byte_is_refused() {
    let mut circuit = PolicyTestCircuit::honest("Str0ngPwd", CE_DEFAULT_POLICY);
    circuit.bytes[20] = 300;
    circuit.active[20] = true;
    assert!(!circuit.satisfied());
}

/// A hole in the password is refused: active positions must be a prefix.
#[test]
fn a_non_prefix_active_pattern_is_refused() {
    let mut circuit = PolicyTestCircuit::honest("Str0ngPwd", CE_DEFAULT_POLICY);
    circuit.bytes[20] = u64::from(b'X');
    circuit.active[20] = true;
    assert!(!circuit.satisfied());
}

/// Each minimum binds exactly at its threshold: one short fails, the
/// threshold itself and one over pass.
#[test]
fn every_threshold_binds_at_n() {
    // (policy with one raised minimum, password with N-1, N, N+1 of that feature)
    let base = PolicyParams {
        min_length: 0,
        min_upper: 0,
        min_lower: 0,
        min_digit: 0,
        min_symbol: 0,
    };
    let cases: [(PolicyParams, [&str; 3]); 5] = [
        (
            PolicyParams {
                min_length: 10,
                ..base
            },
            ["aaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaaa"],
        ),
        (
            PolicyParams {
                min_upper: 3,
                ..base
            },
            ["AAbbbb", "AAAbbb", "AAAAbb"],
        ),
        (
            PolicyParams {
                min_lower: 3,
                ..base
            },
            ["aaBBBB", "aaaBBB", "aaaaBB"],
        ),
        (
            PolicyParams {
                min_digit: 3,
                ..base
            },
            ["11aaaa", "111aaa", "1111aa"],
        ),
        (
            PolicyParams {
                min_symbol: 3,
                ..base
            },
            ["!!aaaa", "!!!aaa", "!!!!aa"],
        ),
    ];
    for (policy, [below, at, above]) in cases {
        assert!(
            !PolicyTestCircuit::honest(below, policy).satisfied(),
            "{below}"
        );
        assert!(PolicyTestCircuit::honest(at, policy).satisfied(), "{at}");
        assert!(
            PolicyTestCircuit::honest(above, policy).satisfied(),
            "{above}"
        );
    }
}

/// Every cell of Gadget A as a prover may set it, laid out under the same
/// selectors, lookups and copies as the chip, so a test can forge any single
/// assignment the honest synthesis would compute.
#[derive(Clone)]
struct RawWitness {
    bytes: [u64; MAX_PASSWORD_LEN],
    active: [u64; MAX_PASSWORD_LEN],
    /// [upper, lower, digit, symbol] per row.
    flags: [[u64; 4]; MAX_PASSWORD_LEN],
    /// [length, upper, lower, digit, symbol] running sums per row.
    acc: [[u64; 5]; MAX_PASSWORD_LEN],
    /// The counts on the comparison rows.
    counts: [u64; 5],
    policy: PolicyParams,
}

impl RawWitness {
    /// The witness the honest chip computes for `password`.
    fn honest(password: &str, policy: PolicyParams) -> Self {
        let mut w = Self {
            bytes: [0; MAX_PASSWORD_LEN],
            active: [0; MAX_PASSWORD_LEN],
            flags: [[0; 4]; MAX_PASSWORD_LEN],
            acc: [[0; 5]; MAX_PASSWORD_LEN],
            counts: [0; 5],
            policy,
        };
        let mut sums = [0u64; 5];
        for i in 0..MAX_PASSWORD_LEN {
            if let Some(&b) = password.as_bytes().get(i) {
                w.bytes[i] = u64::from(b);
                w.active[i] = 1;
                let (u, l, d, s) = PolicyEngineChip::classify(u64::from(b));
                w.flags[i] = [u64::from(u), u64::from(l), u64::from(d), u64::from(s)];
            }
            let increments = [
                w.active[i],
                w.flags[i][0],
                w.flags[i][1],
                w.flags[i][2],
                w.flags[i][3],
            ];
            for k in 0..5 {
                sums[k] += increments[k];
            }
            w.acc[i] = sums;
        }
        w.counts = sums;
        w
    }

    fn satisfied(&self) -> bool {
        MockProver::run(11, self, vec![]).unwrap().verify().is_ok()
    }
}

impl Circuit<pallas::Base> for RawWitness {
    type Config = PolicyEngineConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        PolicyEngineChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let chip = PolicyEngineChip::construct(config.clone(), self.policy);
        chip.load_tables(&mut layouter)?;
        let fe = |v: u64| Value::known(pallas::Base::from(v));
        layouter.assign_region(
            || "raw policy engine",
            |mut region| {
                let flag_cols = [
                    config.is_upper,
                    config.is_lower,
                    config.is_digit,
                    config.is_symbol,
                ];
                let acc_cols = [
                    config.acc_len,
                    config.acc_upper,
                    config.acc_lower,
                    config.acc_digit,
                    config.acc_symbol,
                ];
                let mut last = Vec::new();
                for i in 0..MAX_PASSWORD_LEN {
                    config.q_classify.enable(&mut region, i)?;
                    if i == 0 {
                        config.q_init.enable(&mut region, i)?;
                    } else {
                        config.q_acc.enable(&mut region, i)?;
                    }
                    region.assign_advice(|| "byte", config.byte_col, i, || fe(self.bytes[i]))?;
                    region.assign_advice(
                        || "active",
                        config.active_col,
                        i,
                        || fe(self.active[i]),
                    )?;
                    for (k, col) in flag_cols.into_iter().enumerate() {
                        region.assign_advice(|| "flag", col, i, || fe(self.flags[i][k]))?;
                    }
                    last.clear();
                    for (k, col) in acc_cols.into_iter().enumerate() {
                        last.push(region.assign_advice(|| "acc", col, i, || fe(self.acc[i][k]))?);
                    }
                }
                let minimums = [
                    self.policy.min_length,
                    self.policy.min_upper,
                    self.policy.min_lower,
                    self.policy.min_digit,
                    self.policy.min_symbol,
                ];
                for k in 0..5 {
                    let row = MAX_PASSWORD_LEN + k;
                    config.q_cmp.enable(&mut region, row)?;
                    let count = region.assign_advice(
                        || "count",
                        config.cmp_count,
                        row,
                        || fe(self.counts[k]),
                    )?;
                    region.constrain_equal(last[k].cell(), count.cell())?;
                    region.assign_fixed(
                        || "min",
                        config.policy_min,
                        row,
                        || fe(u64::from(minimums[k])),
                    )?;
                }
                Ok(())
            },
        )
    }
}

/// The raw harness accepts exactly what the chip does for an honest witness.
#[test]
fn the_raw_harness_matches_the_chip() {
    assert!(RawWitness::honest("Str0ngPwd", CE_DEFAULT_POLICY).satisfied());
    assert!(!RawWitness::honest("abc", CE_DEFAULT_POLICY).satisfied());
}

/// A class flag on a byte outside the class is refused (lookup), even with the
/// running sums made consistent with it.
#[test]
fn a_forged_class_flag_is_refused() {
    // "str0ngpwd" has no uppercase letter; claim one for 's'.
    let mut w = RawWitness::honest("str0ngpwd", CE_DEFAULT_POLICY);
    w.flags[0] = [1, 0, 0, 0];
    for row in w.acc.iter_mut() {
        row[1] += 1;
        row[2] -= 1;
    }
    w.counts[1] += 1;
    w.counts[2] -= 1;
    assert!(!w.satisfied());
}

/// A running sum that does not follow its flags is refused.
#[test]
fn a_forged_running_sum_is_refused() {
    let mut w = RawWitness::honest("str0ngpwd", CE_DEFAULT_POLICY);
    w.acc[MAX_PASSWORD_LEN - 1][1] = 1;
    w.counts[1] = 1;
    assert!(!w.satisfied());
}

/// The compared count is the final running sum, not a value of its own.
#[test]
fn a_forged_compared_count_is_refused() {
    let mut w = RawWitness::honest("str0ngpwd", CE_DEFAULT_POLICY);
    w.counts[1] = 1;
    assert!(!w.satisfied());
}

/// A length claimed beyond the active positions is refused.
#[test]
fn a_forged_length_is_refused() {
    let mut w = RawWitness::honest("Str0ng", CE_DEFAULT_POLICY);
    for row in w.acc.iter_mut().skip(6) {
        row[0] = 8;
    }
    w.counts[0] = 8;
    assert!(!w.satisfied());
}

/// A class flag on padding is refused.
#[test]
fn a_flag_on_padding_is_refused() {
    let mut w = RawWitness::honest("Str0ngPwd", CE_DEFAULT_POLICY);
    w.flags[20] = [1, 0, 0, 0];
    for row in w.acc.iter_mut().skip(20) {
        row[1] += 1;
    }
    w.counts[1] += 1;
    assert!(!w.satisfied());
}
