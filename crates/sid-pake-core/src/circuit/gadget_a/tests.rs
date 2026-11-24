use super::*;
use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};

use crate::types::{CE_DEFAULT_POLICY, LONG_POLICY, SHORT_POLICY, STANDARD_POLICY};

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
        Self::of_bytes(password.as_bytes(), policy)
    }

    /// The honest witness for a password of any bytes, as a client that does
    /// not encode text sends them.
    fn of_bytes(password: &[u8], policy: PolicyParams) -> Self {
        let mut bytes = [0u64; MAX_PASSWORD_LEN];
        let mut active = [false; MAX_PASSWORD_LEN];
        for (i, b) in password.iter().enumerate() {
            bytes[i] = u64::from(*b);
            active[i] = true;
        }
        Self {
            bytes,
            active,
            policy,
        }
    }

    fn satisfied(&self) -> bool {
        satisfied(self, self.policy.length_unit)
    }
}

/// A witness of Gadget A alone, laid out under a given configuration.
trait PolicyWitness: Clone {
    fn lay_out(
        &self,
        config: PolicyEngineConfig,
        layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error>;
}

/// `W` under the constraint system of one length unit, which the proving
/// system reads from the type.
#[derive(Clone)]
struct Under<const CODE_POINTS: bool, W>(W);

impl<const CODE_POINTS: bool, W: PolicyWitness> Circuit<pallas::Base> for Under<CODE_POINTS, W> {
    type Config = PolicyEngineConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let unit = if CODE_POINTS {
            LengthUnit::CodePoints
        } else {
            LengthUnit::Bytes
        };
        PolicyEngineChip::configure(meta, unit)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        self.0.lay_out(config, layouter)
    }
}

/// Whether `witness` satisfies Gadget A under the constraint system of
/// `unit`. k=11 gives 2048 rows, enough for the password rows, the
/// comparison rows and the lookup tables.
fn satisfied(witness: &impl PolicyWitness, unit: LengthUnit) -> bool {
    let run = match unit {
        LengthUnit::Bytes => MockProver::run(11, &Under::<false, _>(witness.clone()), vec![]),
        LengthUnit::CodePoints => MockProver::run(11, &Under::<true, _>(witness.clone()), vec![]),
    };
    run.unwrap().verify().is_ok()
}

impl PolicyWitness for PolicyTestCircuit {
    fn lay_out(
        &self,
        config: PolicyEngineConfig,
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
        length_unit: LengthUnit::Bytes,
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

/// A policy counted in code points binds at its minimum for ASCII, with no
/// character class required: one short fails, the minimum and one over pass.
/// Bug prevented: a built-in policy that asks for a class, or accepts a
/// password one character short.
#[test]
fn a_code_point_minimum_binds_at_n() {
    for policy in [SHORT_POLICY, STANDARD_POLICY, LONG_POLICY] {
        let n = policy.min_length as usize;
        assert!(!PolicyTestCircuit::honest(&"a".repeat(n - 1), policy).satisfied());
        assert!(PolicyTestCircuit::honest(&"a".repeat(n), policy).satisfied());
        assert!(PolicyTestCircuit::honest(&"a".repeat(n + 1), policy).satisfied());
    }
}

/// A multibyte character counts once: an input with enough bytes but too few
/// code points is refused, and the same count of code points passes whatever
/// their width.
/// Bug prevented: five four-byte emoji (20 bytes) passing a 12-character
/// minimum.
#[test]
fn a_multibyte_character_counts_once() {
    let emoji = "\u{1F600}";
    assert_eq!(emoji.len(), 4);
    assert!(!PolicyTestCircuit::honest(&emoji.repeat(5), STANDARD_POLICY).satisfied());
    assert!(!PolicyTestCircuit::honest(&emoji.repeat(11), STANDARD_POLICY).satisfied());
    assert!(PolicyTestCircuit::honest(&emoji.repeat(12), STANDARD_POLICY).satisfied());
    // Two-byte and three-byte characters, mixed with ASCII and a space.
    let mixed = "пароль 日本語!";
    assert_eq!(mixed.chars().count(), 11);
    assert!(!PolicyTestCircuit::honest(mixed, STANDARD_POLICY).satisfied());
    assert!(PolicyTestCircuit::honest("пароль 日本語!!", STANDARD_POLICY).satisfied());
}

/// A continuation byte cannot be claimed as the start of a code point: the
/// class flag of a byte outside the table has no witness.
/// Bug prevented: a prover marking every byte of a multibyte character as a
/// character to reach the minimum.
#[test]
fn a_continuation_byte_is_not_a_code_point() {
    let mut w = RawWitness::honest(&"\u{1F600}".repeat(5), STANDARD_POLICY);
    // Claim every active byte starts a code point and carry the sums through.
    let mut sum = 0;
    for i in 0..MAX_PASSWORD_LEN {
        w.flags[i][3] = w.active[i];
        sum += w.active[i];
        w.acc[i][4] = sum;
    }
    w.counts[4] = sum;
    assert_eq!(sum, 20);
    assert!(!w.satisfied());
}

/// The decoder accepts exactly well-formed UTF-8 (RFC 3629 section 4): it
/// agrees with the standard library on every string of one and two bytes and
/// on every three- and four-byte string over the bytes where the grammar's
/// ranges begin and end.
/// Bug prevented: an overlong form, a surrogate or a value above U+10FFFF
/// taken for a character, or a valid one refused.
#[test]
fn the_decoder_accepts_exactly_well_formed_utf8() {
    fn accepted(bytes: &[u8]) -> bool {
        bytes
            .iter()
            .try_fold(0u8, |state, byte| utf8_step(state, *byte))
            == Some(0)
    }
    fn agrees(bytes: &[u8]) {
        // NUL is the padding, not a byte of a password: it is left out.
        assert_eq!(
            accepted(bytes),
            core::str::from_utf8(bytes).is_ok(),
            "{bytes:02x?}"
        );
    }
    for a in 1..=255u8 {
        agrees(&[a]);
        for b in 1..=255u8 {
            agrees(&[a, b]);
        }
    }
    // Each side of every boundary of the grammar's ranges.
    const EDGES: [u8; 22] = [
        0x01, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1, 0xEC,
        0xED, 0xEE, 0xEF, 0xF0, 0xF3, 0xF4, 0xF5,
    ];
    for a in EDGES {
        for b in EDGES {
            for c in EDGES {
                agrees(&[a, b, c]);
                for d in EDGES {
                    agrees(&[a, b, c, d]);
                }
            }
        }
    }
}

/// A policy counted in code points has no proof for bytes that are not
/// well-formed UTF-8, whatever their count: lead bytes with no tail, a tail
/// with no lead, an overlong form, a surrogate, a value above U+10FFFF and a
/// sequence cut by the end of the password are all refused; the same length
/// of well-formed text passes.
/// Bug prevented: twelve lead bytes counted as twelve characters.
#[test]
fn a_code_point_policy_refuses_malformed_utf8() {
    let padded = |malformed: &[u8]| {
        let mut password = b"abcdefghijkl".to_vec();
        password.extend_from_slice(malformed);
        PolicyTestCircuit::of_bytes(&password, STANDARD_POLICY).satisfied()
    };
    assert!(padded("é😀".as_bytes()));
    for malformed in [
        &[0xC2][..],
        &[0x80],
        &[0xC0, 0x80],
        &[0xE0, 0x80, 0x80],
        &[0xED, 0xA0, 0x80],
        &[0xF4, 0x90, 0x80, 0x80],
        &[0xF5, 0x80, 0x80, 0x80],
        &[0xE2, 0x82],
        &[0xFF],
    ] {
        assert!(!padded(malformed), "{malformed:02x?}");
    }
    assert!(!PolicyTestCircuit::of_bytes(&[0xC2; 12], STANDARD_POLICY).satisfied());
    assert!(!PolicyTestCircuit::of_bytes(&[0xC2; 24], STANDARD_POLICY).satisfied());
}

/// A sequence cannot run past the last byte of the buffer: the state after
/// it is a code point boundary, and a prover cannot claim one it did not
/// reach.
/// Bug prevented: a password filling the buffer and ending on a lead byte
/// accepted because no padding row follows to refuse it.
#[test]
fn a_full_buffer_ends_at_a_code_point_boundary() {
    let mut password = "a".repeat(MAX_PASSWORD_LEN - 1).into_bytes();
    password.push(0xC2);
    assert!(!PolicyTestCircuit::of_bytes(&password, STANDARD_POLICY).satisfied());

    let full = "a".repeat(MAX_PASSWORD_LEN - 2) + "é";
    assert!(RawWitness::honest(&full, STANDARD_POLICY).satisfied());
    // The same bytes with the last one replaced by a lead byte, and the
    // final state claimed to be a boundary.
    let mut forged = RawWitness::honest(&full, STANDARD_POLICY);
    forged.bytes[MAX_PASSWORD_LEN - 1] = 0xC2;
    assert!(!forged.satisfied());
    // An honest state chain started inside a sequence.
    let mut forged = RawWitness::honest(&full, STANDARD_POLICY);
    forged.states[0] = 1;
    assert!(!forged.satisfied());
}

/// Version 1 has no encoding check, as released: bytes that are no UTF-8
/// meet it by their count and classes.
/// Bug prevented: the released policy starting to refuse passwords it
/// accepted.
#[test]
fn the_released_policy_takes_any_bytes() {
    let mut password = b"Aa1".to_vec();
    password.extend_from_slice(&[0xFF, 0xC2, 0x80, 0x80, 0xED]);
    assert!(PolicyTestCircuit::of_bytes(&password, CE_DEFAULT_POLICY).satisfied());
}

/// Version 1 counts bytes as it always did: a multibyte input reaches its
/// length minimum by bytes.
/// Bug prevented: the released policy changing its meaning with the new
/// unit.
#[test]
fn the_released_policy_still_counts_bytes() {
    // 3 ASCII class characters and two two-byte letters: 7 bytes, then 9.
    assert!(!PolicyTestCircuit::honest("Aa1пп", CE_DEFAULT_POLICY).satisfied());
    assert!(PolicyTestCircuit::honest("Aa1ппп", CE_DEFAULT_POLICY).satisfied());
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
    /// The UTF-8 decoder's state before each byte and after the last; laid
    /// out only under a policy counted in code points.
    states: [u64; MAX_PASSWORD_LEN + 1],
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
            states: [0; MAX_PASSWORD_LEN + 1],
            policy,
        };
        let mut sums = [0u64; 5];
        let mut state = 0u8;
        for i in 0..MAX_PASSWORD_LEN {
            let byte = password.as_bytes().get(i).copied().unwrap_or(0);
            state = utf8_step(state, byte).unwrap_or(0);
            w.states[i + 1] = u64::from(state);
            if let Some(&b) = password.as_bytes().get(i) {
                w.bytes[i] = u64::from(b);
                w.active[i] = 1;
                let (u, l, d, s) = PolicyEngineChip::classify(policy.length_unit, u64::from(b));
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
        satisfied(self, self.policy.length_unit)
    }
}

impl PolicyWitness for RawWitness {
    fn lay_out(
        &self,
        config: PolicyEngineConfig,
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
                if let Some(utf8) = &config.utf8 {
                    for (i, state) in self.states.iter().enumerate() {
                        region.assign_advice(|| "utf8 state", utf8.state, i, || fe(*state))?;
                        if i < MAX_PASSWORD_LEN {
                            utf8.q_step.enable(&mut region, i)?;
                        }
                    }
                    utf8.q_end.enable(&mut region, MAX_PASSWORD_LEN)?;
                }
                let minimums = self.policy.circuit_minimums();
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
