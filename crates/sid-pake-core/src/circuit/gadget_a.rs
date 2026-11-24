// SPDX-License-Identifier: AGPL-3.0-only
//! Gadget A: Policy Engine
//!
//! Counts the password's length and character classes and constrains each
//! count to meet the policy minimum. The minimums live in a fixed column, so
//! they are part of the proving and verifying keys: a proof verifies only
//! under the key built for the policy it was made against, and a password
//! below that policy has no satisfying witness at all.
//!
//! Layout (rows 0..MAX_PASSWORD_LEN):
//!   byte[i], active[i], is_upper[i], is_lower[i], is_digit[i], is_symbol[i]
//!   acc_len[i], acc_upper[i], acc_lower[i], acc_digit[i], acc_symbol[i]
//! Comparison rows (MAX_PASSWORD_LEN..+5): count copied from the last
//! accumulator row, policy_min fixed.
//!
//! Gates:
//!   q_classify: active and flags boolean, inactive → flags = 0 and byte = 0
//!   q_init (row 0): acc[0] = flag[0]
//!   q_acc (rows 1..): acc[i] = acc[i-1] + flag[i]; active[i] ≤ active[i-1] (prefix)
//!
//! Lookups:
//!   active*is_upper*byte + (1-active*is_upper)*65 ∈ T_UPPER (likewise lower, digit, symbol)
//!   byte ∈ T_RANGE, active*(byte-1) ∈ T_RANGE (an active byte is non-zero)
//!   q_cmp*(count - policy_min) ∈ T_RANGE (each count meets its minimum)
//!
//! A policy counted in code points has, in addition, the UTF-8 decoder's
//! state before each byte: (state[i], byte[i], state[i+1]) ∈ T_UTF8, the
//! state zero on row 0 and on the row after the last byte. The password is
//! then well-formed UTF-8 (RFC 3629 section 4), and its lead bytes, which
//! the fourth class counts, are its code points. A policy counted in bytes
//! has none of this and keeps the released constraint system.

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector, TableColumn},
    poly::Rotation,
};
use pasta_curves::pallas;

use crate::types::{LengthUnit, MAX_PASSWORD_LEN, PolicyParams, starts_code_point};

/// Values a comparison difference may take: counts and minimums are at most
/// `MAX_PASSWORD_LEN`, so a met minimum leaves a difference in this range and
/// an unmet one wraps to a field element far outside it.
const RANGE: u64 = 256;

/// Configuration for the Policy Engine gadget.
#[derive(Debug, Clone)]
pub struct PolicyEngineConfig {
    pub byte_col: Column<Advice>,
    pub active_col: Column<Advice>,
    pub is_upper: Column<Advice>,
    pub is_lower: Column<Advice>,
    pub is_digit: Column<Advice>,
    pub is_symbol: Column<Advice>,
    pub acc_len: Column<Advice>,
    pub acc_upper: Column<Advice>,
    pub acc_lower: Column<Advice>,
    pub acc_digit: Column<Advice>,
    pub acc_symbol: Column<Advice>,
    pub cmp_count: Column<Advice>,
    pub policy_min: Column<Fixed>,
    pub table_upper: TableColumn,
    pub table_lower: TableColumn,
    pub table_digit: TableColumn,
    pub table_symbol: TableColumn,
    pub table_range: TableColumn,
    pub q_classify: Selector,
    pub q_init: Selector,
    pub q_acc: Selector,
    pub q_cmp: Selector,
    /// The UTF-8 well-formedness check, in the constraint system of a policy
    /// counted in code points only.
    pub utf8: Option<Utf8Config>,
}

/// The columns of the UTF-8 well-formedness check: the decoder's state
/// before each byte and its transition table.
#[derive(Debug, Clone)]
pub struct Utf8Config {
    pub state: Column<Advice>,
    pub table_state: TableColumn,
    pub table_byte: TableColumn,
    pub table_next: TableColumn,
    /// On every byte row: the transition lookup.
    pub q_step: Selector,
    /// On the row after the last byte: the final state.
    pub q_end: Selector,
}

/// The decoder's state between two bytes of a UTF-8 string (RFC 3629
/// section 4, `UTF8-octets`): at a code point boundary, or inside a sequence
/// with the bytes its tail may still take.
mod utf8_state {
    /// At a code point boundary: the only state a string starts and ends in.
    pub(super) const BOUNDARY: u8 = 0;
    /// One tail byte `80-BF` left.
    pub(super) const TAIL_1: u8 = 1;
    /// Two tail bytes `80-BF` left.
    pub(super) const TAIL_2: u8 = 2;
    /// After `E0`: `A0-BF`, then one tail byte (no overlong form).
    pub(super) const AFTER_E0: u8 = 3;
    /// After `ED`: `80-9F`, then one tail byte (no surrogate).
    pub(super) const AFTER_ED: u8 = 4;
    /// Three tail bytes `80-BF` left.
    pub(super) const TAIL_3: u8 = 5;
    /// After `F0`: `90-BF`, then two tail bytes (no overlong form).
    pub(super) const AFTER_F0: u8 = 6;
    /// After `F4`: `80-8F`, then two tail bytes (nothing above U+10FFFF).
    pub(super) const AFTER_F4: u8 = 7;
    /// Every state.
    pub(super) const ALL: core::ops::RangeInclusive<u8> = BOUNDARY..=AFTER_F4;
}

/// The state after `byte` in `state`, by the `UTF8-octets` grammar of
/// RFC 3629 section 4; none where the grammar allows no such byte. NUL is
/// the padding after the password: it keeps the boundary and is no byte of a
/// password.
pub(crate) const fn utf8_step(state: u8, byte: u8) -> Option<u8> {
    use utf8_state::*;
    Some(match (state, byte) {
        (BOUNDARY, 0x00..=0x7F) => BOUNDARY,
        (BOUNDARY, 0xC2..=0xDF) => TAIL_1,
        (BOUNDARY, 0xE0) => AFTER_E0,
        (BOUNDARY, 0xE1..=0xEC | 0xEE..=0xEF) => TAIL_2,
        (BOUNDARY, 0xED) => AFTER_ED,
        (BOUNDARY, 0xF0) => AFTER_F0,
        (BOUNDARY, 0xF1..=0xF3) => TAIL_3,
        (BOUNDARY, 0xF4) => AFTER_F4,
        (TAIL_1, 0x80..=0xBF) => BOUNDARY,
        (TAIL_2, 0x80..=0xBF) | (AFTER_E0, 0xA0..=0xBF) | (AFTER_ED, 0x80..=0x9F) => TAIL_1,
        (TAIL_3, 0x80..=0xBF) | (AFTER_F0, 0x90..=0xBF) | (AFTER_F4, 0x80..=0x8F) => TAIL_2,
        _ => return None,
    })
}

/// Chip implementing the Policy Engine gadget.
pub struct PolicyEngineChip {
    config: PolicyEngineConfig,
    policy: PolicyParams,
}

impl PolicyEngineChip {
    /// `policy` supplies the fixed minimums; it binds only at key generation,
    /// a prover cannot change it for a key it did not build.
    pub fn construct(config: PolicyEngineConfig, policy: PolicyParams) -> Self {
        Self { config, policy }
    }

    /// The policy engine's constraints for policies counted in `unit`. The
    /// unit is part of the constraint system, not only of the tables: a
    /// policy counted in code points also proves the password well-formed
    /// UTF-8, and one counted in bytes keeps the released constraint system.
    pub fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        unit: LengthUnit,
    ) -> PolicyEngineConfig {
        let byte_col = meta.advice_column();
        let active_col = meta.advice_column();
        let is_upper = meta.advice_column();
        let is_lower = meta.advice_column();
        let is_digit = meta.advice_column();
        let is_symbol = meta.advice_column();
        let acc_len = meta.advice_column();
        let acc_upper = meta.advice_column();
        let acc_lower = meta.advice_column();
        let acc_digit = meta.advice_column();
        let acc_symbol = meta.advice_column();
        let cmp_count = meta.advice_column();
        let policy_min = meta.fixed_column();

        let table_upper = meta.lookup_table_column();
        let table_lower = meta.lookup_table_column();
        let table_digit = meta.lookup_table_column();
        let table_symbol = meta.lookup_table_column();
        let table_range = meta.lookup_table_column();

        let q_classify = meta.selector();
        let q_init = meta.selector();
        let q_acc = meta.selector();
        let q_cmp = meta.complex_selector();

        meta.enable_equality(byte_col);
        meta.enable_equality(acc_len);
        meta.enable_equality(acc_upper);
        meta.enable_equality(acc_lower);
        meta.enable_equality(acc_digit);
        meta.enable_equality(acc_symbol);
        meta.enable_equality(cmp_count);

        // Gate: active ∈ {0, 1}
        meta.create_gate("active_boolean", |meta| {
            let q = meta.query_selector(q_classify);
            let a = meta.query_advice(active_col, Rotation::cur());
            vec![q * a.clone() * (Expression::Constant(pallas::Base::one()) - a)]
        });

        // Gate: each is_* ∈ {0, 1}
        meta.create_gate("flags_boolean", |meta| {
            let q = meta.query_selector(q_classify);
            let one = Expression::Constant(pallas::Base::one());
            let iu = meta.query_advice(is_upper, Rotation::cur());
            let il = meta.query_advice(is_lower, Rotation::cur());
            let id = meta.query_advice(is_digit, Rotation::cur());
            let is = meta.query_advice(is_symbol, Rotation::cur());
            vec![
                q.clone() * iu.clone() * (one.clone() - iu),
                q.clone() * il.clone() * (one.clone() - il),
                q.clone() * id.clone() * (one.clone() - id),
                q * is.clone() * (one - is),
            ]
        });

        // Gate: inactive → all flags are 0 and the byte is padding (0).
        // With the non-zero lookup below, active[i] = (byte[i] ≠ 0) exactly,
        // and with the prefix gate the password is the non-zero prefix, so
        // the length counted by acc_len is the password's own length.
        meta.create_gate("inactive_is_padding", |meta| {
            let q = meta.query_selector(q_classify);
            let a = meta.query_advice(active_col, Rotation::cur());
            let not_active = Expression::Constant(pallas::Base::one()) - a;
            let byte = meta.query_advice(byte_col, Rotation::cur());
            let iu = meta.query_advice(is_upper, Rotation::cur());
            let il = meta.query_advice(is_lower, Rotation::cur());
            let id = meta.query_advice(is_digit, Rotation::cur());
            let is = meta.query_advice(is_symbol, Rotation::cur());
            vec![
                q.clone() * not_active.clone() * byte,
                q.clone() * not_active.clone() * iu,
                q.clone() * not_active.clone() * il,
                q.clone() * not_active.clone() * id,
                q * not_active * is,
            ]
        });

        // Gate: base case (row 0) — acc[0] = flag[0]
        meta.create_gate("acc_init", |meta| {
            let q = meta.query_selector(q_init);
            let an = meta.query_advice(acc_len, Rotation::cur());
            let au = meta.query_advice(acc_upper, Rotation::cur());
            let al = meta.query_advice(acc_lower, Rotation::cur());
            let ad = meta.query_advice(acc_digit, Rotation::cur());
            let as_ = meta.query_advice(acc_symbol, Rotation::cur());
            let a = meta.query_advice(active_col, Rotation::cur());
            let iu = meta.query_advice(is_upper, Rotation::cur());
            let il = meta.query_advice(is_lower, Rotation::cur());
            let id = meta.query_advice(is_digit, Rotation::cur());
            let is = meta.query_advice(is_symbol, Rotation::cur());
            vec![
                q.clone() * (an - a),
                q.clone() * (au - iu),
                q.clone() * (al - il),
                q.clone() * (ad - id),
                q * (as_ - is),
            ]
        });

        // Gate: the active positions are a prefix, active[cur] ≤ active[prev]:
        // once padding starts, nothing after it is password.
        meta.create_gate("active_prefix", |meta| {
            let q = meta.query_selector(q_acc);
            let prev = meta.query_advice(active_col, Rotation::prev());
            let cur = meta.query_advice(active_col, Rotation::cur());
            vec![q * (Expression::Constant(pallas::Base::one()) - prev) * cur]
        });

        // Gate: accumulation (rows 1..) — acc[cur] = acc[prev] + flag[cur]
        meta.create_gate("acc_step", |meta| {
            let q = meta.query_selector(q_acc);
            let step = |meta: &mut halo2_proofs::plonk::VirtualCells<'_, pallas::Base>,
                        acc: Column<Advice>,
                        flag: Column<Advice>| {
                meta.query_advice(acc, Rotation::cur())
                    - meta.query_advice(acc, Rotation::prev())
                    - meta.query_advice(flag, Rotation::cur())
            };
            vec![
                q.clone() * step(meta, acc_len, active_col),
                q.clone() * step(meta, acc_upper, is_upper),
                q.clone() * step(meta, acc_lower, is_lower),
                q.clone() * step(meta, acc_digit, is_digit),
                q * step(meta, acc_symbol, is_symbol),
            ]
        });

        // Lookups: unconditional, with conditional default value.
        // When active*flag=1: looks up byte (must be in table).
        // When active*flag=0: looks up default (always in table).
        // This way, unassigned rows (all zeros) also produce default → valid.
        for (flag, table, default) in [
            (is_upper, table_upper, 65u64),
            (is_lower, table_lower, 97),
            (is_digit, table_digit, 48),
            (is_symbol, table_symbol, 33),
        ] {
            meta.lookup(|meta| {
                let byte = meta.query_advice(byte_col, Rotation::cur());
                let active = meta.query_advice(active_col, Rotation::cur());
                let flag = meta.query_advice(flag, Rotation::cur());
                let cond = active * flag;
                let default = Expression::Constant(pallas::Base::from(default));
                let val = cond.clone() * byte
                    + (Expression::Constant(pallas::Base::one()) - cond) * default;
                vec![(val, table)]
            });
        }

        // Every byte is a byte: the packing into field elements for the
        // Poseidon input (Gadget C) is then injective, so the password counted
        // here is the password hashed there.
        meta.lookup(|meta| {
            let byte = meta.query_advice(byte_col, Rotation::cur());
            vec![(byte, table_range)]
        });

        // An active byte is not padding: byte - 1 ∈ [0, 255] when active.
        meta.lookup(|meta| {
            let byte = meta.query_advice(byte_col, Rotation::cur());
            let active = meta.query_advice(active_col, Rotation::cur());
            let one = Expression::Constant(pallas::Base::one());
            vec![(active * (byte - one), table_range)]
        });

        // Each count meets its fixed minimum: count - min ∈ [0, 255].
        meta.lookup(|meta| {
            let q = meta.query_selector(q_cmp);
            let count = meta.query_advice(cmp_count, Rotation::cur());
            let min = meta.query_fixed(policy_min);
            vec![(q * (count - min), table_range)]
        });

        // Allocated after everything a byte policy has, so that policy's
        // constraint system is the released one.
        let utf8 = (unit == LengthUnit::CodePoints).then(|| {
            let state = meta.advice_column();
            let table_state = meta.lookup_table_column();
            let table_byte = meta.lookup_table_column();
            let table_next = meta.lookup_table_column();
            let q_end = meta.selector();

            // The string starts at a code point boundary and, after the last
            // byte row, is at one again: no sequence is cut short.
            meta.create_gate("utf8_boundaries", |meta| {
                let state = meta.query_advice(state, Rotation::cur());
                vec![
                    meta.query_selector(q_init) * state.clone(),
                    meta.query_selector(q_end) * state,
                ]
            });

            // Each byte is one the grammar allows in the state before it and
            // leads to the state before the next. Padding keeps the boundary
            // and is allowed nowhere else, so a password that ends inside a
            // sequence has no witness. Off the byte rows the selector makes
            // the all-zero row, which is the padding entry.
            let q_step = meta.complex_selector();
            meta.lookup(|meta| {
                let q = meta.query_selector(q_step);
                vec![
                    (
                        q.clone() * meta.query_advice(state, Rotation::cur()),
                        table_state,
                    ),
                    (
                        q.clone() * meta.query_advice(byte_col, Rotation::cur()),
                        table_byte,
                    ),
                    (q * meta.query_advice(state, Rotation::next()), table_next),
                ]
            });

            Utf8Config {
                state,
                table_state,
                table_byte,
                table_next,
                q_step,
                q_end,
            }
        });

        PolicyEngineConfig {
            byte_col,
            active_col,
            is_upper,
            is_lower,
            is_digit,
            is_symbol,
            acc_len,
            acc_upper,
            acc_lower,
            acc_digit,
            acc_symbol,
            cmp_count,
            policy_min,
            table_upper,
            table_lower,
            table_digit,
            table_symbol,
            table_range,
            q_classify,
            q_init,
            q_acc,
            q_cmp,
            utf8,
        }
    }

    /// Load the ASCII class tables and the byte range table.
    pub fn load_tables(&self, layouter: &mut impl Layouter<pallas::Base>) -> Result<(), Error> {
        let config = &self.config;
        // The fourth class is the policy's: ASCII symbols, or the bytes that
        // start a code point when the policy counts those. The table is part
        // of the keys, like the minimums.
        let symbols: Vec<u64> = (0u8..=255)
            .filter(|b| Self::fourth_class(self.policy.length_unit, *b))
            .map(u64::from)
            .collect();
        let tables: [(&str, TableColumn, Vec<u64>); 5] = [
            ("T_UPPER", config.table_upper, (65u64..=90).collect()),
            ("T_LOWER", config.table_lower, (97u64..=122).collect()),
            ("T_DIGIT", config.table_digit, (48u64..=57).collect()),
            ("T_SYMBOL", config.table_symbol, symbols),
            ("T_RANGE", config.table_range, (0..RANGE).collect()),
        ];
        for (name, column, values) in tables {
            layouter.assign_table(
                || name,
                |mut table| {
                    for (i, val) in values.iter().enumerate() {
                        table.assign_cell(
                            || format!("{name}_{val}"),
                            column,
                            i,
                            || Value::known(pallas::Base::from(*val)),
                        )?;
                    }
                    Ok(())
                },
            )?;
        }
        if let Some(utf8) = &config.utf8 {
            // The padding entry first: it is the all-zero row, which the
            // table's unused rows repeat.
            let transitions = utf8_state::ALL.flat_map(|state| {
                (0u8..=255).filter_map(move |byte| Some((state, byte, utf8_step(state, byte)?)))
            });
            layouter.assign_table(
                || "T_UTF8",
                |mut table| {
                    for (i, (state, byte, next)) in transitions.clone().enumerate() {
                        for (column, value) in [
                            (utf8.table_state, state),
                            (utf8.table_byte, byte),
                            (utf8.table_next, next),
                        ] {
                            table.assign_cell(
                                || format!("T_UTF8_{state}_{byte}"),
                                column,
                                i,
                                || Value::known(pallas::Base::from(u64::from(value))),
                            )?;
                        }
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    /// Whether `byte` counts in the fourth class under `unit`.
    fn fourth_class(unit: LengthUnit, byte: u8) -> bool {
        match unit {
            LengthUnit::Bytes => {
                matches!(byte, 33..=47 | 58..=64 | 91..=96 | 123..=126)
            }
            LengthUnit::CodePoints => starts_code_point(byte),
        }
    }

    /// Classify a single byte into the four counted classes under `unit`.
    fn classify(unit: LengthUnit, b: u64) -> (bool, bool, bool, bool) {
        let is_upper = (65..=90).contains(&b);
        let is_lower = (97..=122).contains(&b);
        let is_digit = (48..=57).contains(&b);
        let fourth = u8::try_from(b).is_ok_and(|byte| Self::fourth_class(unit, byte));
        (is_upper, is_lower, is_digit, fourth)
    }

    /// Synthesize the policy engine over the padded password and its active
    /// flags (1 for a password byte, 0 for padding). Returns the assigned
    /// password byte cells and the length count, the number of active bytes,
    /// for cross-gadget binding.
    pub fn synthesize(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        password: &[Value<pallas::Base>; MAX_PASSWORD_LEN],
        active: &[Value<pallas::Base>; MAX_PASSWORD_LEN],
    ) -> Result<PolicyEngineOutput, Error> {
        let config = &self.config;
        let policy = &self.policy;

        self.load_tables(layouter)?;

        layouter.assign_region(
            || "policy_engine",
            |mut region| {
                let zero = Value::known(pallas::Base::zero());
                let mut last = [zero; 5];
                let mut last_cells = Vec::new();
                let mut byte_cells = Vec::with_capacity(MAX_PASSWORD_LEN);

                for i in 0..MAX_PASSWORD_LEN {
                    config.q_classify.enable(&mut region, i)?;
                    if i == 0 {
                        config.q_init.enable(&mut region, i)?;
                    } else {
                        config.q_acc.enable(&mut region, i)?;
                    }

                    byte_cells.push(region.assign_advice(
                        || format!("byte_{i}"),
                        config.byte_col,
                        i,
                        || password[i],
                    )?);
                    region.assign_advice(
                        || format!("active_{i}"),
                        config.active_col,
                        i,
                        || active[i],
                    )?;

                    // Classify byte when active
                    let flags = password[i].zip(active[i]).map(|(b, a)| {
                        let to_fe = |b: bool| {
                            if b {
                                pallas::Base::one()
                            } else {
                                pallas::Base::zero()
                            }
                        };
                        if a == pallas::Base::zero() {
                            return [pallas::Base::zero(); 4];
                        }
                        let b_u64 = u64::from_le_bytes(b.to_repr()[..8].try_into().unwrap());
                        let (u, l, d, s) = Self::classify(policy.length_unit, b_u64);
                        [to_fe(u), to_fe(l), to_fe(d), to_fe(s)]
                    });
                    let flag_cols = [
                        config.is_upper,
                        config.is_lower,
                        config.is_digit,
                        config.is_symbol,
                    ];
                    for (k, col) in flag_cols.into_iter().enumerate() {
                        region.assign_advice(
                            || format!("flag_{k}_{i}"),
                            col,
                            i,
                            || flags.map(|f| f[k]),
                        )?;
                    }

                    // Accumulators: length (active), then the four classes.
                    let increments = [
                        active[i],
                        flags.map(|f| f[0]),
                        flags.map(|f| f[1]),
                        flags.map(|f| f[2]),
                        flags.map(|f| f[3]),
                    ];
                    let acc_cols = [
                        config.acc_len,
                        config.acc_upper,
                        config.acc_lower,
                        config.acc_digit,
                        config.acc_symbol,
                    ];
                    last_cells.clear();
                    for k in 0..5 {
                        let value = if i == 0 {
                            increments[k]
                        } else {
                            last[k].zip(increments[k]).map(|(acc, f)| acc + f)
                        };
                        last[k] = value;
                        last_cells.push(region.assign_advice(
                            || format!("acc_{k}_{i}"),
                            acc_cols[k],
                            i,
                            || value,
                        )?);
                    }
                }

                // The decoder's state before each byte, and after the last.
                if let Some(utf8) = &config.utf8 {
                    let mut state = Value::known(utf8_state::BOUNDARY);
                    for (i, byte) in password.iter().enumerate() {
                        utf8.q_step.enable(&mut region, i)?;
                        region.assign_advice(
                            || format!("utf8_state_{i}"),
                            utf8.state,
                            i,
                            || state.map(|s| pallas::Base::from(u64::from(s))),
                        )?;
                        // A byte the grammar refuses has no next state: any
                        // value fails the lookup, and the proof is refused.
                        state = state.zip(*byte).map(|(s, b)| {
                            u8::try_from(u64::from_le_bytes(b.to_repr()[..8].try_into().unwrap()))
                                .ok()
                                .and_then(|b| utf8_step(s, b))
                                .unwrap_or(utf8_state::BOUNDARY)
                        });
                    }
                    utf8.q_end.enable(&mut region, MAX_PASSWORD_LEN)?;
                    region.assign_advice(
                        || "utf8_state_end",
                        utf8.state,
                        MAX_PASSWORD_LEN,
                        || state.map(|s| pallas::Base::from(u64::from(s))),
                    )?;
                }

                // Comparison rows: each final count against its fixed minimum,
                // in the order [length, upper, lower, digit, fourth class].
                let minimums = policy.circuit_minimums();
                let length = last_cells[0].clone();
                for (k, (count, min)) in last_cells.iter().zip(minimums).enumerate() {
                    let row = MAX_PASSWORD_LEN + k;
                    config.q_cmp.enable(&mut region, row)?;
                    count.copy_advice(
                        || format!("count_{k}"),
                        &mut region,
                        config.cmp_count,
                        row,
                    )?;
                    region.assign_fixed(
                        || format!("policy_min_{k}"),
                        config.policy_min,
                        row,
                        || Value::known(pallas::Base::from(u64::from(min))),
                    )?;
                }

                Ok(PolicyEngineOutput { byte_cells, length })
            },
        )
    }
}

/// What the policy engine hands the other gadgets.
pub struct PolicyEngineOutput {
    /// The password byte cells, padding included.
    pub byte_cells: Vec<AssignedCell<pallas::Base, pallas::Base>>,
    /// The length count: the number of active bytes, which the prefix and
    /// non-zero constraints make the password's own length.
    pub length: AssignedCell<pallas::Base, pallas::Base>,
}

#[cfg(test)]
mod tests;
