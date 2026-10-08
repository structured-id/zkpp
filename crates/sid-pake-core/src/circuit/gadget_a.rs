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

use ff::PrimeField;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector, TableColumn},
    poly::Rotation,
};
use pasta_curves::pallas;

use crate::types::{MAX_PASSWORD_LEN, PolicyParams};

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

    pub fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> PolicyEngineConfig {
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
        }
    }

    /// Load the ASCII class tables and the byte range table.
    pub fn load_tables(&self, layouter: &mut impl Layouter<pallas::Base>) -> Result<(), Error> {
        let config = &self.config;
        let symbols: Vec<u64> = (33u64..=47)
            .chain(58..=64)
            .chain(91..=96)
            .chain(123..=126)
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
        Ok(())
    }

    /// Classify a single byte into character classes.
    fn classify(b: u64) -> (bool, bool, bool, bool) {
        let is_upper = (65..=90).contains(&b);
        let is_lower = (97..=122).contains(&b);
        let is_digit = (48..=57).contains(&b);
        let is_symbol = (33..=47).contains(&b)
            || (58..=64).contains(&b)
            || (91..=96).contains(&b)
            || (123..=126).contains(&b);
        (is_upper, is_lower, is_digit, is_symbol)
    }

    /// Synthesize the policy engine over the padded password and its active
    /// flags (1 for a password byte, 0 for padding). Returns the assigned
    /// password byte cells for cross-gadget binding.
    pub fn synthesize(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        password: &[Value<pallas::Base>; MAX_PASSWORD_LEN],
        active: &[Value<pallas::Base>; MAX_PASSWORD_LEN],
    ) -> Result<Vec<AssignedCell<pallas::Base, pallas::Base>>, Error> {
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
                        let (u, l, d, s) = Self::classify(b_u64);
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

                // Comparison rows: each final count against its fixed minimum,
                // in the order [length, upper, lower, digit, symbol].
                let minimums = [
                    policy.min_length,
                    policy.min_upper,
                    policy.min_lower,
                    policy.min_digit,
                    policy.min_symbol,
                ];
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

                Ok(byte_cells)
            },
        )
    }
}

#[cfg(test)]
mod tests;
