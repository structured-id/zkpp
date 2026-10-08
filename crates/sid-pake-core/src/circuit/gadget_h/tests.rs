use super::*;
use crate::circuit::gadget_c::{OpaqueBinderChip, OpaqueBinderConfig, hash_to_curve_outside};
use crate::history::{domain_element, history_input, random_blind};
use crate::types::MAX_PASSWORD_LEN;
use group::Curve;
use halo2_proofs::{
    circuit::SimpleFloorPlanner,
    dev::MockProver,
    plonk::{Circuit, Instance},
};
use rand::rngs::OsRng;

/// The binder (for the password hash `u_C` it binds) and Gadget H over it,
/// with instance `[d, c_0.., B.x, B.y, (Z_j.x, Z_j.y, t_j)..]`.
#[derive(Clone)]
struct HistoryTestCircuit {
    password: [u8; MAX_PASSWORD_LEN],
    witness: HistoryTagWitness,
}

#[derive(Clone, Debug)]
struct TestConfig {
    binder: OpaqueBinderConfig,
    history: HistoryTagConfig,
    instance: Column<Instance>,
}

impl Circuit<pallas::Base> for HistoryTestCircuit {
    type Config = TestConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> TestConfig {
        let binder = OpaqueBinderChip::configure(meta);
        let history = HistoryTagChip::configure(
            meta,
            binder.ecc_config.clone(),
            binder.poseidon_config.clone(),
        );
        let instance = meta.instance_column();
        meta.enable_equality(instance);
        TestConfig {
            binder,
            history,
            instance,
        }
    }

    fn synthesize(
        &self,
        config: TestConfig,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let (h_p, _, offset) = hash_to_curve_outside(&self.password);
        let password: [Value<pallas::Base>; MAX_PASSWORD_LEN] =
            core::array::from_fn(|i| Value::known(pallas::Base::from(u64::from(self.password[i]))));
        let binder = OpaqueBinderChip::construct(config.binder.clone()).synthesize(
            &mut layouter,
            &password,
            Value::known(pallas::Base::from(7u64)),
            Value::known(h_p),
            Value::known(offset),
        )?;
        let domains = self.witness.n.len();
        let (d, cs) = layouter.assign_region(
            || "domains",
            |mut region| {
                let col = config.binder.input_col;
                let d = region.assign_advice_from_instance(|| "d", config.instance, 0, col, 0)?;
                let cs = (0..domains)
                    .map(|j| {
                        region.assign_advice_from_instance(
                            || "c",
                            config.instance,
                            1 + j,
                            col,
                            1 + j,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((d, cs))
            },
        )?;
        let out = HistoryTagChip::construct(config.history).synthesize(
            &mut layouter,
            d,
            binder.u_hash,
            &cs,
            Value::known(&self.witness),
        )?;
        let exposed = [&out.b_x, &out.b_y]
            .into_iter()
            .chain(out.tags.iter().flat_map(|tag| [&tag.z_x, &tag.z_y, &tag.t]));
        for (i, cell) in exposed.enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 1 + domains + i)?;
        }
        Ok(())
    }
}

const K: u32 = 12;

fn padded(p: &[u8]) -> [u8; MAX_PASSWORD_LEN] {
    let mut buf = [0u8; MAX_PASSWORD_LEN];
    buf[..p.len()].copy_from_slice(p);
    buf
}

fn owner() -> pallas::Base {
    domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"])
}

fn comparison_domains(n: usize) -> Vec<pallas::Base> {
    (0..n)
        .map(|j| domain_element(b"SID-HISTORY-TAG-v1", &[format!("epoch-{j}").as_bytes()]))
        .collect()
}

fn instances(u: pallas::Base, w: &HistoryTagWitness) -> Vec<pallas::Base> {
    let cs = comparison_domains(w.n.len());
    let (b, tags) = tag_instances(&cs, u, w);
    let b = b.coordinates().unwrap();
    let mut inst = vec![owner()];
    inst.extend(&cs);
    inst.extend([*b.x(), *b.y()]);
    for (z, t) in tags {
        let z = z.coordinates().unwrap();
        inst.extend([*z.x(), *z.y(), t]);
    }
    inst
}

fn verify(password: &[u8], w: HistoryTagWitness, inst: Vec<pallas::Base>) -> String {
    let circuit = HistoryTestCircuit {
        password: padded(password),
        witness: w,
    };
    match MockProver::run(K, &circuit, vec![inst]) {
        Ok(prover) => format!("{:?}", prover.verify()),
        Err(e) => format!("synthesis: {e:?}"),
    }
}

fn keys(n: usize) -> Vec<pallas::Scalar> {
    (0..n).map(|_| pallas::Scalar::random(OsRng)).collect()
}

fn honest_with(password: &[u8], domains: usize) -> (pallas::Base, HistoryTagWitness) {
    let u = history_input(owner(), password);
    let w = HistoryTagWitness::honest(u, random_blind(OsRng), &keys(domains));
    (u, w)
}

fn honest(password: &[u8]) -> (pallas::Base, HistoryTagWitness) {
    honest_with(password, 1)
}

/// A password whose canonical offset is at least `min`, found by search.
fn password_with_offset_at_least(min: u64) -> (Vec<u8>, pallas::Base, HistoryTagWitness) {
    (0u32..)
        .map(|i| format!("Str0ngP@ss{i}").into_bytes())
        .find_map(|p| {
            let (u, w) = honest(&p);
            (w.offset >= min).then_some((p, u, w))
        })
        .expect("offsets are unbounded over passwords")
}

const PASSWORD: &[u8] = b"Str0ngP@ssword!";

/// Smallest k at which the binder with Gadget H over one and two domains lays
/// out: a size measurement, run on demand.
#[test]
#[ignore]
fn measure_gadget_h_k() {
    for domains in [1, 2] {
        let (u, w) = honest_with(PASSWORD, domains);
        let inst = instances(u, &w);
        let circuit = HistoryTestCircuit {
            password: padded(PASSWORD),
            witness: w,
        };
        let k = (10..=16)
            .find(|&k| MockProver::run(k, &circuit, vec![inst.clone()]).is_ok())
            .expect("fits by 16");
        eprintln!("binder + Gadget H, {domains} domain(s): k = {k}");
    }
}

#[test]
fn the_non_square_is_a_non_square() {
    assert!(bool::from(non_square().sqrt().is_none()));
}

#[test]
fn an_honest_tag_satisfies_the_gadget() {
    let (u, w) = honest(PASSWORD);
    let inst = instances(u, &w);
    assert_eq!(verify(PASSWORD, w, inst), "Ok(())");
}

/// Two comparison domains (an older epoch under another key) share one `B`.
#[test]
fn two_domains_satisfy_the_gadget() {
    let (u, w) = honest_with(PASSWORD, 2);
    let inst = instances(u, &w);
    assert_eq!(verify(PASSWORD, w, inst), "Ok(())");
}

/// The witness rebuilt from the evaluator's answers is the honest one.
#[test]
fn the_witness_from_evaluations_is_the_honest_witness() {
    let ks = keys(2);
    let r = random_blind(OsRng);
    let u = history_input(owner(), PASSWORD);
    let b = crate::history::blind_request(u, r);
    let z: Vec<_> = ks
        .iter()
        .map(|k| crate::history::evaluate(*k, b).unwrap())
        .collect();
    let from_z = HistoryTagWitness::from_evaluations(u, r, &z);
    let honest = HistoryTagWitness::honest(u, r, &ks);
    assert_eq!(from_z.n, honest.n);
    assert_eq!(
        (from_z.h, from_z.offset, from_z.w),
        (honest.h, honest.offset, honest.w)
    );
}

/// The honest tags are the ones `crate::history` derives natively.
#[test]
fn the_gadget_tags_match_the_native_tags() {
    let ks = keys(2);
    let r = random_blind(OsRng);
    let u = history_input(owner(), PASSWORD);
    let cs = comparison_domains(2);
    let w = HistoryTagWitness::honest(u, r, &ks);
    let (b, tags) = tag_instances(&cs, u, &w);
    assert_eq!(b, crate::history::blind_request(u, r));
    for ((k, c), (z, t)) in ks.iter().zip(&cs).zip(tags) {
        assert_eq!(Some(z), crate::history::evaluate(*k, b));
        assert_eq!(t, crate::history::finalize_tag(*c, u, r, z));
    }
}

/// Skipping the first valid offset for a later one is refused: the first one
/// has no non-square witness.
#[test]
fn a_later_offset_than_the_first_is_refused() {
    let (u, honest_w) = honest(PASSWORD);
    let later = (honest_w.offset + 1..TRIES as u64)
        .find(|&j| {
            let x = u + pallas::Base::from(j);
            bool::from((x.square() * x + pallas::Base::from(5u64)).sqrt().is_some())
        })
        .expect("another point within the tries");
    let x = u + pallas::Base::from(later);
    let mut y = (x.square() * x + pallas::Base::from(5u64)).sqrt().unwrap();
    if y.to_repr()[0] & 1 == 1 {
        y = -y;
    }
    let h = pallas::Affine::from_xy(x, y).unwrap();
    let w_vals = (0..later)
        .map(|i| {
            let xi = u + pallas::Base::from(i);
            (non_square() * (xi.square() * xi + pallas::Base::from(5u64)))
                .sqrt()
                .unwrap_or(pallas::Base::ONE)
        })
        .collect();
    let k = pallas::Scalar::random(OsRng);
    let w = HistoryTagWitness {
        h,
        offset: later,
        w: w_vals,
        r: honest_w.r,
        n: vec![(pallas::Point::from(h) * k).to_affine()],
    };
    let inst = instances(u, &w);
    let failures = verify(PASSWORD, w, inst);
    assert!(failures.contains("history scan"), "{failures}");
}

/// Claiming an offset below the first valid one: there is no curve point at
/// that x, and the real point does not sit at `u + offset`.
#[test]
fn an_offset_below_the_first_is_refused() {
    let (password, u, honest_w) = password_with_offset_at_least(1);
    let x = u + pallas::Base::from(honest_w.offset - 1);
    assert!(bool::from(
        (x.square() * x + pallas::Base::from(5u64)).sqrt().is_none()
    ));
    let mut w = honest_w.clone();
    w.offset -= 1;
    w.w.pop();
    let inst = instances(u, &honest_w);
    let failures = verify(&password, w, inst);
    assert!(failures.contains("history bind"), "{failures}");
}

/// The odd root is not the prescribed sign.
#[test]
fn the_odd_root_is_refused() {
    let (u, mut w) = honest(PASSWORD);
    w.h = -w.h;
    w.n = w.n.iter().map(|n| -*n).collect();
    let inst = instances(u, &w);
    let failures = verify(PASSWORD, w, inst);
    assert!(failures.contains("Lookup"), "{failures}");
}

#[test]
fn a_zero_blind_is_refused() {
    let (u, mut w) = honest(PASSWORD);
    w.r = pallas::Base::ZERO;
    let c = comparison_domains(1)[0];
    // For r = 0, B and Z are the identity, (0, 0) in the ECC chip; the tag is
    // still the honest one, so only the blind can fail.
    let t = crate::poseidon::poseidon_hash_chain(&[c, u, *w.n[0].coordinates().unwrap().x()]);
    let zero = pallas::Base::ZERO;
    let inst = vec![owner(), c, zero, zero, zero, zero, t];
    let failures = verify(PASSWORD, w, inst);
    assert!(failures.contains("history bind"), "{failures}");
}

/// The tag is bound to the password bytes through the binder's hash: the
/// tag of another password does not verify over these bytes.
#[test]
fn another_passwords_tag_is_refused() {
    let (u_other, w_other) = honest(b"0therP@ssword!!");
    let inst = instances(u_other, &w_other);
    assert_ne!(verify(PASSWORD, w_other, inst), "Ok(())");
}

/// A tag for another owner domain does not verify under this owner's `d`.
#[test]
fn another_owners_tag_is_refused() {
    let other = domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"bob"]);
    let u = history_input(other, PASSWORD);
    let w = HistoryTagWitness::honest(u, random_blind(OsRng), &keys(1));
    let inst = instances(u, &w);
    assert_ne!(verify(PASSWORD, w, inst), "Ok(())");
}
