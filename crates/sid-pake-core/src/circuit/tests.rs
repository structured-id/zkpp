use super::*;
use crate::test_support::{history_evaluation, history_inputs, oprf_element};
use halo2_proofs::dev::MockProver;

fn make_password(s: &str) -> ([u8; MAX_PASSWORD_LEN], u32) {
    let mut buf = [0u8; MAX_PASSWORD_LEN];
    let bytes = s.as_bytes();
    buf[..bytes.len()].copy_from_slice(bytes);
    (buf, bytes.len() as u32)
}

/// The combined circuit for `password` with a breach `filter` and `domains`
/// history comparison domains.
fn circuit(password: &str, filter: BloomFilter, domains: usize) -> ZkppCircuit {
    let (buf, len) = make_password(password);
    ZkppCircuit {
        password: buf,
        password_len: len,
        blind: pallas::Base::from(42u64),
        policy: CE_DEFAULT_POLICY,
        breach_filter: filter,
        history_domains: domains,
        history: Some(history_inputs(password.as_bytes(), domains)),
    }
}

fn satisfied(circuit: &ZkppCircuit) -> bool {
    MockProver::run(ZKPP_K, circuit, vec![circuit.instance_values()])
        .unwrap()
        .verify()
        .is_ok()
}

/// Print circuit statistics: k parameter, total rows, proof size.
///
/// Run with: cargo test -p sid-pake-core -- --nocapture test_circuit_stats
#[test]
fn test_circuit_stats() {
    use crate::keygen::{generate_params, generate_pk};
    use crate::prover::ZkppProver;

    let c = circuit("Str0ngP@ssword!", BloomFilter::new(BREACH_PARAMS), 1);
    MockProver::run(ZKPP_K, &c, vec![c.instance_values()])
        .unwrap()
        .assert_satisfied();

    let shape = CircuitShape::single_domain(CE_DEFAULT_POLICY);
    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, shape).unwrap();
    let prover = ZkppProver::new(params, pk, shape);
    let keys = [<pallas::Scalar as ff::Field>::random(rand::rngs::OsRng)];
    let bound = prover
        .prove(
            b"Str0ngP@ssword!",
            pallas::Scalar::from(42u64),
            b"ctx",
            &history_evaluation(b"Str0ngP@ssword!", &keys),
        )
        .unwrap();

    let total_rows = 1u64 << ZKPP_K;
    let proof_size = bound.snark_proof.0.len();

    let mut params_bytes = Vec::new();
    crate::keygen::write_params(prover.params(), &mut params_bytes).unwrap();

    println!("\n╔══════════════════════════════════════════╗");
    println!("║     ZKPP Circuit Statistics (k={ZKPP_K})      ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║ Total rows:    {total_rows:>8} (2^{ZKPP_K})          ║");
    println!("║ Proof size:    {proof_size:>8} bytes            ║");
    println!(
        "║ Instances:     {:>8}                  ║",
        instance_count(1)
    );
    println!(
        "║ Params:        {:>8} bytes            ║",
        params_bytes.len()
    );
    println!("╚══════════════════════════════════════════╝\n");

    assert!(proof_size > 1000, "proof too small: {proof_size}");
    assert!(proof_size < 100_000, "proof too large: {proof_size}");
}

#[test]
fn test_combined_registration() {
    assert!(satisfied(&circuit(
        "Str0ngP@ssword!",
        BloomFilter::new(BREACH_PARAMS),
        1
    )));
}

/// Two comparison domains (a retained older epoch under another key) fit the
/// same k and satisfy the combined circuit.
#[test]
fn test_combined_two_history_domains() {
    assert!(satisfied(&circuit(
        "Str0ngP@ssword!",
        BloomFilter::new(BREACH_PARAMS),
        2
    )));
}

/// The exposed OPAQUE element is `blind·H(P)` of the checked password: a
/// prover keeping the strong password (and its policy check) cannot expose
/// the element of a weak password, of the same password under another
/// blind, or of an unrelated point.
#[test]
fn the_exposed_element_is_the_blinded_hash_of_the_checked_password() {
    use group::{Curve, Group};
    use pasta_curves::arithmetic::CurveAffine;

    let c = circuit("Str0ngP@ssword!", BloomFilter::new(BREACH_PARAMS), 1);
    let honest = c.instance_values();
    let substitute = |m: pallas::Affine| {
        let coords = m.coordinates().unwrap();
        let mut instances = honest.clone();
        instances[0] = *coords.x();
        instances[1] = *coords.y();
        MockProver::run(ZKPP_K, &c, vec![instances])
            .unwrap()
            .verify()
            .is_ok()
    };
    let (weak, _) = make_password("abc");
    let blind = crate::history::blind_scalar(c.blind);
    let weak_m = oprf_element(&weak[..3], blind);
    assert!(!substitute(weak_m), "a weak password's element");
    let other_blind = oprf_element(b"Str0ngP@ssword!", pallas::Scalar::from(43u64));
    assert!(!substitute(other_blind), "another blind's element");
    let unrelated = (pallas::Point::generator() * pallas::Scalar::from(5u64)).to_affine();
    assert!(!substitute(unrelated), "an unrelated point");
    assert!(
        substitute(oprf_element(b"Str0ngP@ssword!", blind)),
        "the honest element"
    );
}

/// The history tags are over the checked password: tags derived for another
/// password do not satisfy the circuit for this one.
#[test]
fn test_combined_history_of_another_password_rejected() {
    let mut c = circuit("Str0ngP@ssword!", BloomFilter::new(BREACH_PARAMS), 1);
    c.history = Some(history_inputs(b"0therP@ssword!!", 1));
    assert!(!satisfied(&c));
}

/// End-to-end SOUNDNESS: a breached password is rejected by the FULL combined
/// circuit (Gadget D wired + bound to the same password as Gadgets A/C).
#[test]
fn test_combined_breach_member_rejected() {
    use crate::poseidon::{bytes_to_field_elements, poseidon_hash_chain};
    let (password, _) = make_password("Str0ngP@ssword!");
    let mut filter = BloomFilter::new(BREACH_PARAMS);
    filter.insert_hash(poseidon_hash_chain(&bytes_to_field_elements(&password)));
    assert!(
        !satisfied(&circuit("Str0ngP@ssword!", filter, 1)),
        "a breached password must be rejected by the combined circuit"
    );
}

/// End-to-end COMPLETENESS: a non-breached password passes the full circuit
/// even when the filter is populated with OTHER breached passwords.
#[test]
fn test_combined_breach_nonmember_accepted() {
    use crate::poseidon::{bytes_to_field_elements, poseidon_hash_chain};
    let mut filter = BloomFilter::new(BREACH_PARAMS);
    for other in ["password123", "Qwerty123!", "Letmein99#"] {
        let (op, _) = make_password(other);
        filter.insert_hash(poseidon_hash_chain(&bytes_to_field_elements(&op)));
    }
    let (password, _) = make_password("Str0ngP@ssword!");
    let h = poseidon_hash_chain(&bytes_to_field_elements(&password));
    assert!(
        !filter.maybe_contains_hash(h),
        "test password unexpectedly a member"
    );
    assert!(satisfied(&circuit("Str0ngP@ssword!", filter, 1)));
}

/// Stress test: 1000 random passwords — 750 valid, 250 policy failures.
/// Verifies circuit correctness with MockProver across diverse inputs.
///
/// Run with: cargo test -p sid-pake-core -- --nocapture test_stress_1000
#[test]
fn test_stress_1000() {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut pass_count = 0u32;
    let mut fail_count = 0u32;
    let total = 1000;

    for i in 0..total {
        // Generate password: ~75% valid, ~25% failing policy
        let (password, expect_pass) = if i % 4 == 0 {
            let weak = match i % 20 {
                0 => "Ab1",       // too short
                4 => "abcdefg1",  // no uppercase
                8 => "ABCDEFG1",  // no lowercase
                12 => "Abcdefgh", // no digit
                16 => "abcde",    // all lowercase, short
                _ => "a1B",       // short
            };
            (weak.to_string(), false)
        } else {
            let mut pw = String::new();
            pw.push((b'A' + (rng.r#gen::<u8>() % 26)) as char);
            pw.push((b'a' + (rng.r#gen::<u8>() % 26)) as char);
            pw.push((b'0' + (rng.r#gen::<u8>() % 10)) as char);
            for _ in 0..5 + (rng.r#gen::<usize>() % 8) {
                pw.push((b'a' + (rng.r#gen::<u8>() % 26)) as char);
            }
            (pw, true)
        };

        let mut c = circuit(&password, BloomFilter::new(BREACH_PARAMS), 1);
        c.blind = pallas::Base::from((i + 1) as u64);
        // A password below the policy has no satisfying witness.
        if satisfied(&c) {
            assert!(
                expect_pass,
                "iteration {i}: expected policy fail but got pass"
            );
            pass_count += 1;
        } else {
            assert!(
                !expect_pass,
                "iteration {i}: expected policy pass but got fail"
            );
            fail_count += 1;
        }
    }

    println!("\n╔══════════════════════════════════════════╗");
    println!("║     Stress Test Results ({total} iterations)  ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║ Policy PASS:   {pass_count:>8}                  ║");
    println!("║ Policy FAIL:   {fail_count:>8}                  ║");
    println!("║ Total:         {total:>8}                  ║");
    println!("╚══════════════════════════════════════════╝\n");

    assert!(pass_count >= 700, "too few passes: {pass_count}");
    assert!(fail_count >= 250, "too few fails: {fail_count}");
}

/// Performance: prove + verify timing for one and two history domains.
///
/// Reports absolute timings and hardware context; no baselines, results
/// depend on the machine. Run with `--nocapture`.
#[test]
fn test_perf_binding() {
    use crate::keygen::{generate_params, generate_pk};
    use crate::prover::ZkppProver;
    use crate::verifier::ZkppVerifier;

    let params = generate_params(ZKPP_K);
    let password = b"Str0ngP@ssword!";
    let ctx = b"ctx";
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    for domains in [1, 2] {
        let shape = CircuitShape {
            policy: CE_DEFAULT_POLICY,
            history_domains: domains,
        };
        let pk = generate_pk(&params, shape).unwrap();
        let vk = pk.get_vk().clone();
        let prover = ZkppProver::new(params.clone(), pk, shape);
        let verifier = ZkppVerifier::new(params.clone(), vk, shape);
        let keys: Vec<_> = (0..domains)
            .map(|_| <pallas::Scalar as ff::Field>::random(rand::rngs::OsRng))
            .collect();

        // Warm up.
        let _ = prover.prove(
            password,
            pallas::Scalar::from(1u64),
            ctx,
            &history_evaluation(password, &keys),
        );

        let prove_iters = 5;
        let start = std::time::Instant::now();
        let mut last = None;
        for i in 0..prove_iters {
            let blind = pallas::Scalar::from((i + 42) as u64);
            let eval = history_evaluation(password, &keys);
            last = Some((prover.prove(password, blind, ctx, &eval).unwrap(), blind));
        }
        let prove_avg_ms = start.elapsed().as_millis() as f64 / f64::from(prove_iters);
        let (bound, blind) = last.unwrap();
        let m = oprf_element(password, blind);

        let verify_iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..verify_iters {
            verifier.verify(&bound, ctx, m).unwrap();
        }
        let verify_avg_ms = start.elapsed().as_millis() as f64 / f64::from(verify_iters);

        println!(
            "ZKPP k={ZKPP_K} domains={domains} arch={} cores={cores}: prove_avg={prove_avg_ms:.1}ms verify_avg={verify_avg_ms:.1}ms proof={}B",
            std::env::consts::ARCH,
            bound.snark_proof.0.len()
        );
        assert!(prove_avg_ms > 0.0 && verify_avg_ms > 0.0);
    }
}
