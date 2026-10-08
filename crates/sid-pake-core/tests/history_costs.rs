// SPDX-License-Identifier: AGPL-3.0-only
//! Server-side cost of the history operation: evaluator (VOPRF + DLEQ),
//! checker DLEQ verification, and the checker's KSF plus comparison under
//! candidate production Argon2id parameters, alone and under concurrency.
//! A measurement, run on demand with --run-ignored and --no-capture.

use std::time::{Duration, Instant};

use ff::Field;
use group::Group;
use pasta_curves::pallas;
use sid_pake_core::history::{
    HistoryEntry, KsfParams, blind_request, check_candidate, domain_element, evaluate_with_proof,
    finalize_tag, history_input, ksf, random_blind, verify_evaluation,
};

const CANDIDATES: [(&str, KsfParams); 4] = [
    (
        "OWASP 19 MiB t=2 p=1",
        KsfParams {
            memory_kib: 19 * 1024,
            passes: 2,
            lanes: 1,
        },
    ),
    (
        "OWASP 46 MiB t=1 p=1",
        KsfParams {
            memory_kib: 46 * 1024,
            passes: 1,
            lanes: 1,
        },
    ),
    (
        "RFC 9106 64 MiB t=3 p=1",
        KsfParams {
            memory_kib: 64 * 1024,
            passes: 3,
            lanes: 1,
        },
    ),
    (
        "RFC 9106 64 MiB t=3 p=4",
        KsfParams {
            memory_kib: 64 * 1024,
            passes: 3,
            lanes: 4,
        },
    ),
];
const SALT: &[u8] = b"measurement-epoch-salt";

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn pct(xs: &mut [Duration], p: f64) -> Duration {
    xs.sort();
    xs[((xs.len() as f64 - 1.0) * p).round() as usize]
}

fn tag(k: pallas::Scalar, password: &[u8]) -> pallas::Base {
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"bench", b"alice"]);
    let c = domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-0"]);
    let u = history_input(d, password);
    let r = random_blind(rand::rng());
    let b = blind_request(u, r);
    let (z, _) = evaluate_with_proof(k, b, b"op", rand::rng()).unwrap();
    finalize_tag(c, u, r, z)
}

#[test]
#[ignore]
fn measure_history_costs() {
    let k = pallas::Scalar::random(&mut rand::rng());
    let pk = group::Curve::to_affine(&(pallas::Point::generator() * k));
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"bench", b"alice"]);
    let b = blind_request(history_input(d, b"Str0ngP@ssword!"), random_blind(rand::rng()));

    let eval: Vec<_> = (0..200)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(evaluate_with_proof(k, b, b"op", rand::rng()));
            t.elapsed()
        })
        .collect();
    let (z, proof) = evaluate_with_proof(k, b, b"op", rand::rng()).unwrap();
    let verify: Vec<_> = (0..200)
        .map(|_| {
            let t = Instant::now();
            assert!(verify_evaluation(pk, b, z, b"op", &proof));
            t.elapsed()
        })
        .collect();
    eprintln!(
        "evaluator VOPRF+DLEQ: {:?}   checker DLEQ verify: {:?}",
        median(eval),
        median(verify)
    );

    let candidate = tag(k, b"FreshN3wPassw0rd");
    for (name, params) in CANDIDATES {
        for depth in [1usize, 10, 24] {
            let retained: Vec<HistoryEntry> = (0..depth)
                .map(|i| ksf(tag(k, format!("Hist#{i:02}pwQ7").as_bytes()), SALT, params).unwrap())
                .collect();
            let runs: Vec<_> = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    check_candidate(candidate, SALT, params, &retained).unwrap();
                    t.elapsed()
                })
                .collect();
            eprintln!("{name}  N={depth:>2}: check = {:?}", median(runs));
        }
    }

    // Concurrency: 10 checker workers (one per core), 6 checks each, 64 MiB t=3 p=1.
    let params = CANDIDATES[2].1;
    let retained: Vec<HistoryEntry> = (0..10)
        .map(|i| ksf(tag(k, format!("Hist#{i:02}pwQ7").as_bytes()), SALT, params).unwrap())
        .collect();
    for workers in [1usize, 4, 10] {
        let start = Instant::now();
        let mut lat: Vec<Duration> = std::thread::scope(|s| {
            (0..workers)
                .map(|_| {
                    s.spawn(|| {
                        (0..6)
                            .map(|_| {
                                let t = Instant::now();
                                check_candidate(candidate, SALT, params, &retained).unwrap();
                                t.elapsed()
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let wall = start.elapsed();
        let n = lat.len();
        eprintln!(
            "concurrent 64 MiB t=3 p=1, N=10, {workers} workers: P50 {:?} P99 {:?}, {:.1} checks/s, peak KSF memory {} MiB",
            pct(&mut lat, 0.5),
            pct(&mut lat, 0.99),
            n as f64 / wall.as_secs_f64(),
            64 * workers
        );
    }
}
