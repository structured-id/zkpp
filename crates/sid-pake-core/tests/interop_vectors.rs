// Interop reference vectors for the pure-TS Pasta/ZKPP port (no-WASM prover).
// Prints canonical byte encodings the TS side must reproduce exactly. Run:
//   cargo test -p sid-pake-core --test interop_vectors --release -- --nocapture
use ff::PrimeField;
use group::{Curve, Group, GroupEncoding};
use pasta_curves::arithmetic::{Coordinates, CurveAffine};
use pasta_curves::pallas;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn aff_xy(p: pallas::Affine) -> (String, String) {
    let c: Coordinates<pallas::Affine> = Option::from(p.coordinates()).unwrap();
    (hex(c.x().to_repr().as_ref()), hex(c.y().to_repr().as_ref()))
}

/// Native timing for the same primitive the TS port benchmarks (scalarMul).
#[test]
fn time_native_primitives() {
    use ff::FromUniformBytes;
    use std::hint::black_box;
    use std::time::Instant;

    let g = pallas::Point::generator();
    let s = pallas::Scalar::from_uniform_bytes(&[0xab; 64]); // full-width scalar
    let n = 5000u32;

    let t = Instant::now();
    let mut acc = pallas::Point::identity();
    for _ in 0..n {
        acc = black_box(black_box(g) * black_box(s));
    }
    black_box(acc);
    let scalarmul_us = t.elapsed().as_nanos() as f64 / n as f64 / 1000.0;

    std::fs::write(
        "/tmp/sid_native_timing.txt",
        format!("NATIVE_SCALARMUL_US={scalarmul_us:.3}\n"),
    )
    .unwrap();
}

/// Dump P128Pow5T3 Poseidon constants + hash vectors for the TS port.
#[test]
fn dump_poseidon() {
    use halo2_gadgets::poseidon::primitives::{P128Pow5T3, Spec};
    use sid_pake_core::poseidon::{legacy_history_digest, poseidon_hash_1, poseidon_hash_2};

    let (rc, mds, _mds_inv) = <P128Pow5T3 as Spec<pallas::Base, 3, 2>>::constants();

    let mut s = String::new();
    s.push_str(&format!(
        "FULL_ROUNDS={}\nPARTIAL_ROUNDS={}\nROUNDS={}\n",
        <P128Pow5T3 as Spec<pallas::Base, 3, 2>>::full_rounds(),
        <P128Pow5T3 as Spec<pallas::Base, 3, 2>>::partial_rounds(),
        rc.len(),
    ));
    for round in &rc {
        for fe in round {
            s.push_str("RC ");
            s.push_str(&hex(fe.to_repr().as_ref()));
            s.push('\n');
        }
    }
    for row in &mds {
        for fe in row {
            s.push_str("MDS ");
            s.push_str(&hex(fe.to_repr().as_ref()));
            s.push('\n');
        }
    }
    // Hash vectors.
    let h2 = poseidon_hash_2(&[pallas::Base::from(1u64), pallas::Base::from(2u64)]);
    let h1 = poseidon_hash_1(&[pallas::Base::from(7u64)]);
    let hc = legacy_history_digest(b"Str0ngP@ssword!", &[42u8; 32]);
    s.push_str(&format!("HASH2_1_2={}\n", hex(h2.to_repr().as_ref())));
    s.push_str(&format!("HASH1_7={}\n", hex(h1.to_repr().as_ref())));
    s.push_str(&format!(
        "LEGACY_HISTORY_DIGEST={}\n",
        hex(hc.to_repr().as_ref())
    ));

    // Native HashToCurve (gadget C): H_p = (x,y), u = Poseidon(pw), offset.
    let (hp, u, off) = sid_pake_core::circuit::gadget_c::hash_to_curve_outside(b"Str0ngP@ssword!");
    let (hpx, hpy) = aff_xy(hp);
    s.push_str(&format!("HP_X={hpx}\nHP_Y={hpy}\n"));
    s.push_str(&format!("HP_U={}\n", hex(u.to_repr().as_ref())));
    s.push_str(&format!("HP_OFFSET={}\n", hex(off.to_repr().as_ref())));

    std::fs::write("/tmp/sid_poseidon.txt", &s).unwrap();
}

/// Dump EvaluationDomain coset-FFT vector (coeff_to_extended) + ZETA.
#[test]
fn dump_domain() {
    use ff::WithSmallOrderMulGroup;
    use halo2_proofs::poly::EvaluationDomain;

    let k = 2u32;
    let j = 4u32;
    let domain = EvaluationDomain::<pallas::Base>::new(j, k);
    let mut poly = domain.empty_coeff();
    for i in 0..(1usize << k) {
        poly[i] = pallas::Base::from((i as u64) + 1);
    }
    let ext = domain.coeff_to_extended(poly);
    let zeta = <pallas::Base as WithSmallOrderMulGroup<3>>::ZETA;

    let mut out = format!(
        "DOM_ZETA={}\nDOM_K={k}\nDOM_J={j}\nDOM_EXTLEN={}\n",
        hex(zeta.to_repr().as_ref()),
        ext.len(),
    );
    for (i, v) in ext.iter().enumerate() {
        out.push_str(&format!("DOM_EXT_{i}={}\n", hex(v.to_repr().as_ref())));
    }
    std::fs::write("/tmp/sid_domain.txt", &out).unwrap();
}

/// Dump a Blake2b transcript vector: absorb point + scalar, squeeze challenge.
#[test]
fn dump_transcript() {
    use halo2_proofs::transcript::{Blake2bWrite, Challenge255, Transcript};
    use pasta_curves::vesta;

    let mut t = Blake2bWrite::<Vec<u8>, vesta::Affine, Challenge255<vesta::Affine>>::init(vec![]);
    let p = (vesta::Point::generator() * vesta::Scalar::from(3u64)).to_affine();
    t.common_point(p).unwrap();
    let sc = vesta::Scalar::from(42u64);
    t.common_scalar(sc).unwrap();
    let ch = *t.squeeze_challenge_scalar::<()>();

    let out = format!(
        "TR_POINT={}\nTR_SCALAR={}\nTR_CHALLENGE={}\n",
        hex(p.to_bytes().as_ref()),
        hex(sc.to_repr().as_ref()),
        hex(ch.to_repr().as_ref()),
    );
    std::fs::write("/tmp/sid_transcript.txt", &out).unwrap();
}

/// Dump IPA commitment params (g[], w) + a test commitment for the TS commit.
/// g[]/w are pub(crate), so we recover them via commit of unit/zero polynomials:
///   g[i] = commit(e_i, 0),  w = commit(0, 1).
#[test]
fn dump_ipa() {
    use ff::Field as _;
    use halo2_proofs::poly::EvaluationDomain;
    use halo2_proofs::poly::commitment::{Blind, Params};
    use pasta_curves::vesta;

    let k = 2u32;
    let n = 1usize << k;
    let params = Params::<vesta::Affine>::new(k);
    let domain = EvaluationDomain::<vesta::Scalar>::new(1, k);

    let mut out = String::new();
    out.push_str(&format!("IPA_K={k}\n"));
    for i in 0..n {
        let mut poly = domain.empty_coeff();
        poly[i] = vesta::Scalar::ONE;
        let gi = params.commit(&poly, Blind(vesta::Scalar::ZERO)).to_affine();
        out.push_str(&format!("IPA_G_{i}={}\n", hex(gi.to_bytes().as_ref())));
    }
    let zero = domain.empty_coeff();
    let w = params.commit(&zero, Blind(vesta::Scalar::ONE)).to_affine();
    out.push_str(&format!("IPA_W={}\n", hex(w.to_bytes().as_ref())));

    // Test commitment: poly = [1,2,3,4], blind = 7.
    let mut tp = domain.empty_coeff();
    for i in 0..n {
        tp[i] = vesta::Scalar::from((i as u64) + 1);
    }
    let cm = params
        .commit(&tp, Blind(vesta::Scalar::from(7u64)))
        .to_affine();
    out.push_str(&format!(
        "IPA_COMMIT_1234_b7={}\n",
        hex(cm.to_bytes().as_ref())
    ));

    std::fs::write("/tmp/sid_ipa.txt", &out).unwrap();
}

/// Dump Vesta curve vectors (IPA commitment curve: vesta scalar = pallas base).
#[test]
fn dump_vesta() {
    use pasta_curves::vesta;
    let g = vesta::Point::generator();
    let s = vesta::Scalar::from(12345u64);
    let (gx, gy) = {
        let c: Coordinates<vesta::Affine> = Option::from(g.to_affine().coordinates()).unwrap();
        (hex(c.x().to_repr().as_ref()), hex(c.y().to_repr().as_ref()))
    };
    let out = format!(
        "VESTA_GEN={}\nVESTA_GX={gx}\nVESTA_GY={gy}\nVESTA_2G={}\nVESTA_12345G={}\n",
        hex(g.to_affine().to_bytes().as_ref()),
        hex((g + g).to_affine().to_bytes().as_ref()),
        hex((g * s).to_affine().to_bytes().as_ref()),
    );
    std::fs::write("/tmp/sid_vesta.txt", &out).unwrap();
}

/// Dump Pasta Fp FFT params + a best_fft vector for the TS NTT.
#[test]
fn dump_fft() {
    use ff::Field;
    use halo2_proofs::arithmetic::best_fft;

    let s = <pallas::Base as ff::PrimeField>::S;
    let rou = <pallas::Base as ff::PrimeField>::ROOT_OF_UNITY;
    let log_n = 3u32;
    let n = 1usize << log_n;
    let omega = rou.pow_vartime([1u64 << (s - log_n)]);

    let mut data: Vec<pallas::Base> = (1..=n as u64).map(pallas::Base::from).collect();
    best_fft(&mut data, omega, log_n);

    let mut out = String::new();
    out.push_str(&format!("FP_S={s}\n"));
    out.push_str(&format!(
        "FP_ROOT_OF_UNITY={}\n",
        hex(rou.to_repr().as_ref())
    ));
    out.push_str(&format!("FFT_OMEGA8={}\n", hex(omega.to_repr().as_ref())));
    out.push_str(&format!("FFT_LOG_N={log_n}\n"));
    for (i, v) in data.iter().enumerate() {
        out.push_str(&format!("FFT_OUT_{i}={}\n", hex(v.to_repr().as_ref())));
    }
    std::fs::write("/tmp/sid_fft.txt", &out).unwrap();
}

/// Operation-context vector: the transcript scalar both provers absorb first.
#[test]
fn print_operation_context_vector() {
    use sid_pake_core::binding::{operation_context, transcript_context};

    let context = operation_context(&[7u8; 16], b"registration-request");
    let out = format!(
        "OPCTX={}\nOPCTX_SCALAR={}\n",
        hex(&context),
        hex(transcript_context(&context).to_repr().as_ref()),
    );
    std::fs::write("/tmp/sid_operation_context.txt", &out).unwrap();
    print!("{out}");
}

#[test]
fn print_pasta_vectors() {
    // Pallas group generator.
    let g = pallas::Point::generator();
    println!("PALLAS_GEN={}", hex(g.to_affine().to_bytes().as_ref()));

    // Doubling and a fixed scalar multiple (group law + scalar mul checks).
    println!("PALLAS_2G={}", hex((g + g).to_affine().to_bytes().as_ref()));
    let s = pallas::Scalar::from(12345u64);
    println!(
        "PALLAS_12345G={}",
        hex((g * s).to_affine().to_bytes().as_ref())
    );

    // Identity encoding.
    println!(
        "PALLAS_ID={}",
        hex(pallas::Point::identity().to_affine().to_bytes().as_ref())
    );

    // Base-field element canonical encoding (to_repr is little-endian).
    println!("FP_7={}", hex(pallas::Base::from(7u64).to_repr().as_ref()));
    // Scalar-field element canonical encoding.
    println!(
        "FQ_7={}",
        hex(pallas::Scalar::from(7u64).to_repr().as_ref())
    );

    // from_uniform_bytes reference: 64 bytes of 0x01 → scalar (to validate FS challenge reduction).
    use ff::FromUniformBytes;
    let wide = [1u8; 64];
    let red = pallas::Scalar::from_uniform_bytes(&wide);
    println!("FQ_FROMUNIFORM_01x64={}", hex(red.to_repr().as_ref()));
}

/// Dump gadget_a (policy engine) witness for a known password — byte/active/flags/acc.
#[test]
fn dump_gadget_a_witness() {
    use ff::PrimeField;
    let pw = b"Str0ngP@ss"; // len 10: upper S,P; lower t,r,n,g,s,s; digit 0; symbol @
    let len = pw.len();
    let classify = |b: u8| -> (bool, bool, bool, bool) {
        (
            (65..=90).contains(&b),
            (97..=122).contains(&b),
            (48..=57).contains(&b),
            (33..=47).contains(&b)
                || (58..=64).contains(&b)
                || (91..=96).contains(&b)
                || (123..=126).contains(&b),
        )
    };
    let n = 128usize;
    let (mut au, mut al, mut ad, mut as_) = (0u64, 0u64, 0u64, 0u64);
    let mut rows: Vec<String> = Vec::new();
    // The circuit always processes n slots: the password bytes, then inactive padding.
    for slot in pw
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::repeat(None))
        .take(n)
    {
        let active = slot.is_some();
        let b = slot.unwrap_or(0u8);
        let (u, l, d, s) = if active {
            classify(b)
        } else {
            (false, false, false, false)
        };
        au += u as u64;
        al += l as u64;
        ad += d as u64;
        as_ += s as u64;
        rows.push(format!(
            "{},{},{},{},{},{},{},{},{},{}",
            b, active as u8, u as u8, l as u8, d as u8, s as u8, au, al, ad, as_
        ));
    }
    let pol = sid_pake_core::types::CE_DEFAULT_POLICY;
    let compliant = (len as u64) >= pol.min_length as u64
        && au >= pol.min_upper as u64
        && al >= pol.min_lower as u64
        && ad >= pol.min_digit as u64
        && as_ >= pol.min_symbol as u64;
    let _ = pallas::Base::from(1u64).to_repr();
    let out = format!(
        "GA_LEN={len}\nGA_COMPLIANT={}\nGA_FINAL={au},{al},{ad},{as_}\nGA_ROWS={}\n",
        compliant as u8,
        rows.join(";")
    );
    std::fs::write("/tmp/sid_gadget_a.txt", &out).unwrap();
}

/// Dump gadget_b diff-accumulator witness (registration: p_old=0, diff=p_new², acc=sum).
#[test]
fn dump_gadget_b_diffacc() {
    use ff::{Field, PrimeField};
    let pw = b"Str0ngP@ss";
    let n = 128usize;
    let mut acc = pallas::Base::zero();
    let mut rows: Vec<String> = Vec::new();
    let hx = |f: pallas::Base| -> String {
        f.to_repr()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    for i in 0..n {
        let pn = if i < pw.len() {
            pallas::Base::from(pw[i] as u64)
        } else {
            pallas::Base::zero()
        };
        let po = pallas::Base::zero();
        let d = (pn - po) * (pn - po);
        acc += d;
        rows.push(format!("{}:{}:{}", hx(pn), hx(d), hx(acc)));
    }
    let diff_inv = acc.invert().unwrap_or(pallas::Base::zero());
    let out = format!(
        "GB_ACC={}\nGB_DIFFINV={}\nGB_ROWS={}\n",
        hx(acc),
        hx(diff_inv),
        rows.join(";")
    );
    std::fs::write("/tmp/sid_gadget_b.txt", &out).unwrap();
}

/// Dump gadget_d (breach bloom) witness: breach_hash + 255-bit decomp + k indices.
#[test]
fn dump_gadget_d_witness() {
    use ff::PrimeField;
    use sid_pake_core::circuit::gadget_d::{BloomFilter, BloomParams};
    let pw = b"Str0ngP@ss";
    let pad_len = 64usize;
    let params = BloomParams {
        index_bits: 8,
        k: 4,
    };
    let hash = BloomFilter::breach_hash(pw, pad_len);
    let repr = hash.to_repr();
    let bits: String = (0..255usize)
        .map(|j| ((repr[j / 8] >> (j % 8)) & 1).to_string())
        .collect();
    let indices = BloomFilter::indices(params, hash);
    let hx = |f: pallas::Base| -> String {
        f.to_repr()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let out = format!(
        "GD_HASH={}\nGD_PADLEN={pad_len}\nGD_K={}\nGD_IB={}\nGD_BITS={bits}\nGD_INDICES={}\n",
        hx(hash),
        params.k,
        params.index_bits,
        indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    std::fs::write("/tmp/sid_gadget_d.txt", &out).unwrap();
}

/// Dump gadget_c opaque-binder application witness: H_p = HashToCurve(pw), M = blind·H_p.
#[test]
fn dump_gadget_c_binding() {
    use ff::PrimeField;
    use group::Curve;
    use pasta_curves::arithmetic::CurveAffine;
    use sid_pake_core::circuit::gadget_c::hash_to_curve_outside;
    let pw = b"Str0ngP@ss";
    let (hp, u, _off) = hash_to_curve_outside(pw);
    let blind = pallas::Scalar::from(7u64);
    let m = (hp * blind).to_affine();
    let hxb = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    let co = |p: pallas::Affine| {
        let c = p.coordinates().unwrap();
        (hxb(c.x().to_repr().as_ref()), hxb(c.y().to_repr().as_ref()))
    };
    let (hpx, hpy) = co(hp);
    let (mx, my) = co(m);
    let out = format!(
        "GC_U={}\nGC_HP_X={hpx}\nGC_HP_Y={hpy}\nGC_BLIND={}\nGC_M_X={mx}\nGC_M_Y={my}\n",
        hxb(u.to_repr().as_ref()),
        hxb(blind.to_repr().as_ref())
    );
    std::fs::write("/tmp/sid_gadget_c.txt", &out).unwrap();
}

/// Dump ECC incomplete double-and-add witness (the halo2_gadgets variable-base mul
/// core formula) for a known base + bits + init accumulator → z/λ1/λ2/x_a sequence.
#[test]
fn dump_ecc_incomplete() {
    use ff::{Field, PrimeField};
    use group::Curve;
    use pasta_curves::arithmetic::CurveAffine;
    use sid_pake_core::circuit::gadget_c::hash_to_curve_outside;
    let (hp, _u, _o) = hash_to_curve_outside(b"Str0ngP@ss");
    let c = hp.coordinates().unwrap();
    let (xp, yp0) = (*c.x(), *c.y());
    // Init acc = [2]·base, z = 0 (representative init for the formula test).
    let two = (hp + hp).to_affine();
    let ci = two.coordinates().unwrap();
    let (mut xa, mut ya) = (*ci.x(), *ci.y());
    let mut z = pallas::Base::zero();
    let bits = [
        true, false, true, true, false, true, false, false, true, true,
    ];
    let hx = |f: pallas::Base| -> String {
        f.to_repr()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let mut rows: Vec<String> = Vec::new();
    for &k in bits.iter() {
        z = pallas::Base::from(2) * z + pallas::Base::from(k as u64);
        let yp = if k { yp0 } else { -yp0 };
        let l1 = (ya - yp) * (xa - xp).invert().unwrap();
        let xr = l1.square() - xa - xp;
        let l2 = ya * pallas::Base::from(2) * (xa - xr).invert().unwrap() - l1;
        let xa_new = l2.square() - xa - xr;
        let ya_new = l2 * (xa - xa_new) - ya;
        rows.push(format!("{}:{}:{}:{}", hx(z), hx(l1), hx(l2), hx(xa_new)));
        xa = xa_new;
        ya = ya_new;
    }
    let out = format!(
        "ECC_XP={}\nECC_YP={}\nECC_X0={}\nECC_Y0={}\nECC_ROWS={}\n",
        hx(xp),
        hx(yp0),
        hx(*ci.x()),
        hx(*ci.y()),
        rows.join(";")
    );
    std::fs::write("/tmp/sid_ecc_incomplete.txt", &out).unwrap();
}

/// Dump ECC complete-addition point sequence (variable-base mul low 3 bits):
/// per bit acc = [2]·acc + (k ? base : -base), with z = 2·z + k.
#[test]
fn dump_ecc_complete() {
    use ff::PrimeField;
    use group::Curve;
    use pasta_curves::arithmetic::CurveAffine;
    use sid_pake_core::circuit::gadget_c::hash_to_curve_outside;
    let (hp, _u, _o) = hash_to_curve_outside(b"Str0ngP@ss");
    // Init acc = [5]·base (arbitrary non-identity start for the sequence test).
    let mut acc = (hp * pallas::Scalar::from(5u64)).to_affine();
    let mut z = pallas::Base::from(11u64);
    let bits = [true, false, true];
    let hx = |f: pallas::Base| -> String {
        f.to_repr()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    };
    let co = |p: pallas::Affine| {
        let c = p.coordinates().unwrap();
        (hx(*c.x()), hx(*c.y()))
    };
    let mut rows: Vec<String> = Vec::new();
    for &k in bits.iter() {
        z = pallas::Base::from(2) * z + pallas::Base::from(k as u64);
        let u = if k { hp } else { -hp };
        // acc = [2]acc + U  (complete additions)
        let dbl = (acc + acc).to_affine();
        acc = (dbl + u).to_affine();
        let (ax, ay) = co(acc);
        rows.push(format!("{}:{}:{}", hx(z), ax, ay));
    }
    let (ix, iy) = co((hp * pallas::Scalar::from(5u64)).to_affine());
    let out = format!("ECCC_X0={ix}\nECCC_Y0={iy}\nECCC_ROWS={}\n", rows.join(";"));
    std::fs::write("/tmp/sid_ecc_complete.txt", &out).unwrap();
}

/// Dump ECC variable-base-mul scalar decomposition: k = scalar + t_q, big-endian bits.
#[test]
fn dump_ecc_decompose() {
    use ff::PrimeField;
    let scalar = pallas::Scalar::from(7u64);
    // t_q where F_q = 2^254 + t_q.
    let t_q: u128 = 45560315531506369815346746415080538113;
    let mut k = [0u8; 32];
    k.copy_from_slice(scalar.to_repr().as_ref());
    // k = scalar + t_q (256-bit add, little-endian), not reduced.
    let mut carry = 0u16;
    let tq_bytes = t_q.to_le_bytes();
    for i in 0..32 {
        let tb = if i < 16 { tq_bytes[i] as u16 } else { 0 };
        let s = k[i] as u16 + tb + carry;
        k[i] = (s & 0xff) as u8;
        carry = s >> 8;
    }
    // big-endian bits of k (255 bits, MSB-first).
    let bits: String = (0..255usize)
        .rev()
        .map(|j| ((k[j / 8] >> (j % 8)) & 1).to_string())
        .collect();
    let out = format!("ECCD_SCALAR=7\nECCD_TQ={}\nECCD_BITS={bits}\n", t_q);
    std::fs::write("/tmp/sid_ecc_decompose.txt", &out).unwrap();
}
