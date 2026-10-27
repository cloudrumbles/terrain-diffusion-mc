//! Spatial postprocessing for model output, channel-major row-major arrays.
pub fn weights(n: usize) -> Vec<f32> {
    let mid = (n as f32 - 1.0) / 2.0;
    (0..n * n)
        .map(|i| {
            let wy = 1.0 - 0.999 * (((i / n) as f32 - mid).abs() / mid).min(1.0);
            let wx = 1.0 - 0.999 * (((i % n) as f32 - mid).abs() / mid).min(1.0);
            wy * wx
        })
        .collect()
}
pub fn bilinear(src: &[f32], h: usize, w: usize, dh: usize, dw: usize) -> Vec<f32> {
    let mut out = vec![0.0; dh * dw];
    for r in 0..dh {
        for c in 0..dw {
            let y = (r as f32 + 0.5) * h as f32 / dh as f32 - 0.5;
            let x = (c as f32 + 0.5) * w as f32 / dw as f32 - 0.5;
            let r0 = y.floor() as isize;
            let r1 = r0 + 1;
            let wr = y - r0 as f32;
            let c0 = x.floor() as isize;
            let c1 = c0 + 1;
            let wc = x - c0 as f32;
            let ix = |a: isize, b: isize| {
                src[a.clamp(0, h as isize - 1) as usize * w + b.clamp(0, w as isize - 1) as usize]
            };
            out[r * dw + c] = (1.0 - wr) * (1.0 - wc) * ix(r0, c0)
                + (1.0 - wr) * wc * ix(r0, c1)
                + wr * (1.0 - wc) * ix(r1, c0)
                + wr * wc * ix(r1, c1);
        }
    }
    out
}
pub fn bilinear_extrapolate(src: &[f32], h: usize, w: usize, dh: usize, dw: usize) -> Vec<f32> {
    let ph = h + 2;
    let pw = w + 2;
    let mut pad = vec![0.0; ph * pw];
    for r in 0..h {
        pad[(r + 1) * pw + 1..(r + 1) * pw + w + 1].copy_from_slice(&src[r * w..(r + 1) * w]);
    }
    for c in 1..=w {
        pad[c] = if h > 1 {
            2.0 * src[c - 1] - src[w + c - 1]
        } else {
            src[c - 1]
        };
        pad[(h + 1) * pw + c] = if h > 1 {
            2.0 * src[(h - 1) * w + c - 1] - src[(h - 2) * w + c - 1]
        } else {
            src[c - 1]
        };
    }
    for r in 0..ph {
        pad[r * pw] = if w > 1 {
            2.0 * pad[r * pw + 1] - pad[r * pw + 2]
        } else {
            pad[r * pw + 1]
        };
        pad[r * pw + w + 1] = if w > 1 {
            2.0 * pad[r * pw + w] - pad[r * pw + w - 1]
        } else {
            pad[r * pw + w]
        };
    }
    let newh = (dh as f64 + 2.0 * dh as f64 / h as f64).round() as usize;
    let neww = (dw as f64 + 2.0 * dw as f64 / w as f64).round() as usize;
    let resized = bilinear(&pad, ph, pw, newh, neww);
    let pady = (dh as f64 / h as f64).round() as usize;
    let padx = (dw as f64 / w as f64).round() as usize;
    let mut out = vec![0.0; dh * dw];
    for r in 0..dh {
        out[r * dw..(r + 1) * dw]
            .copy_from_slice(&resized[(r + pady) * neww + padx..(r + pady) * neww + padx + dw]);
    }
    out
}
pub fn denoise(
    res: &[f32],
    low: &[f32],
    h: usize,
    w: usize,
    lh: usize,
    lw: usize,
    sigma: f32,
) -> Vec<f32> {
    let up = bilinear_extrapolate(low, lh, lw, h, w);
    let decoded: Vec<f32> = res.iter().zip(up).map(|(&a, b)| a + b).collect();
    let down = bilinear(&decoded, h, w, lh, lw);
    let ks = 11usize;
    let mut kernel: Vec<f32> = (0..ks)
        .map(|i| (-0.5 * ((i as f32 - 5.0) / sigma).powi(2)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    let mut tmp = vec![0.0; lh * lw];
    let mut output = vec![0.0; lh * lw];
    for r in 0..lh {
        for c in 0..lw {
            for (i, &v) in kernel.iter().enumerate() {
                let cc = (c as isize + i as isize - 5).clamp(0, lw as isize - 1) as usize;
                tmp[r * lw + c] += v * down[r * lw + cc];
            }
        }
    }
    for r in 0..lh {
        for c in 0..lw {
            for (i, &v) in kernel.iter().enumerate() {
                let rr = (r as isize + i as isize - 5).clamp(0, lh as isize - 1) as usize;
                output[r * lw + c] += v * tmp[rr * lw + c];
            }
        }
    }
    output
}
pub fn climate_baseline(
    t: &[f32],
    e: &[f32],
    h: usize,
    w: usize,
    win: usize,
) -> (Vec<f32>, Vec<f32>) {
    let oh = h - win + 1;
    let ow = w - win + 1;
    let mut base = vec![0.0; oh * ow];
    let mut beta = base.clone();
    for r in 0..oh {
        for c in 0..ow {
            let mut st = 0.0f64;
            let mut se = 0.0f64;
            let mut se2 = 0.0f64;
            let mut set = 0.0f64;
            let mut count = 0.0f64;
            for dr in 0..win {
                for dc in 0..win {
                    let ix = (r + dr) * w + c + dc;
                    if e[ix] > 0.0 {
                        let el = e[ix] as f64;
                        let te = t[ix] as f64;
                        st += te;
                        se += el;
                        se2 += el * el;
                        set += el * te;
                        count += 1.0;
                    }
                }
            }
            let den = count + 1e-6;
            let mt = st / den;
            let me = se / den;
            let me2 = se2 / den;
            let met = set / den;
            let variance = me2 - me * me;
            let b = if variance < 1.0 || count < 0.02 * (win * win) as f64 {
                -0.0065
            } else {
                (met - me * mt) / (variance + 1e-6)
            };
            let b = b.clamp(-0.012, 0.0) as f32;
            let idx = (r + win / 2) * w + (c + win / 2);
            base[r * ow + c] = t[idx] - b * e[idx];
            beta[r * ow + c] = b;
        }
    }
    (base, beta)
}
pub fn sample_bilinear(src: &[f32], h: usize, w: usize, y: f32, x: f32) -> f32 {
    let y = y.clamp(0.0, (h - 1) as f32);
    let x = x.clamp(0.0, (w - 1) as f32);
    let r = y as usize;
    let c = x as usize;
    let nr = (r + 1).min(h - 1);
    let nc = (c + 1).min(w - 1);
    let wy = y - r as f32;
    let wx = x - c as f32;
    (1.0 - wy) * (1.0 - wx) * src[r * w + c]
        + (1.0 - wy) * wx * src[r * w + nc]
        + wy * (1.0 - wx) * src[nr * w + c]
        + wy * wx * src[nr * w + nc]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_resize() {
        assert!(bilinear(&vec![7.0; 16], 4, 4, 9, 11)
            .iter()
            .all(|&v| (v - 7.0).abs() < 1e-5));
    }
    #[test]
    fn weights_positive_symmetric() {
        let w = weights(64);
        assert!(w.iter().all(|&v| v > 0.0));
        assert_eq!(w[0], w[4095]);
    }
}
