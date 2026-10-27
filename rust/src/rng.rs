//! Exact wrapping PCG32 and Marsaglia polar stream used by the Java/Python code.
const MULT: u64 = 6364136223846793005;
const INC: u64 = 1442695040888963407;
pub fn tile_seed(base: u64, ty: i32, tx: i32) -> u64 {
    base.wrapping_mul(0x9e37_79b9)
        .wrapping_add(ty as u32 as u64)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(tx as u32 as u64)
}
pub fn pcg_next(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(MULT).wrapping_add(INC);
    let x = (((*state >> 18) ^ *state) >> 27) as u32;
    x.rotate_right((*state >> 59) as u32)
}
pub fn normal(seed: u64, out: &mut [f32]) {
    let mut state = seed;
    let mut i = 0;
    while i < out.len() {
        let v1 = 2.0 * (pcg_next(&mut state) as f64 + 1.0) / 4294967296.0 - 1.0;
        let v2 = 2.0 * (pcg_next(&mut state) as f64 + 1.0) / 4294967296.0 - 1.0;
        let s = v1 * v1 + v2 * v2;
        if s > 0.0 && s < 1.0 {
            let factor = (-2.0 * s.ln() / s).sqrt();
            out[i] = (v1 * factor) as f32;
            i += 1;
            if i < out.len() {
                out[i] = (v2 * factor) as f32;
                i += 1;
            }
        }
    }
}
/// Channel-first, row-major Gaussian noise. Pixel values remain invariant for
/// overlapping requests when tile dimensions and seed match.
pub fn gaussian(
    seed: u64,
    y0: i32,
    x0: i32,
    h: usize,
    w: usize,
    ch: usize,
    tile: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; ch * h * w];
    let ty0 = y0.div_euclid(tile as i32);
    let ty1 = (y0 + h as i32 - 1).div_euclid(tile as i32);
    let tx0 = x0.div_euclid(tile as i32);
    let tx1 = (x0 + w as i32 - 1).div_euclid(tile as i32);
    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            let mut values = vec![0.0; ch * tile * tile];
            normal(tile_seed(seed, ty, tx), &mut values);
            let sy = ty * tile as i32;
            let sx = tx * tile as i32;
            let y_start = y0.max(sy);
            let y_end = (y0 + h as i32).min(sy + tile as i32);
            let x_start = x0.max(sx);
            let x_end = (x0 + w as i32).min(sx + tile as i32);
            for c in 0..ch {
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        out[c * h * w + (y - y0) as usize * w + (x - x0) as usize] =
                            values[c * tile * tile + (y - sy) as usize * tile + (x - sx) as usize];
                    }
                }
            }
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlap_and_negative_coords() {
        let a = gaussian(123, -7, -13, 37, 32, 2, 64);
        let b = gaussian(123, -9, -14, 40, 40, 2, 64);
        for c in 0..2 {
            for y in 0..37 {
                for x in 0..32 {
                    assert_eq!(
                        a[c * 37 * 32 + y * 32 + x],
                        b[c * 40 * 40 + (y + 2) * 40 + x + 1]
                    );
                }
            }
        }
    }
    #[test]
    fn pcg_known() {
        let mut s = 0u64;
        assert_eq!(pcg_next(&mut s), 0x602bf3fd);
    }
}
