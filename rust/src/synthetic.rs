//! Seeded, quantile-matched WorldClim synthetic conditioning (5 channels).
use crate::config::ModelConfig;
use anyhow::{ensure, Context, Result};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use serde::Deserialize;
use std::{fs, path::Path};
#[derive(Debug, Deserialize)]
struct PipelineData {
    n_quantiles: usize,
    data_quantile_tables: Vec<Vec<f32>>,
    a_temp_std: f32,
    b_temp_std: f32,
    temp_std_p1: f32,
    temp_std_p99: f32,
}
pub struct Synthetic {
    noise: Vec<FastNoiseLite>,
    nq: Vec<Vec<f32>>,
    dq: Vec<Vec<f32>>,
    data: PipelineData,
}
fn interp(x: f32, xp: &[f32], yp: &[f32]) -> f32 {
    if x <= xp[0] {
        return yp[0];
    }
    if x >= xp[xp.len() - 1] {
        return yp[yp.len() - 1];
    }
    let mut lo = 0;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (hi + lo) / 2;
        if xp[mid] <= x {
            lo = mid
        } else {
            hi = mid
        }
    }
    yp[lo] + (x - xp[lo]) / (xp[hi] - xp[lo]) * (yp[hi] - yp[lo])
}
impl Synthetic {
    pub fn new(seed: u64, cfg: &ModelConfig, dir: &Path) -> Result<Self> {
        let path = dir.join("pipeline_data.json");
        let data: PipelineData = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )?;
        ensure!(
            data.data_quantile_tables.len() == 5 && data.n_quantiles >= 2,
            "invalid quantile tables"
        );
        ensure!(
            data.data_quantile_tables
                .iter()
                .all(|a| a.len() == data.n_quantiles),
            "invalid quantile lengths"
        );
        let mut noise = Vec::new();
        let mut nq = Vec::new();
        for ch in 0..5 {
            let mut f =
                FastNoiseLite::with_seed((seed.wrapping_add(ch as u64 + 1) & 0x7fff_ffff) as i32);
            f.set_noise_type(Some(NoiseType::Perlin));
            f.set_frequency(Some(0.05 * cfg.frequency_mult[ch]));
            f.set_fractal_type(Some(FractalType::FBm));
            f.set_fractal_octaves(Some([4, 2, 4, 4, 4][ch]));
            f.set_fractal_lacunarity(Some(2.0));
            f.set_fractal_gain(Some(0.5));
            let mut values = Vec::with_capacity(1024 * 1024);
            for r in 0..1024 {
                for c in 0..1024 {
                    values.push(f.get_noise_2d((c * 32) as f32, (r * 32) as f32));
                }
            }
            values.sort_unstable_by(f32::total_cmp);
            let count = values.len();
            let mut q = Vec::with_capacity(64);
            for i in 0..64 {
                let p = 1e-4_f32 + i as f32 * (1.0 - 2e-4) / 63.0;
                let idx = p * (count - 1) as f32;
                let lo = idx as usize;
                let hi = (lo + 1).min(count - 1);
                q.push(values[lo] + (idx - lo as f32) * (values[hi] - values[lo]));
            }
            let mindiff = q
                .windows(2)
                .filter_map(|v| if v[1] > v[0] { Some(v[1] - v[0]) } else { None })
                .fold(f32::INFINITY, f32::min);
            let mindiff = if mindiff.is_finite() { mindiff } else { 1e-10 };
            for i in 1..q.len() {
                if q[i] <= q[i - 1] {
                    q[i] = q[i - 1] + mindiff * 0.1;
                }
            }
            noise.push(f);
            nq.push(q);
        }
        Ok(Self {
            noise,
            nq,
            dq: data.data_quantile_tables.clone(),
            data,
        })
    }
    pub fn sample(&self, x: i32, y: i32, h: usize, w: usize) -> Vec<f32> {
        let n = h * w;
        let mut out = vec![0.0; 5 * n];
        for ch in 0..5 {
            for r in 0..h {
                for c in 0..w {
                    let v =
                        self.noise[ch].get_noise_2d((x + c as i32) as f32, (y + r as i32) as f32);
                    out[ch * n + r * w + c] = interp(v, &self.nq[ch], &self.dq[ch]);
                }
            }
        }
        for k in 0..n {
            let elev = out[k];
            let mut temp = out[n + k];
            let mut ts = out[2 * n + k];
            let p = out[3 * n + k];
            let mut ps = out[4 * n + k];
            let lapse = (-6.5 + 0.0015 * p).clamp(-9.8, -4.0) / 1000.0;
            temp = (temp + lapse * elev.max(0.0)).clamp(-10.0, 40.0);
            let baseline = self.data.a_temp_std * temp + self.data.b_temp_std;
            let t01 =
                (ts - self.data.temp_std_p1) / (self.data.temp_std_p99 - self.data.temp_std_p1);
            let clipped = self.data.temp_std_p1.max(-baseline);
            ts = (t01 * (self.data.temp_std_p99 - clipped) + clipped + baseline).max(20.0);
            ps *= ((185.0 - 0.04111 * p) / 185.0).max(0.0);
            out[k] = elev.signum() * elev.abs().sqrt();
            out[n + k] = temp;
            out[2 * n + k] = ts;
            out[4 * n + k] = ps;
        }
        out
    }
}
