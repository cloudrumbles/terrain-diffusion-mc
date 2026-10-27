//! Standalone three-stage Terrain Diffusion pipeline, with lazy overlapping tile cache.
//! Coordinates are (row, column) in the model's native pixel grid, not Minecraft blocks.
use crate::{
    config::ModelConfig,
    inference::{Runner, Stage},
    math, rng,
    scheduler::{Edm, SD, SIGMA_MAX},
    synthetic::Synthetic,
};
use anyhow::{ensure, Result};
use std::collections::HashMap;
use std::path::Path;

const COND_MEANS: [f32; 7] = [14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66];
const COND_STDS: [f32; 7] = [21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    Coarse,
    LatentInit,
    Latent,
    Decoder,
}
impl Layer {
    fn parameters(self) -> (usize, i32, usize) {
        match self {
            Self::Coarse => (64, 48, 7),
            Self::LatentInit | Self::Latent => (64, 32, 6),
            Self::Decoder => (256, 192, 2),
        }
    }
}
#[derive(Debug)]
pub struct TerrainOutput {
    pub height: usize,
    pub width: usize,
    /// Height in meters, pixel-major (row, then column).
    pub elevation: Vec<f32>,
    /// Optional five channel-major fields: temperature, precip, temp_std, precip_std, lapse.
    pub climate: Option<Vec<f32>>,
}
pub struct Terrain<R: Runner> {
    pub seed: u64,
    pub config: ModelConfig,
    synthetic: Synthetic,
    runner: R,
    coarse: HashMap<(i32, i32), Vec<f32>>,
    init: HashMap<(i32, i32), Vec<f32>>,
    latent: HashMap<(i32, i32), Vec<f32>>,
    decoder: HashMap<(i32, i32), Vec<f32>>,
}
impl<R: Runner> Terrain<R> {
    pub fn new(seed: u64, asset_dir: &Path, runner: R) -> Result<Self> {
        let config = ModelConfig::load(asset_dir)?;
        let synthetic = Synthetic::new(seed, &config, asset_dir)?;
        Ok(Self {
            seed,
            config,
            synthetic,
            runner,
            coarse: HashMap::new(),
            init: HashMap::new(),
            latent: HashMap::new(),
            decoder: HashMap::new(),
        })
    }
    pub fn set_seed(&mut self, seed: u64, dir: &Path) -> Result<()> {
        if self.seed != seed {
            self.synthetic = Synthetic::new(seed, &self.config, dir)?;
            self.seed = seed;
            self.coarse.clear();
            self.init.clear();
            self.latent.clear();
            self.decoder.clear();
        }
        Ok(())
    }
    pub fn tile_count(&self) -> usize {
        self.coarse.len() + self.init.len() + self.latent.len() + self.decoder.len()
    }
    /// Accumulates all intersecting weighted tiles; final channel is the sum of weights.
    fn slice(&mut self, layer: Layer, y0: i32, x0: i32, h: usize, w: usize) -> Result<Vec<f32>> {
        ensure!(
            h > 0 && w > 0 && h <= 8192 && w <= 8192,
            "invalid tensor slice dimensions"
        );
        let (size, stride, ch) = layer.parameters();
        let stride = stride as i32;
        let size = size as i32;
        let ty0 = -((-(y0 - size + 1)).div_euclid(stride));
        let tx0 = -((-(x0 - size + 1)).div_euclid(stride));
        let ty1 = (y0 + h as i32 - 1).div_euclid(stride);
        let tx1 = (x0 + w as i32 - 1).div_euclid(stride);
        let mut out = vec![0.0; ch * h * w];
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let tile = self.tile(layer, ty, tx)?;
                let sy = ty * stride;
                let sx = tx * stride;
                for y in y0.max(sy)..(y0 + h as i32).min(sy + size) {
                    for x in x0.max(sx)..(x0 + w as i32).min(sx + size) {
                        let dst = (y - y0) as usize * w + (x - x0) as usize;
                        let src = (y - sy) as usize * size as usize + (x - sx) as usize;
                        for c in 0..ch {
                            out[c * h * w + dst] += tile[c * size as usize * size as usize + src];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
    fn tile(&mut self, layer: Layer, ty: i32, tx: i32) -> Result<Vec<f32>> {
        let key = (ty, tx);
        let hit = match layer {
            Layer::Coarse => self.coarse.get(&key),
            Layer::LatentInit => self.init.get(&key),
            Layer::Latent => self.latent.get(&key),
            Layer::Decoder => self.decoder.get(&key),
        };
        if let Some(t) = hit {
            return Ok(t.clone());
        }
        let t = match layer {
            Layer::Coarse => self.coarse_tile(ty, tx)?,
            Layer::LatentInit => self.latent_tile(ty, tx, false)?,
            Layer::Latent => self.latent_tile(ty, tx, true)?,
            Layer::Decoder => self.decoder_tile(ty, tx)?,
        };
        match layer {
            Layer::Coarse => &mut self.coarse,
            Layer::LatentInit => &mut self.init,
            Layer::Latent => &mut self.latent,
            Layer::Decoder => &mut self.decoder,
        }
        .insert(key, t.clone());
        Ok(t)
    }
    fn coarse_tile(&mut self, ty: i32, tx: i32) -> Result<Vec<f32>> {
        const N: usize = 64;
        const P: usize = 64 * 64;
        let y = ty * 48;
        let x = tx * 48;
        let mut syn = self.synthetic.sample(x, y, N, N);
        for v in &mut syn[P..2 * P] {
            if *v <= 20.0 {
                *v = (*v - 20.0) * 1.25 + 20.0;
            }
        }
        let idx = [0, 2, 3, 4, 5];
        let noise = rng::gaussian(self.seed, y, x, N, N, 5, N);
        let mut mixed = vec![0.0; 5 * P];
        for c in 0..5 {
            let angle = self.config.cond_snr[c].atan();
            for k in 0..P {
                let value = (syn[c * P + k] - self.config.coarse_means[idx[c]])
                    / self.config.coarse_stds[idx[c]];
                mixed[c * P + k] = angle.cos() * value + angle.sin() * noise[c * P + k];
            }
        }
        let mut sched = Edm::new(20);
        let mut sample = rng::gaussian(self.seed.wrapping_add(1), y, x, N, N, 6, N);
        for v in &mut sample {
            *v *= sched.sigmas[0];
        }
        let cond: Vec<_> = self
            .config
            .cond_snr
            .iter()
            .map(|&snr| (vec![(snr / 8.0).ln()], vec![1]))
            .collect();
        for step in 0..20 {
            let sigma = sched.sigmas[step];
            let mut input = Edm::scale(&sample, sigma);
            input.extend_from_slice(&mixed);
            let predicted = self.runner.infer(
                Stage::Coarse,
                input,
                &[1, 11, N, N],
                vec![(sigma / SD).atan()],
                cond.clone(),
            )?;
            ensure!(
                predicted.len() == 6 * P,
                "coarse model returned wrong tensor size"
            );
            sample = sched.step(&predicted, &sample);
        }
        for c in 0..6 {
            for k in 0..P {
                sample[c * P + k] = (sample[c * P + k] / SD) * self.config.coarse_stds[c]
                    + self.config.coarse_means[c];
            }
        }
        for k in 0..P {
            sample[P + k] = sample[k] - sample[P + k];
        }
        let weights = math::weights(N);
        let mut result = vec![0.0; 7 * P];
        for c in 0..6 {
            for k in 0..P {
                result[c * P + k] = sample[c * P + k] * weights[k];
            }
        }
        result[6 * P..].copy_from_slice(&weights);
        Ok(result)
    }
    fn latent_conditioning(&self, coarse: &[f32]) -> [f32; 58] {
        let mut vals = [[0.0; 16]; 7];
        for c in 0..6 {
            for k in 0..16 {
                let w = coarse[6 * 16 + k];
                let raw = if w > 1e-6 {
                    coarse[c * 16 + k] / w
                } else {
                    0.0
                };
                vals[c][k] = (raw - COND_MEANS[c]) / COND_STDS[c];
            }
        }
        for k in 0..16 {
            vals[6][k] = (1.0 - COND_MEANS[6]) / COND_STDS[6];
        }
        let mut climates = [0.0; 4];
        for c in 0..4 {
            climates[c] = [5, 6, 9, 10].iter().map(|&k| vals[c + 2][k]).sum::<f32>() / 4.0;
        }
        let hist = self.config.histogram_raw.unwrap_or([0.0; 5]);
        let groups: [&[f32]; 6] = [
            &vals[0],
            &vals[1],
            &climates,
            &vals[6],
            &hist,
            &[-0.5 * (12.0f32).sqrt()],
        ];
        let sizes = [16.0f32, 16.0, 4.0, 16.0, 5.0, 1.0];
        let factor = (58.0f32 * 6.0).sqrt() / 6.0;
        let mut output = [0.0; 58];
        let mut offset = 0;
        for (i, group) in groups.into_iter().enumerate() {
            for &val in group {
                output[offset] = val * factor / sizes[i].sqrt();
                offset += 1;
            }
        }
        output
    }
    fn latent_tile(&mut self, ty: i32, tx: i32, final_step: bool) -> Result<Vec<f32>> {
        const N: usize = 64;
        const P: usize = 64 * 64;
        let y = ty * 32;
        let x = tx * 32;
        // The coarse dependency window is (ty-1,tx-1) with shape (4,4).
        let coarse = self.slice(Layer::Coarse, ty - 1, tx - 1, 4, 4)?;
        let cond = self.latent_conditioning(&coarse);
        let (t, seed_add) = if final_step {
            ((0.35 / SD).atan(), 5820u64)
        } else {
            ((SIGMA_MAX / SD).atan(), 5819u64)
        };
        let prev = if final_step {
            Some(self.slice(Layer::LatentInit, y, x, N, N)?)
        } else {
            None
        };
        let mut input = vec![0.0; 5 * P];
        let noise = rng::gaussian(self.seed.wrapping_add(seed_add), y, x, N, N, 5, N);
        let mut xt = vec![0.0; 5 * P];
        for c in 0..5 {
            for k in 0..P {
                let sample = if let Some(prev) = &prev {
                    let w = prev[5 * P + k];
                    if w > 1e-6 {
                        prev[c * P + k] / w * SD
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                xt[c * P + k] = t.cos() * sample + t.sin() * noise[c * P + k] * SD;
                input[c * P + k] = xt[c * P + k] / SD;
            }
        }
        let pred = self.runner.infer(
            Stage::Base,
            input,
            &[1, 5, N, N],
            vec![t],
            vec![(cond.to_vec(), vec![1, 58])],
        )?;
        ensure!(pred.len() == 5 * P, "base model returned wrong tensor size");
        let ww = math::weights(N);
        let mut out = vec![0.0; 6 * P];
        for c in 0..5 {
            for k in 0..P {
                out[c * P + k] =
                    ((t.cos() * xt[c * P + k] + t.sin() * SD * pred[c * P + k]) / SD) * ww[k];
            }
        }
        out[5 * P..].copy_from_slice(&ww);
        Ok(out)
    }
    fn decoder_tile(&mut self, ty: i32, tx: i32) -> Result<Vec<f32>> {
        const N: usize = 256;
        const P: usize = 256 * 256;
        let lc = self.config.latent_compression;
        let s = N / lc;
        let lp = s * s;
        let y = ty * 192;
        let x = tx * 192;
        let lat = self.slice(Layer::Latent, y / lc as i32, x / lc as i32, s, s)?;
        let mut input = vec![0.0; 5 * P];
        for c in 0..4 {
            for r in 0..N {
                for col in 0..N {
                    let ly = r / lc;
                    let lx = col / lc;
                    let idx = ly * s + lx;
                    let w = lat[5 * lp + idx];
                    input[(c + 1) * P + r * N + col] =
                        if w > 1e-6 { lat[c * lp + idx] / w } else { 0.0 };
                }
            }
        }
        let t = (SIGMA_MAX / SD).atan();
        let noise = rng::gaussian(self.seed.wrapping_add(5819), y, x, N, N, 1, N);
        let mut xt = vec![0.0; P];
        for k in 0..P {
            xt[k] = t.sin() * noise[k] * SD;
            input[k] = xt[k] / SD;
        }
        let pred = self
            .runner
            .infer(Stage::Decoder, input, &[1, 5, N, N], vec![t], vec![])?;
        ensure!(pred.len() == P, "decoder model returned wrong tensor size");
        let ww = math::weights(N);
        let mut out = vec![0.0; 2 * P];
        for k in 0..P {
            out[k] = ((t.cos() * xt[k] + t.sin() * SD * pred[k]) / SD) * ww[k];
        }
        out[P..].copy_from_slice(&ww);
        Ok(out)
    }
    pub fn get_coarse_slice(&mut self, y0: i32, x0: i32, h: usize, w: usize) -> Result<Vec<f32>> {
        self.slice(Layer::Coarse, y0, x0, h, w)
    }
    pub fn generate(
        &mut self,
        y: i32,
        x: i32,
        h: usize,
        w: usize,
        with_climate: bool,
    ) -> Result<TerrainOutput> {
        ensure!(
            h > 0 && w > 0 && h <= 4096 && w <= 4096,
            "invalid output bounds"
        );
        let lc = self.config.latent_compression as i32;
        let padding = 6 * lc;
        let y0 = (y - padding).div_euclid(lc) * lc;
        let x0 = (x - padding).div_euclid(lc) * lc;
        let y1 = -((-(y + h as i32 + padding)).div_euclid(lc)) * lc;
        let x1 = -((-(x + w as i32 + padding)).div_euclid(lc)) * lc;
        let ph = (y1 - y0) as usize;
        let pw = (x1 - x0) as usize;
        let residual = self.slice(Layer::Decoder, y0, x0, ph, pw)?;
        let mut res = vec![0.0; ph * pw];
        for k in 0..ph * pw {
            let weight = residual[ph * pw + k];
            let v = if weight > 1e-6 {
                residual[k] / weight
            } else {
                0.0
            };
            res[k] = v * self.config.residual_std + self.config.residual_mean;
        }
        let lh = ph / lc as usize;
        let lw = pw / lc as usize;
        let latent = self.slice(Layer::Latent, y0 / lc, x0 / lc, lh, lw)?;
        let mut low = vec![0.0; lh * lw];
        for k in 0..lh * lw {
            let weight = latent[5 * lh * lw + k];
            let v = if weight > 1e-6 {
                latent[4 * lh * lw + k] / weight
            } else {
                0.0
            };
            low[k] = v * 38.6 - 31.4;
        }
        let denoised = math::denoise(&res, &low, ph, pw, lh, lw, 5.0);
        let elevated_low = math::bilinear(&denoised, lh, lw, ph, pw);
        let mut elevation = vec![0.0; h * w];
        for r in 0..h {
            for c in 0..w {
                let ix = (y - y0) as usize + r;
                let jx = (x - x0) as usize + c;
                let v = res[ix * pw + jx] + elevated_low[ix * pw + jx];
                elevation[r * w + c] = v.signum() * v * v;
            }
        }
        let climate = if with_climate {
            Some(self.get_climate(y, x, h, w, &elevation)?)
        } else {
            None
        };
        Ok(TerrainOutput {
            height: h,
            width: w,
            elevation,
            climate,
        })
    }
    fn get_climate(
        &mut self,
        y: i32,
        x: i32,
        h: usize,
        w: usize,
        elev: &[f32],
    ) -> Result<Vec<f32>> {
        let lc = self.config.latent_compression as i32;
        let stride = 32 * lc;
        let cy0 = y.div_euclid(stride);
        let cx0 = x.div_euclid(stride);
        let cy1 = -((-(y + h as i32)).div_euclid(stride));
        let cx1 = -((-(x + w as i32)).div_euclid(stride));
        let pad = 8i32;
        let ch = (cy1 - cy0 + 2 * pad) as usize;
        let cw = (cx1 - cx0 + 2 * pad) as usize;
        let coarse = self.slice(Layer::Coarse, cy0 - pad, cx0 - pad, ch, cw)?;
        let mut channels = vec![0.0; 6 * ch * cw];
        for c in 0..6 {
            for k in 0..ch * cw {
                let weight = coarse[6 * ch * cw + k];
                channels[c * ch * cw + k] = if weight > 1e-6 {
                    coarse[c * ch * cw + k] / weight
                } else {
                    0.0
                };
            }
        }
        let coarse_elev: Vec<f32> = channels[0..ch * cw]
            .iter()
            .map(|&v| v.max(0.0).powi(2))
            .collect();
        let (baseline, beta) = math::climate_baseline(
            &channels[2 * ch * cw..3 * ch * cw],
            &coarse_elev,
            ch,
            cw,
            15,
        );
        let bh = ch - 14;
        let bw = cw - 14;
        let mut subs = [vec![0.0; bh * bw], vec![0.0; bh * bw], vec![0.0; bh * bw]];
        for cc in 0..3 {
            for sr in 0..bh {
                for sc in 0..bw {
                    subs[cc][sr * bw + sc] = channels[(3 + cc) * ch * cw + (sr + 7) * cw + sc + 7];
                }
            }
        }
        let mut out = vec![0.0; 5 * h * w];
        for r in 0..h {
            for c in 0..w {
                let gy = (y as f32 + r as f32 + 0.5) / stride as f32 - cy0 as f32 + 0.5;
                let gx = (x as f32 + c as f32 + 0.5) / stride as f32 - cx0 as f32 + 0.5;
                let b = math::sample_bilinear(&beta, bh, bw, gy, gx);
                let tb = math::sample_bilinear(&baseline, bh, bw, gy, gx);
                let k = r * w + c;
                out[k] = tb + b * elev[k].max(0.0);
                for cc in 0..3 {
                    out[(cc + 1) * h * w + k] = math::sample_bilinear(&subs[cc], bh, bw, gy, gx);
                }
                out[4 * h * w + k] = b;
            }
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn negative_window_boundaries() {
        // Window stride=48, size=64. A slice [-2,2) crosses tiles -1 and 0.
        assert_eq!(-((65i32).div_euclid(48)), -1);
        assert_eq!((1i32).div_euclid(48), 0);
    }
}
