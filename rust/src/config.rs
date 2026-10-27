use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};
#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub native_resolution: f32,
    pub latent_compression: usize,
    pub coarse_means: [f32; 6],
    pub coarse_stds: [f32; 6],
    pub cond_snr: [f32; 5],
    pub frequency_mult: [f32; 5],
    pub residual_mean: f32,
    pub residual_std: f32,
    pub coarse_pooling: usize,
    pub histogram_raw: Option<[f32; 5]>,
}
impl ModelConfig {
    pub fn load(folder: &Path) -> Result<Self> {
        let path = folder.join("world_pipeline_config.json");
        let cfg: Self = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )?;
        ensure!(
            cfg.coarse_pooling == 1,
            "only coarse_pooling=1 is supported (same as Java)"
        );
        ensure!(
            cfg.native_resolution > 0.0,
            "native resolution must be positive"
        );
        Ok(cfg)
    }
}
