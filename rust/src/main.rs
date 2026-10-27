use anyhow::{ensure, Context, Result};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
use terrain_diffusion::{inference::OnnxRunner, Terrain};
fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 7 {
        eprintln!("usage: terrain-diffusion <model-dir> <seed> <row> <col> <height> <width> [--climate] [output-prefix]");
        eprintln!("writes <prefix>.elevation.f32 (raw little-endian f32 row-major), and optionally <prefix>.climate.f32 (5-channel float32)");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let seed = args[2].parse::<u64>().context("seed must be a u64")?;
    let y = args[3].parse::<i32>()?;
    let x = args[4].parse::<i32>()?;
    let h = args[5].parse::<usize>()?;
    let w = args[6].parse::<usize>()?;
    let climate = args.iter().any(|s| s == "--climate");
    let prefix = args
        .iter()
        .skip(7)
        .find(|s| s.as_str() != "--climate")
        .map(String::as_str)
        .unwrap_or("terrain");
    ensure!(
        h > 0 && w > 0 && h <= 4096 && w <= 4096,
        "dimensions must be in 1..=4096"
    );
    let runner = OnnxRunner::new(&dir)?;
    let mut world = Terrain::new(seed, &dir, runner)?;
    let result = world.generate(y, x, h, w, climate)?;
    save(&format!("{prefix}.elevation.f32"), &result.elevation)?;
    if let Some(ref c) = result.climate {
        save(&format!("{prefix}.climate.f32"), c)?;
    }
    println!(
        "generated {}x{} native pixels; {} cached inference tiles; native resolution {}m/pixel",
        h,
        w,
        world.tile_count(),
        world.config.native_resolution
    );
    Ok(())
}
fn save(path: &str, values: &[f32]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path).with_context(|| format!("creating {path}"))?);
    for v in values {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()?;
    println!("wrote {path}");
    Ok(())
}
