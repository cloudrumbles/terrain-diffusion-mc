# Terrain Diffusion (standalone Rust)

Rust port of the world-generation pipeline in the parent project's Java/Fabric implementation. No JVM, Minecraft server, Fabric loader, biome registry or game assets are required.

Includes:
- Seeded FastNoiseLite Perlin/fBm synthetic conditioning, WorldClim quantile transforms, temperature/precipitation corrections
- Portable PCG Gaussian tiled noise, stable across overlapping and negative-coordinate requests
- 20-step EDM/DPM-Solver++ Karras coarse diffusion (64x64 at stride 48), two-step flow-matching latent model (64x64 at stride 32), single-step decoder (256x256 at stride 192)
- Overlapping weighted tile accumulation with lazy per-stage caching
- Laplacian elevation reconstruction and 5-channel climate computation
- ONNX Runtime inference with the original ONNX models, exported raw float32 grids

## Build

Use Rust 1.92+:

    cargo test
    cargo build --release

For CUDA-capable ONNX Runtime builds, use `cargo build --release --features cuda` and supply a CUDA-enabled ONNX Runtime shared library.

The Rust binding dynamically loads ONNX Runtime. Set the runtime shared library path before executing:

    export ORT_DYLIB_PATH=/path/to/libonnxruntime.so

Use a matching ONNX Runtime shared library (ONNX Runtime 1.23.x is expected by the pinned ort 2.0.0-rc.15 binding). The Rust crate is *not* the ONNX runtime itself.

## Model assets

Supply the same five model files used by the original project, under one directory:

    models/
      coarse_model.onnx
      base_model.onnx
      decoder_model.onnx
      pipeline_data.json
      world_pipeline_config.json

Source: https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/tree/ad2df557eca5645f588766101cf3bc3682455c3e

These are ~2.5GB total. Download only from the pinned revision to preserve compatibility with the model constants.

## Generate a grid

    cargo run --release -- ./models 123456789 -32 -32 64 64 --climate terrain

This writes \`terrain.elevation.f32\` (64×64, meters, f32 LE row-major), and \`terrain.climate.f32\` (5×64×64, channel-first f32 LE). Channels are temperature, precipitation, temperature std, precipitation std, and lapse rate. Coordinates are native model grid pixels (nominal 30 m/pixel; see model config), *not* block positions.

To integrate with a game, depend on this crate and use:

    use terrain_diffusion::{Terrain, inference::OnnxRunner};
    let models = std::path::Path::new("./models");
    let mut terrain = Terrain::new(42, models, OnnxRunner::new(models)?)?;
    let patch = terrain.generate(0, 0, 32, 32, true)?;

## Status

The source algorithms and ONNX input layout have been ported. Unit tests check the portable noise, solver, interpolation, tile boundaries and blend weights. Full numerical parity with the Java/Python implementation and runtime performance have **not** been measured with the multi-GB models. The current cache is unbounded and the latent inference stage runs with batch size 1 (the Java implementation batches up to 4).
