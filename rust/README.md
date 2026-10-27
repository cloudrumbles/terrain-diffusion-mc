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

Checked on 2026-10-27 against the Java pipeline (`WorldPipeline.get`, the same ONNX files at revision ad2df557, ONNX Runtime 1.31.1), seed 123456789, 64x64 native pixels:

| Region (row, col) | Terrain | Max elevation difference | Max climate difference |
| --- | --- | ---: | ---: |
| -32, -32 | ocean, -787 to -330 m | 0.0007 m | 0.00024 |
| -29755, -8080 | mountains, 2,660 to 3,411 m | 0.0017 m (CUDA: 0.0027 m) | 0.00024 |

On a GTX 1650 Ti with `--features cuda`, a 1024x1024 region took 27.3 s including about 8 s of model loading, about 55,000 native pixels a second, at a peak of 5.7 GB RSS. The cache is unbounded and the latent stage runs with batch size 1 (the Java implementation batches up to 4).

With the `cuda` feature the models must run on CUDA: a session fails instead of falling back to the CPU. ONNX Runtime's CUDA provider needs the CUDA 13 and cuDNN 9 libraries on `LD_LIBRARY_PATH`.
