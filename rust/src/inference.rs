//! ONNX model execution. No Minecraft or JVM dependencies.
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Coarse,
    Base,
    Decoder,
}
pub trait Runner {
    /// Input/output tensors are channel-major f32.
    fn infer(
        &mut self,
        stage: Stage,
        x: Vec<f32>,
        shape: &[usize],
        noise: Vec<f32>,
        cond: Vec<(Vec<f32>, Vec<usize>)>,
    ) -> Result<Vec<f32>>;
}
#[cfg(feature = "onnx")]
pub struct OnnxRunner {
    models: [Option<ort::session::Session>; 3],
    dir: PathBuf,
    env: ort::environment::Environment,
}
#[cfg(feature = "onnx")]
impl OnnxRunner {
    pub fn new(dir: &Path) -> Result<Self> {
        for f in ["coarse_model.onnx", "base_model.onnx", "decoder_model.onnx"] {
            ensure!(
                dir.join(f).is_file(),
                "missing model {}, download the pinned model assets first",
                dir.join(f).display()
            );
        }
        let env = ort::init()
            .build()
            .context("initializing ONNX Runtime; set ORT_DYLIB_PATH to libonnxruntime.so")?;
        Ok(Self {
            models: [None, None, None],
            dir: dir.to_path_buf(),
            env,
        })
    }
}
#[cfg(feature = "onnx")]
impl Runner for OnnxRunner {
    fn infer(
        &mut self,
        stage: Stage,
        x: Vec<f32>,
        shape: &[usize],
        noise: Vec<f32>,
        cond: Vec<(Vec<f32>, Vec<usize>)>,
    ) -> Result<Vec<f32>> {
        use ort::{session::Session, value::Tensor};
        let ix = match stage {
            Stage::Coarse => 0,
            Stage::Base => 1,
            Stage::Decoder => 2,
        };
        if self.models[ix].is_none() {
            let path = self
                .dir
                .join(["coarse_model.onnx", "base_model.onnx", "decoder_model.onnx"][ix]);
            let session = Session::builder(&self.env)?
                .commit_from_file(&path)
                .with_context(|| format!("loading {}", path.display()))?;
            self.models[ix] = Some(session);
        }
        let mut inputs = ort::inputs![
            "x"=>Tensor::from_array((shape.to_vec(),x))?,
            "noise_labels"=>Tensor::from_array(([noise.len()],noise))?,
        ];
        for (i, (data, dims)) in cond.into_iter().enumerate() {
            inputs.push((
                format!("cond_{i}").into(),
                Tensor::from_array((dims, data))?.into(),
            ));
        }
        let results = self.models[ix].as_mut().unwrap().run(inputs)?;
        let (_, out) = results[0].try_extract_tensor::<f32>()?;
        Ok(out.to_vec())
    }
}
pub struct ZeroRunner;
impl Runner for ZeroRunner {
    fn infer(
        &mut self,
        stage: Stage,
        _x: Vec<f32>,
        shape: &[usize],
        _noise: Vec<f32>,
        _cond: Vec<(Vec<f32>, Vec<usize>)>,
    ) -> Result<Vec<f32>> {
        let b = shape[0];
        let s = shape[2];
        let ch = if stage == Stage::Coarse {
            6
        } else if stage == Stage::Base {
            5
        } else {
            1
        };
        Ok(vec![0.0; b * ch * s * s])
    }
}
