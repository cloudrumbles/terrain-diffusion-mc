//! EDM/Karras 20-step DPM-Solver++ midpoint scheduler.
pub const SD: f32 = 0.5;
pub const SIGMA_MAX: f32 = 80.0;
pub struct Edm {
    pub sigmas: Vec<f32>,
    index: usize,
    previous: Option<Vec<f32>>,
}
impl Edm {
    pub fn new(steps: usize) -> Self {
        assert!(steps >= 2);
        let min_inv = 0.002_f32.powf(1.0 / 7.0);
        let max_inv = SIGMA_MAX.powf(1.0 / 7.0);
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| (max_inv + i as f32 / (steps - 1) as f32 * (min_inv - max_inv)).powf(7.0))
            .collect();
        sigmas.push(0.0);
        Self {
            sigmas,
            index: 0,
            previous: None,
        }
    }
    pub fn scale(sample: &[f32], sigma: f32) -> Vec<f32> {
        let k = 1.0 / (sigma * sigma + SD * SD).sqrt();
        sample.iter().map(|&v| v * k).collect()
    }
    pub fn step(&mut self, output: &[f32], sample: &[f32]) -> Vec<f32> {
        assert_eq!(output.len(), sample.len());
        let s = self.sigmas[self.index];
        let t = self.sigmas[self.index + 1];
        let skip = SD * SD / (s * s + SD * SD);
        let out = s * SD / (s * s + SD * SD).sqrt();
        let predicted: Vec<f32> = sample
            .iter()
            .zip(output)
            .map(|(&a, &b)| skip * a + out * b)
            .collect();
        let ratio = t / s;
        let result = if self.index == 0 || self.index == self.sigmas.len() - 2 {
            sample
                .iter()
                .zip(&predicted)
                .map(|(&a, &b)| ratio * a - (ratio - 1.0) * b)
                .collect()
        } else {
            let previous = self.previous.as_ref().expect("previous solver output");
            let prev_sigma = self.sigmas[self.index - 1];
            let h = (s as f64 / t as f64).ln();
            let h0 = (prev_sigma as f64 / s as f64).ln();
            let r = (h0 / h) as f32;
            sample
                .iter()
                .zip(&predicted)
                .zip(previous)
                .map(|((&a, &b), &p)| {
                    ratio * a - (ratio - 1.0) * b - 0.5 * (ratio - 1.0) * (b - p) / r
                })
                .collect()
        };
        self.previous = Some(predicted);
        self.index += 1;
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn last_sigma_zero() {
        let mut e = Edm::new(20);
        assert_eq!(*e.sigmas.last().unwrap(), 0.0);
        let mut x = vec![1.0; 12];
        for _ in 0..20 {
            x = e.step(&vec![0.0; 12], &x);
        }
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
