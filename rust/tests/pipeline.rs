use std::{fs, path::Path};
use terrain_diffusion::{inference::ZeroRunner, Terrain};
fn mock_models(path: &Path) {
    fs::write(
        path.join("world_pipeline_config.json"),
        serde_json::json!({
            "native_resolution":30.0,"latent_compression":4,"coarse_means":[0.,0.,0.,0.,0.,0.],
            "coarse_stds":[1.,1.,1.,1.,1.,1.],"cond_snr":[1.,2.,3.,4.,5.],
            "frequency_mult":[1.,1.,1.,1.,1.],"residual_mean":0.0,
            "residual_std":1.0,"coarse_pooling":1
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        path.join("pipeline_data.json"),
        serde_json::json!({
            "n_quantiles":64,"data_quantile_tables":[
                (0..64).map(|i|i as f32/63.0*100.0).collect::<Vec<_>>(),
                vec![20.0;64],vec![10.0;64],vec![200.0;64],vec![100.0;64]],
            "a_temp_std":0.0,"b_temp_std":0.0,"temp_std_p1":1.0,"temp_std_p99":20.0
        })
        .to_string(),
    )
    .unwrap();
}
#[test]
fn model_free_pipeline_generates_elevation_and_climate_at_negative_coords() {
    let temp = tempfile::tempdir().unwrap();
    mock_models(temp.path());
    let mut world = Terrain::new(1234, temp.path(), ZeroRunner).unwrap();
    let a = world.generate(-18, -21, 8, 8, true).unwrap();
    assert_eq!(a.elevation.len(), 64);
    assert!(a.elevation.iter().all(|v| v.is_finite()));
    assert_eq!(a.climate.as_ref().unwrap().len(), 5 * 64);
    assert!(a.climate.as_ref().unwrap().iter().all(|v| v.is_finite()));
    let count = world.tile_count();
    let b = world.generate(-18, -21, 8, 8, true).unwrap();
    assert_eq!(a.elevation, b.elevation);
    assert_eq!(a.climate, b.climate);
    assert_eq!(world.tile_count(), count);
    let c = world.generate(-17, -20, 6, 6, false).unwrap();
    for r in 0..6 {
        for col in 0..6 {
            assert!((a.elevation[(r + 1) * 8 + col + 1] - c.elevation[r * 6 + col]).abs() < 50.0);
        }
    }
}
