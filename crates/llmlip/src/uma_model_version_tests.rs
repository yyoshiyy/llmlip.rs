use super::*;

fn section(model: &str) -> UmaSection {
    let old = model == "uma-s-1p2";
    UmaSection {
        state: "uma_state.json".into(),
        reference_package: "fairchem".into(),
        reference_version: if old { "2.20.0" } else { "2.22.0" }.into(),
        reference_commit: if old {
            "28292f090743f36c1151002964140b04d4b2c6af"
        } else {
            "bb0a062a9a08d8b3ad3995aca5b9765e07804990"
        }
        .into(),
        selected_model: model.into(),
        state_schema: format!("{model}-state-dict-v1"),
        artifact_sha256: if old {
            "ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b"
        } else {
            "b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce"
        }
        .into(),
        artifact_size_bytes: if old { 2_333_393_167 } else { 2_333_393_615 },
    }
}

#[test]
fn accepts_both_pinned_model_versions() -> Result<(), ModelError> {
    validate_uma_manifest(&section("uma-s-1p2"))?;
    validate_uma_manifest(&section("uma-s-1p2p1"))?;
    Ok(())
}

#[test]
fn rejects_renamed_checkpoint_or_mixed_version_metadata() {
    for model in ["uma-s-1p2", "uma-s-1p2p1"] {
        for field in [
            "sha256",
            "size",
            "schema",
            "reference_version",
            "reference_commit",
        ] {
            let mut mixed = section(model);
            let other = section(if model == "uma-s-1p2" {
                "uma-s-1p2p1"
            } else {
                "uma-s-1p2"
            });
            match field {
                "sha256" => mixed.artifact_sha256 = other.artifact_sha256,
                "size" => mixed.artifact_size_bytes = other.artifact_size_bytes,
                "schema" => mixed.state_schema = other.state_schema,
                "reference_version" => mixed.reference_version = other.reference_version,
                "reference_commit" => mixed.reference_commit = other.reference_commit,
                _ => unreachable!(),
            }
            assert!(validate_uma_manifest(&mixed).is_err(), "{model}: {field}");
        }
    }
    assert!(validate_uma_manifest(&section("uma-s-unknown")).is_err());
}

fn state(model: &str) -> serde_json::Value {
    let task_names = [
        "omc_energy",
        "omol_energy",
        "odac_energy",
        "oc20_energy",
        "omat_energy",
        "oc22_energy",
        "oc25_energy",
        "omc_forces",
        "oc20_forces",
        "omat_forces",
        "omol_forces",
        "odac_forces",
        "oc25_forces",
        "oc22_forces",
        "omc_stress",
        "omat_stress",
    ];
    serde_json::json!({
        "selected_model": model,
        "model_config": {
            "model_id": if model == "uma-s-1p2" {"UMA-S-1.2"} else {"UMA-S-1.2.1"},
            "backbone": {
                "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
                "num_experts": 64, "lmax": 2, "mmax": 2, "num_layers": 4,
                "num_distance_basis": 32, "sphere_channels": 128, "hidden_channels": 128,
                "edge_channels": 128, "max_num_elements": 100, "max_neighbors": 300,
                "cutoff": 6.0, "direct_forces": false, "use_pbc": true, "use_pbc_single": true,
                "use_composition_embedding": true, "distance_function": "gaussian",
                "act_type": "gate", "norm_type": "rms_norm_sh", "ff_type": "spectral"
            },
            "heads": {"energyandforcehead": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head"
            }}
        },
        "tasks_config": task_names.iter().map(|name| serde_json::json!({"name": name})).collect::<Vec<_>>()
    })
}

#[test]
fn accepts_matching_checkpoint_config_for_both_versions() -> Result<(), ModelError> {
    validate_uma_state_config(&state("uma-s-1p2"))?;
    validate_uma_state_config(&state("uma-s-1p2p1"))?;
    Ok(())
}

#[test]
fn rejects_checkpoint_model_id_from_other_version() {
    let mut mixed = state("uma-s-1p2p1");
    mixed["model_config"]["model_id"] = serde_json::json!("UMA-S-1.2");
    assert!(validate_uma_state_config(&mixed).is_err());
}

fn composition_store(model: &str) -> Result<UmaTensorStore, ModelError> {
    // An embedding equal to atomic number plus a channel-dependent offset gives
    // a mean we can calculate without an upstream model or learned parameters.
    let values: Vec<f64> = (0..100)
        .flat_map(|z| (0..UMA_SPHERE_CHANNELS).map(move |ch| z as f64 + ch as f64 / 128.0))
        .collect();
    let tensor = UmaTensorPayload::F64 {
        shape: vec![100, UMA_SPHERE_CHANNELS],
        values_f32: values.iter().map(|&x| x as f32).collect(),
        values,
    };
    Ok(UmaTensorStore {
        composition_include_self: uma_model_spec(model)?.composition_include_self,
        pack_words: None,
        tensors: HashMap::from([(UMA_TENSOR_COMPOSITION_EMBEDDING.into(), tensor)]),
        total_bytes: 100 * UMA_SPHERE_CHANNELS * 8,
    })
}

#[test]
fn composition_121_is_an_average_and_replication_invariant() -> Result<(), ModelError> {
    let store = composition_store("uma-s-1p2p1")?;
    let view = UmaRuntimeTensorView { store: &store };
    let mut single = vec![0.0; UMA_SPHERE_CHANNELS];
    view.composition_embedding_for_atomic_numbers(&[1, 1, 8], &mut single)?;
    for (ch, &value) in single.iter().enumerate() {
        assert!((value - (10.0 / 3.0 + ch as f64 / 128.0)).abs() < 1e-13);
    }
    let repeated = [1, 1, 8].repeat(64);
    let mut multiple = vec![0.0; UMA_SPHERE_CHANNELS];
    view.composition_embedding_for_atomic_numbers(&repeated, &mut multiple)?;
    for (&a, &b) in single.iter().zip(&multiple) {
        assert!((a - b).abs() < 1e-13);
    }
    view.composition_embedding_for_atomic_numbers(&[8], &mut single)?;
    assert_eq!(single[0], 8.0);
    Ok(())
}

#[test]
fn composition_12_preserves_legacy_extra_zero() -> Result<(), ModelError> {
    let store = composition_store("uma-s-1p2")?;
    let view = UmaRuntimeTensorView { store: &store };
    let mut out = vec![0.0; UMA_SPHERE_CHANNELS];
    view.composition_embedding_for_atomic_numbers(&[1, 1, 8], &mut out)?;
    assert_eq!(out[0], 2.5);
    view.composition_embedding_for_atomic_numbers(&[8], &mut out)?;
    assert_eq!(out[0], 4.0);
    Ok(())
}
