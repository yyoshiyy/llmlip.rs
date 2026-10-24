use super::*;

#[test]
fn quaternion_wigner_edge_vjp_matches_finite_difference() -> Result<(), ModelError> {
    // Avoid the quaternion chart's sign boundary; include both O-H vectors
    // from the H2O reproduction and generic directions in all quadrants.
    let vectors = [
        [0.0, 0.763239, -0.596309],
        [0.0, -0.763239, -0.596309],
        [0.23, 0.75, -0.59],
        [-0.61, -0.35, 0.43],
    ];
    let cotangent: Vec<f64> = (0..UMA_BASIS_DIM * UMA_BASIS_DIM)
        .map(|i| (i as f64 * 0.37 + 0.19).sin())
        .collect();
    let step = 1.0e-6;
    for vector in vectors {
        let analytic = uma_quaternion_wigner_inv_edge_vector_vjp(vector, &cotangent)?;
        for axis in 0..3 {
            let mut plus_vector = vector;
            let mut minus_vector = vector;
            plus_vector[axis] += step;
            minus_vector[axis] -= step;
            let mut plus = [0.0; UMA_BASIS_DIM * UMA_BASIS_DIM];
            let mut minus = [0.0; UMA_BASIS_DIM * UMA_BASIS_DIM];
            compute_uma_quaternion_wigner_inv_from_project_edge_vector(plus_vector, &mut plus)?;
            compute_uma_quaternion_wigner_inv_from_project_edge_vector(minus_vector, &mut minus)?;
            let numeric: f64 = cotangent
                .iter()
                .zip(plus.iter().zip(&minus))
                .map(|(g, (p, m))| g * (p - m) / (2.0 * step))
                .sum();
            assert!(
                (analytic[axis] - numeric).abs() < 1.0e-8,
                "vector={vector:?}, axis={axis}, analytic={}, numeric={numeric}",
                analytic[axis]
            );
        }
    }
    Ok(())
}

struct BlockFixture {
    runtime: UmaRuntimeModel,
    prepared: UmaPreparedInput,
    weights: UmaBlockMixedWeights,
    input: Vec<f64>,
    cotangent: Vec<f64>,
}

fn values(n: usize, scale: f64, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|i| scale * (i as f64 * 0.137 + phase).sin())
        .collect()
}

fn synthetic_so2(config: EsenSo2ConvolutionConfig) -> Result<UmaMixedSo2Weights, ModelError> {
    let m0 = values(config.m0_weight_len(), 0.025, 0.2);
    let m: Vec<Vec<f64>> = (1..=config.m_max())
        .map(|order| values(config.m_conv_weight_len(order), 0.025, order as f64))
        .collect();
    let refs: Vec<&[f64]> = m.iter().map(Vec::as_slice).collect();
    mix_uma_so2_weights_minor(config, 1, &m0, &refs, &[1.0])
}

fn block_fixture() -> Result<BlockFixture, ModelError> {
    let channels = UMA_SPHERE_CHANNELS;
    let conv1 = EsenSo2ConvolutionConfig::new(
        UMA_L_MAX,
        UMA_M_MAX,
        2 * channels,
        channels,
        UMA_BLOCK_CONV1_EXTRA_M0_CHANNELS,
    )?;
    let conv2 = EsenSo2ConvolutionConfig::new(UMA_L_MAX, UMA_M_MAX, channels, channels, 0)?;
    let atomwise = EsenSpectralAtomwiseConfig::new(UMA_L_MAX, channels, channels)?;
    let weights = UmaBlockMixedWeights {
        conv1: synthetic_so2(conv1)?,
        conv2: synthetic_so2(conv2)?,
        static_weights: UmaBlockStaticWeights {
            norm1_weight: vec![1.0; (UMA_L_MAX + 1) * channels],
            norm1_bias: vec![0.0; channels],
            norm2_weight: vec![1.0; (UMA_L_MAX + 1) * channels],
            norm2_bias: vec![0.0; channels],
            conv1_m0_bias: values(conv1.m0_output_feature_dim(), 0.1, 0.3),
            conv2_m0_bias: values(conv2.m0_output_feature_dim(), 0.1, 0.4),
            atomwise_scalar_weight: values(atomwise.scalar_gate_weight_len(), 0.02, 0.5),
            atomwise_scalar_bias: values(atomwise.scalar_gate_bias_len(), 0.1, 0.6),
            atomwise_so3_linear1_weight: values(
                atomwise.so3_linear1_config()?.weight_len(),
                0.02,
                0.7,
            ),
            atomwise_so3_linear1_bias: vec![0.0; channels],
            atomwise_so3_linear2_weight: values(
                atomwise.so3_linear2_config()?.weight_len(),
                0.02,
                0.8,
            ),
            atomwise_so3_linear2_bias: vec![0.0; channels],
        },
    };
    // Constant radial output isolates the matrix derivative from distance terms.
    let radial = EsenRadialMlp::new(
        vec![EsenRadialMlpLayer::new(
            UMA_EDGE_DEGREE_RADIAL_INPUT_DIM,
            UMA_BLOCK_CONV1_RADIAL_OUTPUT_DIM,
            vec![0.0; UMA_EDGE_DEGREE_RADIAL_INPUT_DIM * UMA_BLOCK_CONV1_RADIAL_OUTPUT_DIM],
            vec![0.8; UMA_BLOCK_CONV1_RADIAL_OUTPUT_DIM],
            None,
        )?],
        UMA_RADIAL_MLP_LAYER_NORM_EPS,
    )?;
    let runtime = UmaRuntimeModel {
        edge_degree_radial_mlp: radial.clone(),
        block_conv1_radial_mlps: vec![radial],
    };
    let mut wigner_inv = vec![0.0; UMA_BASIS_DIM * UMA_BASIS_DIM];
    compute_uma_quaternion_wigner_inv_from_project_edge_vector(
        [0.23, 0.75, -0.59],
        &mut wigner_inv,
    )?;
    let mut wigner = vec![0.0; wigner_inv.len()];
    transpose_uma_batched_wigner_inv(&wigner_inv, &mut wigner);
    let prepared = UmaPreparedInput {
        atomic_numbers: vec![1, 8],
        cell: None,
        pbc: [false; 3],
        edge_sources: vec![0],
        edge_targets: vec![1],
        edge_source_offsets: vec![0, 1, 1],
        edge_source_edge_indices: vec![0],
        edge_target_offsets: vec![0, 0, 1],
        edge_target_edge_indices: vec![0],
        edge_vectors: vec![[0.23, 0.75, -0.59]],
        edge_distances: vec![1.0],
        edge_envelopes: vec![0.8],
        edge_envelope_derivatives: vec![0.0],
        node_sphere_embeddings: Vec::new(),
        edge_degree_mlp_inputs: vec![0.3; UMA_EDGE_DEGREE_RADIAL_INPUT_DIM],
        edge_degree_mlp_outputs: Vec::new(),
        edge_degree_mlp_caches: Default::default(),
        edge_degree_m_coefficients: Vec::new(),
        sys_node_embedding: vec![0.0; channels],
        node_message_before_edge_degree: Vec::new(),
        edge_wigner_inv: wigner_inv.clone(),
        edge_compact_wigner: wigner,
        edge_compact_wigner_inv: wigner_inv,
        node_message_after_edge_degree: Vec::new(),
        edge_compact_wigner_f32: Vec::new(),
        edge_compact_wigner_inv_f32: Vec::new(),
        edge_compact_wigner_inv_order_packed_f32: Vec::new(),
        edge_degree_mlp_inputs_f32: Vec::new(),
        edge_degree_mlp_outputs_f32: Vec::new(),
        edge_degree_mlp_caches_f32: Default::default(),
        edge_degree_m_coefficients_f32: Vec::new(),
        node_message_after_edge_degree_f32: Vec::new(),
        edge_envelopes_f32: Vec::new(),
        edge_envelope_derivatives_f32: Vec::new(),
        dataset: UmaDataset::Omol,
        charge: 0,
        spin: 1,
    };
    Ok(BlockFixture {
        runtime,
        prepared,
        weights,
        input: values(2 * UMA_NODE_FEATURE_DIM, 0.3, 0.9),
        cotangent: values(2 * UMA_NODE_FEATURE_DIM, 0.2, 1.1),
    })
}

fn check_block_wigner_vjp(profiled: bool) -> Result<(), ModelError> {
    let fixture = block_fixture()?;
    let store = UmaTensorStore {
        pack_words: None,
        tensors: HashMap::new(),
        total_bytes: 0,
    };
    // These block paths use already mixed weights; no checkpoint access occurs.
    let tensors = UmaRuntimeTensorView { store: &store };
    let forward = |prepared: &UmaPreparedInput| {
        if profiled {
            replay_uma_block_profiled(
                &tensors,
                &fixture.runtime,
                0,
                &fixture.input,
                prepared,
                &fixture.weights,
                &prepared.sys_node_embedding,
                false,
                None,
            )
        } else {
            replay_uma_block(
                &tensors,
                &fixture.runtime,
                0,
                &fixture.input,
                prepared,
                &fixture.weights,
                &prepared.sys_node_embedding,
                false,
            )
        }
    };
    let replay = forward(&fixture.prepared)?;
    // Nonzero initial adjoints also check accumulation across multiple blocks.
    let initial_gradient = 0.17;
    let mut geometry = UmaPreparedGeometryVjp {
        grad_edge_distances: vec![0.0],
        grad_edge_degree_mlp_inputs: vec![0.0; UMA_EDGE_DEGREE_RADIAL_INPUT_DIM],
        grad_wigner_inv: vec![initial_gradient; UMA_BASIS_DIM * UMA_BASIS_DIM],
    };
    if profiled {
        replay_uma_block_vjp_profiled(
            &tensors,
            &fixture.runtime,
            0,
            &fixture.input,
            &fixture.prepared,
            &fixture.weights,
            &fixture.cotangent,
            &mut geometry,
            &replay,
            None,
        )?;
    } else {
        replay_uma_block_vjp_from_replay(
            &tensors,
            &fixture.runtime,
            0,
            &fixture.input,
            &fixture.prepared,
            &fixture.weights,
            &fixture.cotangent,
            &mut geometry,
            &replay,
        )?;
    }
    let loss = |prepared: &UmaPreparedInput| -> Result<f64, ModelError> {
        Ok(forward(prepared)?
            .output
            .iter()
            .zip(&fixture.cotangent)
            .map(|(x, g)| x * g)
            .sum())
    };
    let step = 1.0e-5;
    for (row, col) in [(1, 2), (2, 3), (4, 8), (8, 4)] {
        let index = row * UMA_BASIS_DIM + col;
        let mut plus = fixture.prepared.clone();
        let mut minus = fixture.prepared.clone();
        plus.edge_wigner_inv[index] += step;
        minus.edge_wigner_inv[index] -= step;
        for prepared in [&mut plus, &mut minus] {
            prepared
                .edge_compact_wigner_inv
                .clone_from(&prepared.edge_wigner_inv);
            transpose_uma_batched_wigner_inv(
                &prepared.edge_wigner_inv,
                &mut prepared.edge_compact_wigner,
            );
        }
        let numeric = (loss(&plus)? - loss(&minus)?) / (2.0 * step);
        let analytic = geometry.grad_wigner_inv[index] - initial_gradient;
        assert!(
            (analytic - numeric).abs() < 1.0e-7,
            "profiled={profiled}, matrix[{row},{col}]: analytic={analytic}, numeric={numeric}"
        );
    }
    Ok(())
}

#[test]
fn block_wigner_vjp_matches_finite_difference() -> Result<(), ModelError> {
    check_block_wigner_vjp(false)
}

#[test]
fn profiled_block_wigner_vjp_matches_finite_difference() -> Result<(), ModelError> {
    check_block_wigner_vjp(true)
}
