from josephs_sae_dashboards.dashboard_runner import DashboardRunner

for hook in ['resid_pre', 'attn_out', 'resid_mid', 'mlp_out', 'resid_post']:
    DashboardRunner(
        sae_path = None,
        dashboard_parent_folder = "./feature_dashboards",
        wandb_artifact_path = f"lucyfarnik/mats_sae_training_language_models/sparse_autoencoder_gelu-1l_blocks.0.hook_{hook}_16384:v4",
        wandb_sparsity_version = 4,
        init_session = True,
        n_batches_to_sample_from = 2**5,#2**12,
        n_prompts_to_select = 128,#4096*6,
        n_features_at_a_time = 8,#128,
        max_batch_size = 1,#256,
        buffer_tokens = 8,
        use_wandb = True,
        continue_existing_dashboard = True,
        final_index=32,
    ).run()

