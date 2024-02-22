import sys
sys.path.append('..')
import os
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
from IPython.display import HTML, display

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.activations_store import ActivationsStore
from feature_viz.sae_visualizer.data_fns import get_feature_data 
from sae_training.utils import LMSparseAutoencoderSessionloader

def show_sae_feature(sae_path: str,
                     hook_point: str,
                     hook_layer: int,
                     feature_idx: int,
                     n_batches_to_sample_from: int = 2**12,
                     n_prompts_to_select: int = 4096 * 6,
                     max_batch_size: int = 256):
    sae = SparseAutoencoder.load_from_pretrained(sae_path)
    model = HookedTransformer.from_pretrained(sae.cfg.model_name)
    model.to(sae.device)

    tokens_path = f"tokens_{n_prompts_to_select}.pt"
    # if the tokens are already on disk, load them
    if os.path.exists(tokens_path):
        tokens = torch.load(tokens_path)
    else: # otherwise load them from HuggingFace
        activation_store = ActivationsStore(sae.cfg, model)
        all_tokens_list = []
        for _ in tqdm(range(n_batches_to_sample_from)):
            batch_tokens = activation_store.get_batch_tokens()
            batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
                : batch_tokens.shape[0]
            ]
            all_tokens_list.append(batch_tokens)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        tokens = all_tokens[:n_prompts_to_select]

        torch.save(tokens, tokens_path)


    feature_data = get_feature_data(
        encoder=sae,
        model=model,
        tokens=tokens,
        hook_point=hook_point,
        hook_layer=hook_layer,
        feature_idx=feature_idx,
        max_batch_size=max_batch_size,
    )
    
    html_str = feature_data[feature_idx].get_all_html()

    return display(HTML(html_str))
    
    