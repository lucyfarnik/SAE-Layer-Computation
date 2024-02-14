import math
from tqdm import tqdm
import sys
sys.path.append('..')
from sae_training.sparse_autoencoder import SparseAutoencoder
from datasets import load_dataset
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEFAULT_HOOKS = ['resid_pre', 'attn_out', 'resid_mid', 'mlp_out', 'resid_post']

def load_saes(path, hooks=DEFAULT_HOOKS):
    if 'HOOK_GOES_HERE' not in path:
        raise ValueError("Your path must contain the substring 'HOOK_GOES_HERE', to be replaced with the hook names.")
    
    saes = {}
    for hook in hooks:
        hook_path = path.replace('HOOK_GOES_HERE', hook)
        saes[hook] = SparseAutoencoder.load_from_pretrained(hook_path)
        saes[hook].eval()
    return saes

def get_sae_acts(sae, act):
    return F.relu(((act - sae.b_dec) @ sae.W_enc) + sae.b_enc)

def get_all_sae_acts(saes, cache, seq_pos=-1,
                     hooks=DEFAULT_HOOKS) -> tuple[dict[str, torch.Tensor],
                                                   list[list[int]]]:
    sae_activations = {}
    for hook in hooks:
        sae_activations[hook] = get_sae_acts(saes[hook], cache[hook, 0])[0, seq_pos, :]
    sae_active_features = [torch.nonzero(act).squeeze(-1).tolist()
                           for act in sae_activations.values()]
    
    return sae_activations, sae_active_features

def plot_all_saes(saes, cache, cache2=None, title_suffix="", hooks=DEFAULT_HOOKS):
    sae_activations, sae_active_features = get_all_sae_acts(saes, cache)
    sae_activations2, sae_active_features2 = get_all_sae_acts(saes, cache2) if cache2 else (None, [])
    
    # TODO change the plot to show the strength of the activations

    fig = go.Figure()

    for i, (active_features, hook) in enumerate(zip(sae_active_features, hooks)):
        # Initialize lists to hold positions, colors, shapes, and hover texts
        positions = []
        colors = []
        shapes = []
        hover_texts = []
        
        # Determine unique features in both caches for comparison
        unique_features = set(active_features + (sae_active_features2[i] if sae_active_features2 else []))
        sorted_features = sorted(unique_features)
        
        for feature in sorted_features:
            position = sorted_features.index(feature)
            positions.append(position)
            hover_texts.append(f"Feature {feature}")
            
            # Determine color and shape based on activation in caches
            if (cache2 is None
                or feature in active_features
                    and (sae_active_features2 and feature in sae_active_features2[i])):
                colors.append("green")  # Active in both
                shapes.append("circle")
            elif feature in active_features:
                colors.append("red")  # Only in cache
                shapes.append("star")
            elif sae_active_features2 and feature in sae_active_features2[i]:
                colors.append("blue")  # Only in cache2
                shapes.append("diamond")

        # Add trace with customized markers
        fig.add_trace(go.Scatter(
            x=positions,
            y=[i] * len(positions),
            mode='markers+text',
            marker=dict(
                size=20, 
                color=colors,
                symbol=shapes
            ),
            text=sorted_features,  # Original text displayed below each marker
            textfont=dict(size=7),
            hoverinfo='text',  # Use custom text for hover info
            hovertext=hover_texts,  # Custom hover text for each point
            textposition="bottom center",
            name=hook,
            showlegend=False,
        ))
        
    if cache2 is not None:
        # Add dummy traces for legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green', symbol='circle'), name='Active in both'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red', symbol='star'), name='Only in original cache'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='blue', symbol='diamond'), name='Only in modified cache'))


    # Customize the layout
    fig.update_layout(
        title='Sparse Autoencoder Activations ' + title_suffix,
        xaxis=dict(
            title='',
            showticklabels=False
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sae_active_features))),
            ticktext=hooks,
            autorange="reversed"
        ),
        showlegend=(cache2 is not None),
    )

    fig.show()

def logits_plot(logits, model, title_suffix="", seq_pos=-1):
    logits_at_pos = torch.softmax(logits[0, seq_pos, :].detach(), dim=0)
    probs, idxs = logits_at_pos.topk(10)
    labels = [f"'{model.to_str_tokens(i)[0]}'" for i in idxs]

    other_prob = 1 - probs.sum().item()
    probs = probs.tolist() + [other_prob]
    labels.append("Other")

    # Create a bar chart
    fig = px.bar(x=labels, y=probs,
                labels={'x': 'Predictions', 'y': 'Probability'},
                title="Top 10 Predictions of the Language Model " + title_suffix)

    fig.show()

def full_zero_ablate_hook(acts, hook, seq_pos=-1):
    acts[:, seq_pos] = 0
    return acts

def plot_sae(saes, cache, hook, seq_pos=-1):
    sae_acts = get_sae_acts(saes[hook], cache[hook, 0][0, seq_pos])
    sae_active_features = torch.nonzero(sae_acts).squeeze(-1).tolist()
    act_values = sae_acts[sae_active_features].cpu().detach().numpy()
    # show as bar chart (w labels)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Feature {i}" for i in sae_active_features],
                        y=act_values))
    fig.update_layout(title="Activation Strength of Attention SAE Features",
                        xaxis_title="Feature",
                        yaxis_title="Activation Strength")

    fig.show()
    
def ablate_sae_feature_enc(acts, hook, saes, sae_feature_indices=[], seq_pos=-1):
    print("WARNING: You're ablating with W_enc, which generally underperforms W_dec")
    
    hook_name = hook.name[14:] # remove the "block.0.hook_" from the name
    sae = saes[hook_name]
    for feature in sae_feature_indices:
        current_act_strength = F.relu((acts[:, seq_pos] - sae.b_dec) @ sae.W_enc[:, feature])
        acts[:, seq_pos] -= current_act_strength * sae.W_enc[:, feature]
    return acts

def ablate_sae_feature(saes,
                       sae_feature_indices: list[int] | int = [],
                       seq_pos: int = -1):
    if isinstance(sae_feature_indices, int):
        sae_feature_indices = [sae_feature_indices]
    
    def hook_func(acts, hook):
        hook_name = hook.name[14:] # remove the "block.0.hook_" from the name
        sae = saes[hook_name]
        for feature in sae_feature_indices:
            current_act_strength = F.relu((acts[:, seq_pos] - sae.b_dec) @ sae.W_enc[:, feature])
            acts[:, seq_pos] -= current_act_strength * sae.W_dec[feature]
        
        return acts
    
    return hook_func

def gib_tokens(n_tokens: int = 10_000,
               dataloader: bool = False,
               dataloader_batch_size: int = 16):
    n_batches = math.ceil(n_tokens / 1024)
    
    dataset = iter(load_dataset("NeelNanda/c4-tokenized-2b", split="train", streaming=True))

    tokens = []
    for _ in tqdm(range(n_batches)):
        tokens.append(torch.tensor(next(dataset)["tokens"],
                                dtype=torch.long,
                                device="mps",
                                requires_grad=False))
    tokens = torch.stack(tokens, dim=0)
    
    if dataloader:
        return DataLoader(tokens, batch_size=dataloader_batch_size, shuffle=True)
    return tokens
    