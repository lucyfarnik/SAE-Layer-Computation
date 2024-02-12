import torch
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_HOOKS = ['resid_pre', 'attn_out', 'resid_mid', 'mlp_out', 'resid_post']

def logits_vis(logits, model, seq_pos=-1):
    """
    Visualizes the top 10 predictions from the logits of a model.
    
    Parameters:
    - logits: A PyTorch tensor of logits from the model's output.
    """
    # Ensure logits are converted to a PyTorch tensor if not already
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    
    # Ensure logits are flattened and that we're only using the last prediction
    logits.squeeze_()  # Remove batch dim
    if logits.ndim > 1:
        logits = logits[seq_pos]
    
    # Compute softmax to get probabilities
    probabilities = torch.softmax(logits, dim=0)
    
    # Sort and select top 10 predictions
    probs, idxs = torch.topk(probabilities, 10)
    labels = [f"'{model.to_str_tokens(i)[0]}'" for i in idxs]  # Convert indices to labels
    
    # Calculate the probability of "Other" as 1 minus the sum of top 10 probabilities
    other_prob = 1 - probs.sum().item()
    
    # Append "Other" category to the data
    labels.append("Other")
    probs = probs.tolist() + [other_prob]
    
    # Plot with Plotly Express
    fig = px.bar(x=labels, y=probs, labels={'x': 'Predictions', 'y': 'Probability'},
                 title="Top 10 Predictions of the Language Model")
    fig.show()
   
def get_sae_acts(cache, saes, seq_pos=-1,
                 hooks=DEFAULT_HOOKS) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    sae_activations = {}
    for hook in hooks:
        sae_activations[hook] = saes[hook](cache[hook, 0])[1][0, seq_pos, :]
    sae_active_features = [torch.nonzero(act).squeeze(-1).tolist()
                           for act in sae_activations.values()]
    
    return sae_activations, sae_active_features

def plot_sae_activations(cache, cache2=None, title_suffix="", hooks=DEFAULT_HOOKS):
    sae_activations, sae_active_features = get_sae_acts(cache)
    sae_activations2, sae_active_features2 = get_sae_acts(cache2) if cache2 else (None, [])
    
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
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red', symbol='square'), name='Only in original cache'))
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

def plot_sae_acts(cache, hook):
    sae_acts, sae_active_features = get_sae_acts(cache)
    act_values = sae_acts[hook][sae_active_features[1]].cpu().detach().numpy()
    # show as bar chart (w labels)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Feature {i}" for i in sae_active_features[1]],
                        y=act_values))
    fig.update_layout(title="Activation Strength of Attention SAE Features",
                        xaxis_title="Feature",
                        yaxis_title="Activation Strength")

    fig.show()
    
def ablate_sae_feature_enc(acts, hook, saes, sae_feature_indices=[], seq_pos=-1):
    hook_name = hook.name[14:] # remove the "block.0.hook_" from the name
    sae = saes[hook_name]
    for feature in sae_feature_indices:
        current_act_strength = (acts[:, seq_pos] - sae.b_dec) @ sae.W_enc[:, feature]
        acts[:, seq_pos] -= current_act_strength * sae.W_enc[:, feature]
    return acts

def ablate_sae_feature_dec(acts, hook, saes, sae_feature_indices=[], seq_pos=-1):
    hook_name = hook.name[14:] # remove the "block.0.hook_" from the name
    sae = saes[hook_name]
    for feature in sae_feature_indices:
        current_act_strength = (acts[:, seq_pos] - sae.b_dec) @ sae.W_enc[:, feature]
        acts[:, seq_pos] -= current_act_strength * sae.W_dec[feature]
    return acts
