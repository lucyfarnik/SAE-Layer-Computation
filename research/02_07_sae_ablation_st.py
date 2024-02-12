# DEPRECATED: I tried building this into a Streamlit app, but it was too annoying to iterate on.
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from transformer_lens import HookedTransformer
import sys
sys.path.append('.')
from sae_training.sparse_autoencoder import SparseAutoencoder
import torch

st.write("# Ablation experiments on gelu-1l")
model = HookedTransformer.from_pretrained("gelu-1l")

st.write("## Prompt")
# Let the user enter a prompt
prompt = st.text_area("Prompt", placeholder="Enter your prompt here")

st.write("## Tokenization")
str_tokens = model.to_str_tokens(prompt)

# Updated CSS and function to use Flexbox for token layout
token_css = "border: 1px solid white; margin: 0.1rem; padding: 0.1rem 0.75rem; border-radius: 0.5rem; display: inline-flex; align-items: center; justify-content: center;"
container_css = "display: flex; flex-wrap: wrap; gap: 0.5rem;"

def generate_html_for_tokens(tokens):
    # Start building the HTML string with Flexbox container
    html_str = f'<div style="{container_css}">'
    # Add each token with styling
    for token in tokens:
        html_str += f'<span style="{token_css}">"{token}"</span>'
    html_str += '</div>'
    return html_str

# Assume `str_tokens` is your list of token strings
html_to_display = generate_html_for_tokens(str_tokens)
st.markdown(html_to_display, unsafe_allow_html=True)

# Pick the sequence position to analyze
if len(str_tokens) > 1:
    st.write("## Sequence Position")
    seq_pos = st.slider("Select a sequence position",
                        min_value=0,
                        max_value=len(str_tokens) - 1,
                        value=len(str_tokens) - 1)
else:
    seq_pos = 0

st.write(f"###### Selected Token: '{str_tokens[seq_pos]}'")

# Get the SAE activations
st.write("## SAE Activations")
hooks = ['resid_pre', 'attn_out', 'resid_mid', 'mlp_out', 'resid_post']
saes = {}
for hook in hooks:
    #TODO do this only once, not each time the page reloads
    saes[hook] = SparseAutoencoder.load_from_pretrained(f'./weights/saes32/final_sparse_autoencoder_gelu-1l_blocks.0.hook_{hook}_16384.pt')
    saes[hook].eval()

clean_logits, cache = model.run_with_cache(prompt)

sae_activations = {}
for hook in hooks:
    sae_activations[hook] = saes[hook](cache[hook, 0])[1][0, seq_pos, :]
sae_active_features = [torch.nonzero(act).squeeze(-1).tolist()
                       for act in sae_activations.values()]

# reset the state everytime the prompt changes
if ('prompt' not in st.session_state
    or 'ablated_features' not in st.session_state
    or st.session_state.prompt != prompt):
    st.session_state.ablated_features = [[] for _ in range(5)]
    st.session_state.prompt = prompt

with st.expander("Ablate SAE features"):
    # Specify disabled features for each SAE
    # For each hook, allow the user to select which features to disable
    for i, hook in enumerate(hooks):
        st.session_state.ablated_features[i] = st.multiselect(
            f"Select features to disable for {hook}:",
            options=sae_active_features[i],  # You'll need to define `max_feature_index` appropriately
            default=st.session_state.ablated_features[i]
        )

# Create a Plotly figure
fig = go.Figure()

for i, (active_features, hook, disabled) in enumerate(zip(sae_active_features,
                                                          hooks,
                                                          st.session_state.ablated_features)):
    sorted_features = sorted(active_features)
    positions = list(range(len(sorted_features)))
    colors = ['red' if feature in disabled else 'green' for feature in sorted_features]

    # Generate hover text for each feature
    hover_texts = [f"Feature {feature} {'(Disabled)' if feature in disabled else '(Active)'}"
                   for feature in sorted_features]
    
    fig.add_trace(go.Scatter(
        x=positions,
        y=[i] * len(sorted_features),
        mode='markers+text',
        marker=dict(size=20, color=colors),
        text=sorted_features,  # Original text displayed below each marker
        hoverinfo='text',  # Use custom text for hover info
        hovertext=hover_texts,  # Custom hover text for each point
        textposition="bottom center",
        name=hook,
    ))

# Customize the layout
fig.update_layout(
    title='Sparse Autoencoder Activations',
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
    showlegend=False
)

# Display the figure in Streamlit
st.plotly_chart(fig)


# Show the logits
st.write("## Logits")

def get_top_logits(logits):
    logits_at_pos = torch.softmax(logits[0, seq_pos, :].detach(), dim=0)
    top10_probs, top10_idxs = logits_at_pos.topk(10)
    labels = [f"'{model.to_str_tokens(i)[0]}'" for i in top10_idxs]

    other_prob = 1 - top10_probs.sum().item()
    top10_probs = top10_probs.tolist() + [other_prob]
    labels.append("Other")
    
    return top10_probs, labels
clean_probs, clean_labels = get_top_logits(clean_logits)

# Create a bar chart
fig = px.bar(x=clean_labels, y=clean_probs,
             labels={'x': 'Predictions', 'y': 'Probability'},
             title="Top 10 Predictions of the Language Model")

# Display the figure in Streamlit
st.plotly_chart(fig)
