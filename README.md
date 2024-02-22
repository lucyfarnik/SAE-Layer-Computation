# Circuit-style analysis in SAE basis
This is a MATS project by Lucy Farnik, supervised by Neel Nanda, Arthur Conmy, and Laurence Aitchison.

THIS IS A WORK IN PROGRESS and therefore will probably be slightly chaotic for a while.

I'm working on understanding how a small model like Gelu-1l predicts the next token by doing ~circuit-style analysis on SAE features. The plan is to figure out which SAE features are causally important, how ablating resid_mid features influences mlp_out features, etc., and overall tell a compelling story of how the model can predict the next token in a specific setting.

I'm hoping to figure out:
- Whether SAE features are a better basis for doing circuit-style analysis than eg attn heads, how easy it is to do this, how easy it might be to automate
- Whether there are causally important features that aren't interpretable with max activating examples (maybe because they're too abstract or correspond to intermediate variables computed by the model and used by subsequent layers)
- Get some traction on "how neatly decomposable is reasoning in SAE basis" vs "how heuristic-y and messy and distributed it is"

My high-level direction is basically taking a big black box like a transformer, and breaking it down into a bunch of smaller black boxes (ie. layers) where I can perfectly monitor the communication between them, and hopefully each small black box will be too stupid to take over the world on its own. If this is doable, it would be awesome for evals and inference-time control, which is my main theory of change for mech interp.

This project is the first baby step towards being able to do that kind of monitoring.
It's also really neat because this basically completely generalizes to any neural net with residual connections (and generalizing to nets without residual connections should be pretty easy), so if AGI isn't a transformer, this research should hopefully still be equally useful.

## Structure and credit
The `sae_training` directory comes from [Joseph Bloom's excellent SAE training library](https://github.com/jbloomAus/mats_sae_training).
The `sae_dashboards` directory mostly comes from [Callum McDougall's SAE visualizer codebase](https://github.com/callummcdougall/sae_visualizer), with some modifications by Joseph Bloom (which can be found in his SAE training library), and others by myself.
The `feature_viz/sae_visualizer` and `feature_viz/sae_vis` are also both from Callum — I was doing some quick monkey patching on them.

`weights` contains the trained SAE artifacts. File names in this directory correspond to the hyperparameters of the SAE.
