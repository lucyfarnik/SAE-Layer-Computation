#!/bin/bash
zip -r sae_layer_computation.zip . -x .git/**\* artifacts/**\* checkpoints/**\* \
    env/**\* feature_dashboards/**\* sae_layer_computation.zip wandb/**\* weights/**\* **.pyc

