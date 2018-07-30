#!/usr/bin/env bash

python train-autoencoder.py \
    /scratch4/sunxm/text_ae/checkpoints \
    -e 100 \
    -n 50 \
    ../hri_data/vocabulary.txt \
    ../hri_data/train-data.npz \
    ../hri_data/valid-data.npz \
