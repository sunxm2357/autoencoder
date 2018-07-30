#!/usr/bin/env bash

python prepare-data.py \
    ../hri_data/caption_events.txt \
    ../hri_data \
    --max-length=25 \
    --min-freq=1 \
