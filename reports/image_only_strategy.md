# Image-Only Strategy Report

## Recommended Image-Only Stack
1. Preprocess each 2000x1000 quadrat with vegetation segmentation using excess-green and saturation thresholds.
2. Generate canopy-focused crops: full image, center crop, left/right halves, and top vegetation-density patches.
3. Extract frozen embeddings per crop using DINOv2 Large, SigLIP, and ConvNeXtV2.
4. Add image-derived pseudo-metadata: predicted NDVI and predicted height.
5. Train per-target GBDTs and a small CNN/ConvNeXt/EfficientNet model; ensemble OOF predictions.
6. Use multi-crop test-time averaging and enforce non-negativity plus validation-tested biomass consistency.

## Why
Raw full-image embeddings underperformed metadata, but embeddings-only models beat hand-crafted image features. The likely missing piece is crop quality: the images are wide 2:1 quadrats with substantial background/frame variation.