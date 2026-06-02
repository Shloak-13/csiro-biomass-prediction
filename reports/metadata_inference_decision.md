# Metadata Inference Decision

## Image to Metadata
| model      |   ndvi_r2 |   height_r2 |   mean_r2 |   runtime_sec | feature_set       |
|:-----------|----------:|------------:|----------:|--------------:|:------------------|
| ridge      |  0.907641 |    0.892698 |  0.900169 |     0.378978  | all_embeddings    |
| ridge      |  0.860398 |    0.904745 |  0.882572 |     0.207361  | dinov2_large      |
| lightgbm   |  0.80257  |    0.871968 |  0.837269 |    73.4358    | all_embeddings    |
| ridge      |  0.862198 |    0.76869  |  0.815444 |     0.169479  | siglip_base       |
| ridge      |  0.813012 |    0.793374 |  0.803193 |     0.230619  | convnextv2_base   |
| lightgbm   |  0.811504 |    0.786495 |  0.798999 |    22.423     | siglip_base       |
| lightgbm   |  0.721732 |    0.857159 |  0.789446 |    28.2919    | dinov2_large      |
| lightgbm   |  0.876085 |    0.608033 |  0.742059 |     3.5652    | handcrafted_image |
| lightgbm   |  0.718127 |    0.690936 |  0.704532 |    25.5456    | convnextv2_base   |
| extratrees |  0.728478 |    0.634501 |  0.68149  |     3.44871   | handcrafted_image |
| extratrees |  0.494271 |    0.8636   |  0.678935 |    76.6841    | all_embeddings    |
| extratrees |  0.536603 |    0.809132 |  0.672868 |    23.3181    | siglip_base       |
| extratrees |  0.434425 |    0.86495  |  0.649687 |    30.6185    | dinov2_large      |
| extratrees |  0.466842 |    0.757408 |  0.612125 |    29.7003    | convnextv2_base   |
| ridge      |  0.858741 |   -1.45062  | -0.295938 |     0.0665662 | handcrafted_image |

## Pseudo-Metadata to Biomass
| feature_set                             |   cv_weighted_r2 |   Dry_Green_g |   Dry_Dead_g |   Dry_Clover_g |    GDM_g |   Dry_Total_g |
|:----------------------------------------|-----------------:|--------------:|-------------:|---------------:|---------:|--------------:|
| pseudo_ndvi_height_plus_all_embeddings  |         0.596703 |      0.601785 |     0.440367 |       0.598867 | 0.666592 |      0.598565 |
| pseudo_ndvi_height_plus_dinov2_large    |         0.563754 |      0.613414 |     0.357311 |       0.569574 | 0.665312 |      0.553323 |
| pseudo_ndvi_height_plus_siglip_base     |         0.549962 |      0.604842 |     0.274665 |       0.498096 | 0.606961 |      0.581619 |
| pseudo_ndvi_height_plus_convnextv2_base |         0.549374 |      0.567029 |     0.360965 |       0.392583 | 0.644481 |      0.57684  |
| pseudo_ndvi_height_only                 |         0.263394 |      0.325369 |    -0.194184 |      -0.120295 | 0.43018  |      0.352537 |

## Lightweight End-to-End Result
|                | value                     |
|:---------------|:--------------------------|
| model          | tf_efficientnetv2_b0.in1k |
| folds          | 3                         |
| epochs         | 2                         |
| image_size     | 160                       |
| device         | cpu                       |
| runtime_sec    | 253.57638430595398        |
| cv_weighted_r2 | -64.301386336868          |
| Dry_Green_g    | -265.18675285800407       |
| Dry_Dead_g     | -113.89049769324826       |
| Dry_Clover_g   | -18.885875611741433       |
| GDM_g          | -73.41603227496847        |
| Dry_Total_g    | -19.643734531149875       |

## Final Answers
1. Metadata cannot be used directly at inference under the provided schema.
2. Strongest current image-only strategy: full-image DINOv2 Large + XGBoost embedding baseline at `0.556414`, then improve it with canopy crops and multi-crop fusion.
3. NDVI/Height can be predicted from images with best mean R2 `0.900169` (`all_embeddings` + `ridge`), but inspect per-target R2 before trusting pseudo-metadata.
4. Pseudo-metadata viability: best pseudo-metadata biomass CV `0.596703`. Compare this against pure embedding CV before using it.
5. Next experiment: canopy-cropped DINOv2/SigLIP/ConvNeXtV2 embeddings plus pseudo-NDVI/height, blended with the lightweight CNN OOF if it is positive.