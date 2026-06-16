# Metadata Importance

## Metadata-only CV Scores
| model             |   cv_weighted_r2 |   fold_0 |   fold_1 |   fold_2 |   fold_3 |   fold_4 |
|:------------------|-----------------:|---------:|---------:|---------:|---------:|---------:|
| LightGBM_metadata |         0.711018 | 0.564705 | 0.773923 | 0.762723 | 0.758065 | 0.714698 |
| CatBoost_metadata |         0.668325 | 0.56315  | 0.681401 | 0.689808 | 0.710925 | 0.712173 |

## Metadata + Image Feature CV Scores
| model                            |   cv_weighted_r2 | fold_scores                                                                                          |
|:---------------------------------|-----------------:|:-----------------------------------------------------------------------------------------------------|
| LightGBM_metadata_image_features |         0.670438 | [0.5626662863316951, 0.6598979459899486, 0.7220639207137667, 0.6955809352024043, 0.7289560636486068] |
| CatBoost_metadata_image_features |         0.651022 | [0.568026441712278, 0.6600195569693693, 0.6826901297672299, 0.663812267613, 0.6961430654670901]      |

## Interpretation
Metadata dominates this quick research phase. `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, and date-derived features with LightGBM reach `0.71102` weighted CV R2, while hand-crafted image features alone reach `0.39257` and the 96-image embedding subset reaches `0.36231`. Adding hand-crafted image features to metadata reduced CV to `0.67044`, so simple concatenation is not yet justified without stronger regularization, better validation, or foundation embeddings.
