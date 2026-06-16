# Image Analysis

## Top Predictive Image Features by Spearman Correlation
| feature                 | target       |   spearman |   abs_spearman |
|:------------------------|:-------------|-----------:|---------------:|
| green_pixel_ratio       | GDM_g        |   0.570284 |       0.570284 |
| exg_mean                | GDM_g        |   0.5246   |       0.5246   |
| hist_g_6                | Dry_Green_g  |   0.517508 |       0.517508 |
| hist_exg_3              | GDM_g        |  -0.50586  |       0.50586  |
| hist_g_6                | GDM_g        |   0.494386 |       0.494386 |
| hist_b_6                | Dry_Clover_g |  -0.482776 |       0.482776 |
| vegetation_coverage_exg | GDM_g        |   0.460948 |       0.460948 |
| hist_b_5                | Dry_Clover_g |  -0.454641 |       0.454641 |
| green_pixel_ratio       | Dry_Green_g  |   0.430632 |       0.430632 |
| hist_exg_3              | Dry_Green_g  |  -0.42623  |       0.42623  |
| hist_exg_6              | GDM_g        |   0.426077 |       0.426077 |
| exg_mean                | Dry_Green_g  |   0.423248 |       0.423248 |
| hist_g_7                | Dry_Green_g  |   0.414286 |       0.414286 |
| saturation_mean         | GDM_g        |   0.410364 |       0.410364 |
| hist_exg_5              | GDM_g        |   0.407315 |       0.407315 |
| hist_g_6                | Dry_Total_g  |   0.404589 |       0.404589 |
| saturation_std          | Dry_Total_g  |   0.39767  |       0.39767  |
| hist_b_0                | GDM_g        |   0.394587 |       0.394587 |
| hist_g_7                | Dry_Total_g  |   0.39101  |       0.39101  |
| hist_gray_6             | Dry_Green_g  |   0.384826 |       0.384826 |
| vegetation_coverage_exg | Dry_Green_g  |   0.380771 |       0.380771 |
| hist_r_6                | Dry_Dead_g   |   0.374179 |       0.374179 |
| hist_r_7                | Dry_Dead_g   |   0.371114 |       0.371114 |
| hist_g_7                | GDM_g        |   0.3702   |       0.3702   |
| exg_mean                | Dry_Total_g  |   0.369181 |       0.369181 |
| hist_exg_3              | Dry_Total_g  |  -0.367327 |       0.367327 |
| saturation_std          | Dry_Green_g  |   0.365073 |       0.365073 |
| hist_b_7                | Dry_Clover_g |  -0.36329  |       0.36329  |
| hist_b_2                | Dry_Green_g  |  -0.359423 |       0.359423 |
| hist_r_0                | GDM_g        |   0.359414 |       0.359414 |

## Top Predictive Image Features by ExtraTrees Importance
| feature                 |   extra_trees_importance |
|:------------------------|-------------------------:|
| hist_g_6                |                0.0674349 |
| vegetation_coverage_exg |                0.0644427 |
| hist_exg_3              |                0.0499589 |
| green_pixel_ratio       |                0.0406367 |
| hist_gray_6             |                0.0398257 |
| hist_b_6                |                0.0396648 |
| exg_mean                |                0.0355992 |
| hist_exg_6              |                0.0345854 |
| texture_laplacian_var   |                0.0302184 |
| hist_exg_5              |                0.0296874 |
| hist_b_7                |                0.0293363 |
| hist_g_7                |                0.0232301 |
| saturation_std          |                0.0213509 |
| hist_b_5                |                0.0209656 |
| texture_sobel_mean      |                0.0205358 |
| gray_entropy            |                0.0195413 |
| hist_exg_7              |                0.0194838 |
| hist_b_2                |                0.0181395 |
| hist_g_3                |                0.018103  |
| hist_g_0                |                0.0179882 |
| hist_exg_4              |                0.0177244 |
| saturation_mean         |                0.0172002 |
| hist_g_2                |                0.0169556 |
| exg_std                 |                0.0153463 |
| hist_gray_3             |                0.0152385 |
| hist_r_2                |                0.0150338 |
| hist_r_5                |                0.0150078 |
| hist_r_6                |                0.0146062 |
| hist_gray_7             |                0.0144217 |
| hist_gray_2             |                0.0141211 |

## Hand-crafted Image Feature CV Scores
| model                   |   cv_weighted_r2 | fold_scores                                                                                             |
|:------------------------|-----------------:|:--------------------------------------------------------------------------------------------------------|
| LightGBM_image_features |         0.392569 | [0.2811007999468943, 0.3589427276882717, 0.46914791981307497, 0.3613213113189509, 0.5063644907314331]   |
| CatBoost_image_features |         0.368593 | [0.24021513405100284, 0.39493293245299027, 0.4649163005250798, 0.3361481825122463, 0.43022921361896116] |

## Metadata + Hand-crafted Image Feature CV Scores
| model                            |   cv_weighted_r2 | fold_scores                                                                                          |
|:---------------------------------|-----------------:|:-----------------------------------------------------------------------------------------------------|
| LightGBM_metadata_image_features |         0.670438 | [0.5626662863316951, 0.6598979459899486, 0.7220639207137667, 0.6955809352024043, 0.7289560636486068] |
| CatBoost_metadata_image_features |         0.651022 | [0.568026441712278, 0.6600195569693693, 0.6826901297672299, 0.663812267613, 0.6961430654670901]      |