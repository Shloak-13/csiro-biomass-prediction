# Multi-Crop Representation Benchmark

## PCA-Compressed Image -> Metadata
| feature_set                      |   dim |   ndvi_r2 |   height_r2 |   mean_r2 |
|:---------------------------------|------:|----------:|------------:|----------:|
| all_models_center_pca256         |   256 |  0.59374  |    0.436448 |  0.515094 |
| all_models_original_pca256       |   256 |  0.583029 |    0.413716 |  0.498373 |
| dinov2_large_all_crops_pca256    |   256 |  0.568016 |    0.424534 |  0.496275 |
| all_models_highveg_pca256        |   256 |  0.618928 |    0.370806 |  0.494867 |
| siglip_base_all_crops_pca256     |   256 |  0.56916  |    0.403569 |  0.486364 |
| all_models_all_crops_pca256      |   256 |  0.594225 |    0.349082 |  0.471654 |
| all_models_canopy_pca256         |   256 |  0.459017 |    0.237572 |  0.348294 |
| convnextv2_base_all_crops_pca256 |   256 |  0.521331 |    0.153034 |  0.337183 |

## Raw Image -> Metadata
| feature_set              |   dim |   ndvi_r2 |   height_r2 |   mean_r2 |
|:-------------------------|------:|----------:|------------:|----------:|
| raw_all_models_all_crops | 11264 |  0.920985 |    0.896017 |  0.908501 |
| raw_all_models_original  |  2816 |  0.907641 |    0.892698 |  0.900169 |
| raw_all_models_highveg   |  2816 |  0.909143 |    0.878387 |  0.893765 |

## PCA-Compressed Multi-Task / Biomass
| feature_set                      |   dim | task               | model    |   cv_weighted_r2 |   Dry_Green_g |   Dry_Dead_g |   Dry_Clover_g |    GDM_g |   Dry_Total_g |
|:---------------------------------|------:|:-------------------|:---------|-----------------:|--------------:|-------------:|---------------:|---------:|--------------:|
| all_models_all_crops_pca256      |   256 | biomass_only       | xgboost  |         0.530823 |      0.620786 |     0.224897 |       0.480362 | 0.640386 |      0.540282 |
| all_models_all_crops_pca256      |   256 | joint_meta_biomass | xgboost  |         0.530823 |      0.620786 |     0.224897 |       0.480362 | 0.640386 |      0.540282 |
| all_models_original_pca256       |   256 | biomass_only       | xgboost  |         0.527013 |      0.595003 |     0.272318 |       0.47881  | 0.613256 |      0.539497 |
| all_models_original_pca256       |   256 | joint_meta_biomass | xgboost  |         0.527013 |      0.595003 |     0.272318 |       0.47881  | 0.613256 |      0.539497 |
| all_models_center_pca256         |   256 | biomass_only       | xgboost  |         0.525297 |      0.605739 |     0.215809 |       0.420421 | 0.591979 |      0.565408 |
| all_models_center_pca256         |   256 | joint_meta_biomass | xgboost  |         0.525297 |      0.605739 |     0.215809 |       0.420421 | 0.591979 |      0.565408 |
| all_models_all_crops_pca256      |   256 | biomass_only       | lightgbm |         0.521726 |      0.611144 |     0.244077 |       0.530697 | 0.635795 |      0.511951 |
| all_models_all_crops_pca256      |   256 | joint_meta_biomass | lightgbm |         0.521726 |      0.611144 |     0.244077 |       0.530697 | 0.635795 |      0.511951 |
| all_models_original_pca256       |   256 | biomass_only       | lightgbm |         0.515142 |      0.572578 |     0.274399 |       0.500029 | 0.604866 |      0.518936 |
| all_models_original_pca256       |   256 | joint_meta_biomass | lightgbm |         0.515142 |      0.572578 |     0.274399 |       0.500029 | 0.604866 |      0.518936 |
| all_models_highveg_pca256        |   256 | biomass_only       | lightgbm |         0.503664 |      0.540616 |     0.204459 |       0.436816 | 0.614532 |      0.525136 |
| all_models_highveg_pca256        |   256 | joint_meta_biomass | lightgbm |         0.503664 |      0.540616 |     0.204459 |       0.436816 | 0.614532 |      0.525136 |
| all_models_center_pca256         |   256 | biomass_only       | lightgbm |         0.50165  |      0.546312 |     0.199906 |       0.487491 | 0.587865 |      0.521412 |
| all_models_center_pca256         |   256 | joint_meta_biomass | lightgbm |         0.50165  |      0.546312 |     0.199906 |       0.487491 | 0.587865 |      0.521412 |
| all_models_highveg_pca256        |   256 | biomass_only       | xgboost  |         0.501645 |      0.532603 |     0.213501 |       0.391825 | 0.598551 |      0.536284 |
| all_models_highveg_pca256        |   256 | joint_meta_biomass | xgboost  |         0.501645 |      0.532603 |     0.213501 |       0.391825 | 0.598551 |      0.536284 |
| convnextv2_base_all_crops_pca256 |   256 | biomass_only       | xgboost  |         0.494744 |      0.517809 |     0.202367 |       0.337847 | 0.568035 |      0.55067  |
| convnextv2_base_all_crops_pca256 |   256 | joint_meta_biomass | xgboost  |         0.494744 |      0.517809 |     0.202367 |       0.337847 | 0.568035 |      0.55067  |
| siglip_base_all_crops_pca256     |   256 | joint_meta_biomass | xgboost  |         0.491779 |      0.546692 |     0.151389 |       0.407243 | 0.533376 |      0.549144 |
| siglip_base_all_crops_pca256     |   256 | biomass_only       | xgboost  |         0.491779 |      0.546692 |     0.151389 |       0.407243 | 0.533376 |      0.549144 |

## PCA-Compressed Fusion
| feature_set                      | metadata_source          |   dim | model    |   cv_weighted_r2 |   Dry_Green_g |   Dry_Dead_g |   Dry_Clover_g |    GDM_g |   Dry_Total_g |
|:---------------------------------|:-------------------------|------:|:---------|-----------------:|--------------:|-------------:|---------------:|---------:|--------------:|
| all_models_all_crops_pca256      | all_models_center_pca256 |   258 | xgboost  |         0.540323 |      0.606063 |     0.241815 |       0.487741 | 0.605648 |      0.571263 |
| all_models_original_pca256       | all_models_center_pca256 |   258 | xgboost  |         0.528716 |      0.614132 |     0.293469 |       0.435343 | 0.619088 |      0.541209 |
| all_models_all_crops_pca256      | all_models_center_pca256 |   258 | lightgbm |         0.52517  |      0.597685 |     0.234163 |       0.519155 | 0.628208 |      0.528856 |
| all_models_center_pca256         | all_models_center_pca256 |   258 | xgboost  |         0.511987 |      0.590679 |     0.186931 |       0.457167 | 0.577116 |      0.546173 |
| all_models_original_pca256       | all_models_center_pca256 |   258 | lightgbm |         0.509555 |      0.604095 |     0.280631 |       0.465311 | 0.608153 |      0.505841 |
| all_models_highveg_pca256        | all_models_center_pca256 |   258 | lightgbm |         0.508364 |      0.567697 |     0.210753 |       0.415964 | 0.617848 |      0.530706 |
| dinov2_large_all_crops_pca256    | all_models_center_pca256 |   258 | xgboost  |         0.506327 |      0.582746 |     0.252774 |       0.404819 | 0.599721 |      0.524698 |
| all_models_highveg_pca256        | all_models_center_pca256 |   258 | xgboost  |         0.504707 |      0.544644 |     0.1952   |       0.379311 | 0.618703 |      0.538103 |
| convnextv2_base_all_crops_pca256 | all_models_center_pca256 |   258 | xgboost  |         0.493253 |      0.548167 |     0.203489 |       0.342479 | 0.58745  |      0.532698 |
| all_models_center_pca256         | all_models_center_pca256 |   258 | lightgbm |         0.491972 |      0.553417 |     0.179122 |       0.470814 | 0.563019 |      0.518065 |
| convnextv2_base_all_crops_pca256 | all_models_center_pca256 |   258 | lightgbm |         0.490148 |      0.519266 |     0.222235 |       0.373217 | 0.574658 |      0.527489 |
| siglip_base_all_crops_pca256     | all_models_center_pca256 |   258 | xgboost  |         0.487336 |      0.555942 |     0.140511 |       0.368622 | 0.560331 |      0.537525 |
| all_models_all_crops_pca256      | all_models_center_pca256 |   258 | catboost |         0.483215 |      0.523816 |     0.203736 |       0.272592 | 0.579057 |      0.534778 |
| siglip_base_all_crops_pca256     | all_models_center_pca256 |   258 | lightgbm |         0.482038 |      0.554023 |     0.134589 |       0.457202 | 0.550831 |      0.51458  |
| dinov2_large_all_crops_pca256    | all_models_center_pca256 |   258 | lightgbm |         0.479015 |      0.584781 |     0.209514 |       0.372167 | 0.589854 |      0.488796 |
| all_models_original_pca256       | all_models_center_pca256 |   258 | catboost |         0.46029  |      0.50524  |     0.208433 |       0.258034 | 0.552889 |      0.505083 |
| all_models_canopy_pca256         | all_models_center_pca256 |   258 | lightgbm |         0.458621 |      0.502884 |     0.124975 |       0.556949 | 0.570187 |      0.452207 |
| all_models_canopy_pca256         | all_models_center_pca256 |   258 | xgboost  |         0.448416 |      0.513588 |     0.127456 |       0.504096 | 0.555005 |      0.445803 |
| all_models_highveg_pca256        | all_models_center_pca256 |   258 | catboost |         0.442544 |      0.475951 |     0.164918 |       0.254176 | 0.530851 |      0.49374  |
| all_models_center_pca256         | all_models_center_pca256 |   258 | catboost |         0.437233 |      0.482337 |     0.155808 |       0.238898 | 0.518719 |      0.491569 |

## Raw Fusion With Predicted NDVI/Height
| feature_set              | model    |   dim |   cv_weighted_r2 |   Dry_Green_g |   Dry_Dead_g |   Dry_Clover_g |    GDM_g |   Dry_Total_g |
|:-------------------------|:---------|------:|-----------------:|--------------:|-------------:|---------------:|---------:|--------------:|
| raw_all_models_original  | xgboost  |  2818 |         0.608336 |      0.649443 |     0.417719 |       0.563076 | 0.691813 |      0.6139   |
| raw_all_models_all_crops | xgboost  | 11266 |         0.6072   |      0.628539 |     0.392526 |       0.57125  | 0.678161 |      0.624673 |
| raw_all_models_original  | lightgbm |  2818 |         0.597237 |      0.611095 |     0.442211 |       0.599825 | 0.668081 |      0.596615 |
| raw_all_models_all_crops | lightgbm | 11266 |         0.595523 |      0.619582 |     0.39765  |       0.584665 | 0.657327 |      0.607735 |
| raw_all_models_highveg   | xgboost  |  2818 |         0.589181 |      0.599728 |     0.376139 |       0.573908 | 0.660272 |      0.604298 |
| raw_all_models_highveg   | lightgbm |  2818 |         0.572882 |      0.582532 |     0.374308 |       0.587177 | 0.610331 |      0.592829 |

## CatBoost Note
CatBoost on raw all-crop concatenated embeddings failed with a memory allocation error. CatBoost results are available for PCA-compressed features only.

## Success Criteria
- Beat baseline `0.596703`: yes
- Reach `0.63+`: no
- Best raw CV: `0.608336`

## Interpretation
Raw all-model original embeddings plus predicted NDVI/Height beat the previous pseudo-metadata baseline. Canopy/high-vegetation crops did not improve CV in this first implementation; the crop algorithm may be too coarse, and PCA-compressed multi-crop features underperform raw embeddings.