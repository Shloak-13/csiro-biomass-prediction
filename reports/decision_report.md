# Decision Report

## Answers
1. Image information adds value beyond metadata: not yet in this benchmark.
2. CV gain from the best fused model: `-0.038692`.
3. Multimodal fusion is not justified yet without better embeddings/crops based on this benchmark.
4. Highest ROI model now: `lightgbm` with `A_metadata_only` / `none` at CV `0.716104`.
5. Train next: tune the best metadata/constraint model first, then retry full embeddings with canopy-aware crops if fusion is not positive.

## Metadata Importance / Leakage Check
High NDVI/height SHAP importance means they are strong proxies, but not proof of leakage. They are provided in train metadata and absent from the sample test metadata here, so deployment/test schema must be verified before relying on them.

| feature                                                             |   mean_abs_shap |
|:--------------------------------------------------------------------|----------------:|
| ndvi_x_height                                                       |      0.277617   |
| height_per_ndvi                                                     |      0.159999   |
| dayofyear                                                           |      0.154151   |
| Pre_GSHH_NDVI                                                       |      0.137849   |
| Species_Clover                                                      |      0.105516   |
| State_WA                                                            |      0.0784214  |
| Height_Ave_cm                                                       |      0.0741396  |
| month_cos                                                           |      0.0700609  |
| month                                                               |      0.0615753  |
| Species_Ryegrass                                                    |      0.0465308  |
| State_Tas                                                           |      0.0366872  |
| Species_Lucerne                                                     |      0.0310537  |
| Species_Fescue                                                      |      0.029909   |
| State_Vic                                                           |      0.0276832  |
| State_NSW                                                           |      0.0262776  |
| Species_Ryegrass_Clover                                             |      0.0253396  |
| month_sin                                                           |      0.0241974  |
| Species_WhiteClover                                                 |      0.0127276  |
| Species_Phalaris_Clover                                             |      0.0113847  |
| season_winter                                                       |      0.00923565 |
| season_autumn                                                       |      0.00762715 |
| Species_Fescue_CrumbWeed                                            |      0.00358682 |
| season_spring                                                       |      0.00329444 |
| season_summer                                                       |      0.0017817  |
| Species_Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed |      0.00127258 |
| Species_SubcloverLosa                                               |      0          |
| Species_SubcloverDalkeith                                           |      0          |
| Species_Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass             |      0          |
| Species_Phalaris_Ryegrass_Clover                                    |      0          |
| Species_Mixed                                                       |      0          |

## Top 5 Next Experiments
| Rank | Experiment | Rationale |
|---:|:---|:---|
| 1 | Hyperparameter tune metadata LightGBM/CatBoost/XGBoost with target constraints | Highest current CV and fastest iteration |
| 2 | Validate `predict_4_derive_total` plus GDM consistency across folds | Biomass identity is almost exact and high weight sits on total |
| 3 | Full embedding fusion with canopy crop/resize variants | Current raw embeddings may underuse 2000x1000 plot detail |
| 4 | OOF weighted ensemble of metadata, constrained, and best embedding/fusion models | Low effort, usually robust |
| 5 | Group/date/state validation stress test and residual calibration | Fold instability implies distribution shift risk |