# Highest-ROI Strategy Before Training

The Kaggle API currently returns `401 Unauthorized`, so the real CSV/image audit must run after Kaggle credentials are configured and the competition rules are accepted. These are the approaches with the highest expected leaderboard return once data is accessible.

1. **Validation that mirrors hidden distribution**
   - Compare stratified biomass folds, state/date grouped folds, and date-holdout folds.
   - Pick the strategy whose out-of-fold errors are stable by `State`, `Species`, and `Sampling_Date`.
   - Reason: in small agricultural image competitions, leaderboard overfit usually comes from random fold leakage through site, date, or near-duplicate plots.

2. **Target-structure modeling and post-processing**
   - Audit whether `Dry_Total_g ~= Dry_Green_g + Dry_Dead_g + Dry_Clover_g` and whether `GDM_g ~= Dry_Green_g + Dry_Clover_g`.
   - Train unconstrained models, soft-constrained models, and total-derived post-processing.
   - Reason: `Dry_Total_g` has 0.5 metric weight, so even small consistency gains can dominate.

3. **Frozen foundation embeddings plus gradient boosting**
   - Extract DINOv2, SigLIP, CLIP, ConvNeXtV2, and EVA-02 embeddings.
   - Train CatBoost/LightGBM/XGBoost/MLP on embeddings plus metadata.
   - Reason: this usually gives a strong score quickly, is less GPU-hungry than full fine-tuning, and ensembling diverse encoders is high leverage.

4. **Multimodal end-to-end model with metadata fusion**
   - Fine-tune ConvNeXtV2/EVA/Swin with image branch plus metadata branch.
   - Compare concat, gated fusion, attention fusion, and cross-attention.
   - Reason: image texture and canopy cover explain biomass, while NDVI, height, state, species, and season correct systematic bias.

5. **Diversity-first ensemble**
   - Blend metadata-only, frozen embeddings, end-to-end vision, multimodal, and constrained variants.
   - Optimize weights on honest OOF predictions, with per-target weights aligned to the competition metric.
   - Reason: private leaderboard robustness usually comes from validation-respecting diversity, not one maximal public-LB model.

External data should be treated as a second wave: weather, soil moisture, Sentinel/Landsat/MODIS vegetation products, and Australian environmental datasets may help, but only if location/date resolution is specific enough and leakage risk is documented.
