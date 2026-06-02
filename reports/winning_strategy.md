# Winning Strategy

## Core Findings
- Best metadata-only CV weighted R2 observed: `0.71102`
- Best image-only hand-crafted-feature CV weighted R2 observed: `0.39257`
- Best metadata + hand-crafted-image-feature CV weighted R2 observed: `0.67044`
- Best subset embedding CV weighted R2 observed: `0.36231`
- Biological total constraint is strongly justified based on median absolute residual.
- Current answer to the research questions: metadata dominates; subset embeddings do not dominate; naive metadata+image-stat fusion does not help; biological constraints should help; cropping/preprocessing should be investigated before end-to-end CNN training.

## Ranked Top 10 Approaches
| Rank | Approach | Expected CV score | Complexity | Training time | GPU requirements | Expected leaderboard gain |
|---:|:---|---:|:---|:---|:---|:---|
| 1 | Strong metadata LightGBM/CatBoost with fold audit and hyperparameter search | 0.72-0.76 | Low | 15-45 min | No | Highest immediate gain; metadata is dominant so far |
| 2 | Constraint-aware multi-target post-processing for `GDM_g` and `Dry_Total_g` | 0.72-0.77 | Low | minutes | No | High ROI because train identity is nearly exact |
| 3 | Date/state/species grouped validation and leakage audit | score stabilizer | Medium | 30-60 min | No | Prevents public-LB overfit on only 357 images |
| 4 | Full-train DINOv2/SigLIP/CLIP embeddings + metadata fusion | 0.73-0.78 | Medium | 1-3 h CPU, faster GPU | Helpful | Best next image route; current subset embeddings add diversity but do not dominate |
| 5 | OOF weighted ensemble of metadata, constrained, embedding, and image-stat models | 0.74-0.79 | Low | minutes after OOFs | No | Usually improves private robustness |
| 6 | Canopy-aware crop/resize preprocessing before embeddings | 0.73-0.79 | Medium | 1-3 h | Optional | Likely useful because all images are 2000x1000 and vegetation pixels correlate with GDM |
| 7 | Per-target specialized heads/models, especially clover and dead biomass | 0.73-0.78 | Medium | 1-2 h | No | Targets have different drivers; clover is weakly tied to total |
| 8 | Residual calibration by state/date/species | 0.73-0.78 | Medium | 30-90 min | No | Fold 0 is much weaker, indicating distribution sensitivity |
| 9 | End-to-end ConvNeXtV2/EVA fine-tuning after embedding proof | 0.74-0.82 | High | 4-12 h | Yes | Potentially high, but not first because metadata currently beats image-only routes |
| 10 | External weather/satellite joins | 0.72-0.80 | High | 4-10 h | No | Uncertain without exact location; leakage/license risk must be handled |

## Recommended Next Experiment

Run the multimodal frozen-embedding experiment on all train images, then blend with metadata:

```powershell
cd D:\csiro\CSIRO-Biomass
& 'C:\Users\sapna shetty\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m src.datasets.research_phase --data-dir data\csiro-biomass --embedding-subset 357
```
