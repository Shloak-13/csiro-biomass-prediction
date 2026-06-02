# Kaggle GPU Execution Plan

Current deployable baseline: `0.608336`

Teacher upper bound with unavailable metadata: `0.716104`

The goal is to spend the next GPU run on the experiment most likely to convert image evidence into the missing NDVI/height signal without repeating completed embedding, canopy, or metadata-only work.

## Automation Created

Scripts:

- `scripts/kaggle/create_kernel.py`
- `scripts/kaggle/push_kernel.py`
- `scripts/kaggle/monitor_kernel.py`
- `scripts/kaggle/download_outputs.py`
- `scripts/kaggle/run_experiment.py`

Notebook templates:

- `notebooks/kaggle_templates/kaggle_multitask_training.ipynb`
- `notebooks/kaggle_templates/kaggle_distillation_training.ipynb`
- `notebooks/kaggle_templates/kaggle_ensemble_training.ipynb`
- `notebooks/kaggle_templates/kaggle_submission.ipynb`

Registry:

- `experiments/registry.csv`

Generated Kaggle packages are written under:

- `kaggle_kernels/<kernel_slug>/`

Downloaded outputs are written under:

- `kaggle_outputs/<experiment_name>/<timestamp>/`

## Candidate Ranking

| Rank | Experiment | Expected CV | Expected gain vs 0.608336 | Complexity | Runtime | GPU memory | Recommendation |
|---:|---|---:|---:|---|---|---|---|
| 1 | Multitask NDVI/Height Learning | 0.620-0.650 | +0.012 to +0.042 | Medium | 2-4h | 8-12 GB | Run first |
| 2 | Teacher Distillation | 0.615-0.645 | +0.007 to +0.037 | Medium | 2-4h | 8-12 GB | Run second if multitask helps |
| 3 | OOF Stacking | 0.615-0.635 | +0.007 to +0.027 | Low | 10-30m | 2-4 GB | Run after at least one strong GPU OOF |
| 4 | Per-Target Specialist Models | 0.615-0.640 | +0.007 to +0.032 | High | 5-10h | 8-16 GB | Defer until one shared model is validated |
| 5 | End-to-End DINOv2 Fine-Tuning | 0.625-0.660 | +0.017 to +0.052 | Very high | 6-12h | 16-24 GB | High ceiling, poor first-run ROI |

## First Experiment

Run exactly one GPU experiment first:

`multitask_ndvi_height_v1`

Why this is highest ROI:

- Metadata is unavailable in test, but local image to NDVI/height prediction was strong.
- Pseudo NDVI/height already moved image features toward `0.608336`.
- A GPU image model with auxiliary NDVI and height heads directly attacks the gap between deployable image models and the metadata teacher.
- It is lower risk and cheaper than DINOv2 fine-tuning.
- It produces OOF predictions and checkpoints that can feed later distillation, stacking, and submission notebooks.

Default configuration:

- Backbone: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- Image size: `384`
- Folds: `5`
- Epochs: `12`
- Batch size: `16`
- Loss: weighted biomass MSE plus auxiliary NDVI/height SmoothL1 loss
- Outputs: fold checkpoints, OOF predictions, training log, metrics, submission

Expected runtime:

- Kaggle T4: 3-4h
- Kaggle P100: 2-3h
- Kaggle L4: 1.5-2.5h

Expected GPU memory:

- About 8-12 GB at 384px and batch size 16.
- If out of memory, reduce `BATCH_SIZE` to 8 before reducing image size.

## Exact Commands

Run from:

```powershell
cd D:\csiro\CSIRO-Biomass
```

Set your Kaggle owner once:

```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
```

Create the kernel package:

```powershell
python scripts\kaggle\create_kernel.py --experiment-name multitask_ndvi_height_v1 --template kaggle_multitask_training.ipynb --owner $env:KAGGLE_USERNAME --kernel-slug csiro-multitask-ndvi-height-v1 --title "CSIRO Multitask NDVI Height V1" --internet --force
```

Push and launch the Kaggle run:

```powershell
python scripts\kaggle\push_kernel.py --package-dir kaggle_kernels\csiro-multitask-ndvi-height-v1
```

Monitor execution:

```powershell
python scripts\kaggle\monitor_kernel.py --kernel "$env:KAGGLE_USERNAME/csiro-multitask-ndvi-height-v1" --interval 120 --timeout-minutes 720
```

Download outputs:

```powershell
python scripts\kaggle\download_outputs.py --kernel "$env:KAGGLE_USERNAME/csiro-multitask-ndvi-height-v1" --experiment-name multitask_ndvi_height_v1 --force
```

Single-command orchestration:

```powershell
python scripts\kaggle\run_experiment.py --experiment-name multitask_ndvi_height_v1 --template kaggle_multitask_training.ipynb --owner $env:KAGGLE_USERNAME --kernel-slug csiro-multitask-ndvi-height-v1 --title "CSIRO Multitask NDVI Height V1" --internet --force --monitor --download
```

## Failure Recovery

If kernel creation fails:

- Check that the template exists in `notebooks/kaggle_templates/`.
- Re-run `create_kernel.py` with `--force`.

If push fails:

- Run `kaggle kernels list --mine` to verify authentication.
- Confirm `kernel-metadata.json` exists in `kaggle_kernels/csiro-multitask-ndvi-height-v1/`.
- Re-run `push_kernel.py`.

If the run starts without GPU:

- Stop the run from Kaggle UI.
- Recreate the package and confirm `enable_gpu` is `true` in `kernel-metadata.json`.
- Re-push when GPU quota is available.

If the run hits GPU OOM:

- Recreate the kernel with environment variables in the notebook or edit config defaults:
  - `BATCH_SIZE=8`
  - keep `IMAGE_SIZE=384`
  - if still failing, use `IMAGE_SIZE=320`

If the run exceeds time:

- First reduce `EPOCHS` from `12` to `8`.
- If still too slow, use `N_FOLDS=3` for a fast signal, then re-run full 5-fold only if CV improves.

If output download fails:

```powershell
kaggle kernels status "$env:KAGGLE_USERNAME/csiro-multitask-ndvi-height-v1"
kaggle kernels output "$env:KAGGLE_USERNAME/csiro-multitask-ndvi-height-v1" -p kaggle_outputs\multitask_ndvi_height_v1\manual --force
```

## Expected Outputs

After download, inspect:

- `kaggle_outputs/multitask_ndvi_height_v1/<timestamp>/metrics.json`
- `kaggle_outputs/multitask_ndvi_height_v1/<timestamp>/oof_predictions.csv`
- `kaggle_outputs/multitask_ndvi_height_v1/<timestamp>/training_log.csv`
- `kaggle_outputs/multitask_ndvi_height_v1/<timestamp>/submission.csv`
- `kaggle_outputs/multitask_ndvi_height_v1/<timestamp>/checkpoints/fold*.pt`

## Decision Rule

If `cv_weighted_r2 >= 0.620`, continue with teacher distillation using the same backbone and image size.

If `cv_weighted_r2 >= 0.630`, immediately create an OOF ensemble with the current best local model and this GPU model.

If `cv_weighted_r2 < 0.608`, do not spend more GPU on the same architecture. Switch to teacher distillation or DINOv2 fine-tuning only after error analysis.

## Recommended Next Experiment After Completion

If the first run improves CV:

`teacher_distillation_v1`

Command:

```powershell
python scripts\kaggle\run_experiment.py --experiment-name teacher_distillation_v1 --template kaggle_distillation_training.ipynb --owner $env:KAGGLE_USERNAME --kernel-slug csiro-teacher-distillation-v1 --title "CSIRO Teacher Distillation V1" --internet --force --monitor --download
```

If the first run does not improve CV:

Run OOF stacking only after attaching the first run outputs and the best local model outputs as Kaggle datasets. Do not proceed to expensive DINOv2 fine-tuning until the multitask failure mode is understood.
