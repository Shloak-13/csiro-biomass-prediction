# Inference Metadata Audit

## Local Competition Files
- `train.csv` columns: `['sample_id', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm', 'target_name', 'target']`
- `test.csv` columns: `['sample_id', 'image_path', 'target_name']`
- `sample_submission.csv` columns: `['sample_id', 'target']`
- Local test images: `1` image, `5` target rows.

## Conclusion
Metadata is not available in the provided inference schema. The public `test.csv` contains only `sample_id`, `image_path`, and `target_name`; no `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, or `Sampling_Date` columns are present.

Kaggle hidden test rows normally follow the same schema as public `test.csv` and `sample_submission.csv`. There is no local file, sidecar metadata table, or sample submission field that would reconstruct metadata at scoring time.

## Answers
1. Are metadata fields available in hidden test? Unknown directly, but the evidence strongly indicates no: hidden test should match `test.csv` schema.
2. Are metadata fields reconstructed by Kaggle? No evidence. The submission interface only asks for `sample_id,target`.
3. Are metadata fields intentionally withheld? Yes, operationally: train has metadata, test does not.
4. Are metadata features allowed at inference? Only if derived from allowed inputs. Raw train metadata columns cannot be used for test rows because they are absent.

## Strategic Implication
Treat metadata-only CV as an optimistic upper bound or teacher signal, not a directly submit-ready model. Submission-ready models must use images and `target_name`, or pseudo-metadata predicted from images.