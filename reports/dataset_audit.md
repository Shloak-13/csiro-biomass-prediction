# Dataset Audit

## Dimensions
- `train.csv`: 1785 rows x 9 columns
- `test.csv`: 5 rows x 3 columns
- `sample_submission.csv`: 5 rows x 2 columns
- Unique train images: 357
- Unique test images: 1
- Number of targets: 5 (Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g)

## Missing Values
|               |   missing_train |
|:--------------|----------------:|
| sample_id     |               0 |
| image_path    |               0 |
| Sampling_Date |               0 |
| State         |               0 |
| Species       |               0 |
| Pre_GSHH_NDVI |               0 |
| Height_Ave_cm |               0 |
| target_name   |               0 |
| target        |               0 |

### Test Missing Values
|             |   missing_test |
|:------------|---------------:|
| sample_id   |              0 |
| image_path  |              0 |
| target_name |              0 |

## Duplicate Rows and Images
- Full duplicate train rows: 0
- Duplicate perceptual hashes: 0 rows
- Near-duplicate pairs with pHash distance <= 4: 0

## Image Resolutions and Aspect Ratios
|       |   width |   height |   aspect_ratio |
|:------|--------:|---------:|---------------:|
| count |     357 |      357 |            357 |
| mean  |    2000 |     1000 |              2 |
| std   |       0 |        0 |              0 |
| min   |    2000 |     1000 |              2 |
| 25%   |    2000 |     1000 |              2 |
| 50%   |    2000 |     1000 |              2 |
| 75%   |    2000 |     1000 |              2 |
| max   |    2000 |     1000 |              2 |

## Date Distribution
| Sampling_Date   |   count |
|:----------------|--------:|
| 2015/1/15       |      17 |
| 2015/10/13      |      11 |
| 2015/10/14      |       7 |
| 2015/10/6       |      11 |
| 2015/11/10      |      17 |
| 2015/11/9       |      20 |
| 2015/2/24       |      15 |
| 2015/2/25       |       9 |
| 2015/4/1        |      10 |
| 2015/5/18       |      22 |
| 2015/5/19       |       7 |
| 2015/5/7        |      13 |
| 2015/6/26       |      37 |
| 2015/6/29       |      10 |
| 2015/6/30       |       6 |
| 2015/7/1        |      19 |
| 2015/7/2        |      10 |
| 2015/7/8        |      12 |
| 2015/8/14       |      18 |
| 2015/8/18       |      10 |
| 2015/8/19       |       1 |
| 2015/8/21       |       8 |
| 2015/9/1        |      12 |
| 2015/9/11       |       9 |
| 2015/9/29       |      11 |
| 2015/9/3        |       9 |
| 2015/9/30       |      10 |
| 2015/9/4        |      16 |

## State Distribution
| State   |   count |
|:--------|--------:|
| Tas     |     138 |
| Vic     |     112 |
| NSW     |      75 |
| WA      |      32 |

## Species Distribution
| Species                                                     |   count |
|:------------------------------------------------------------|--------:|
| Ryegrass_Clover                                             |      98 |
| Ryegrass                                                    |      62 |
| Phalaris_Clover                                             |      42 |
| Clover                                                      |      41 |
| Fescue                                                      |      28 |
| Lucerne                                                     |      22 |
| Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed |      11 |
| Fescue_CrumbWeed                                            |      10 |
| WhiteClover                                                 |      10 |
| Phalaris_Ryegrass_Clover                                    |       8 |
| Phalaris                                                    |       8 |
| Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass             |       7 |
| SubcloverLosa                                               |       5 |
| SubcloverDalkeith                                           |       3 |
| Mixed                                                       |       2 |

## Target Distributions
|              |   count |     mean |     std |   min |     1% |       5% |     50% |      95% |      99% |      max |
|:-------------|--------:|---------:|--------:|------:|-------:|---------:|--------:|---------:|---------:|---------:|
| Dry_Green_g  |     357 | 26.6247  | 25.4012 |  0    | 0      |  0.12544 | 20.8    |  80.0672 | 114.561  | 157.984  |
| Dry_Dead_g   |     357 | 12.0445  | 12.402  |  0    | 0      |  0       |  7.9809 |  36.2389 |  53.0754 |  83.8407 |
| Dry_Clover_g |     357 |  6.64969 | 12.1178 |  0    | 0      |  0       |  1.4235 |  33.1108 |  58.5776 |  71.7865 |
| GDM_g        |     357 | 33.2744  | 24.9358 |  1.04 | 2.4448 |  5.97778 | 27.1082 |  83.1474 | 114.561  | 157.984  |
| Dry_Total_g  |     357 | 45.3181  | 27.984  |  1.04 | 5.456  | 10.6817  | 40.3    | 105.42   | 129.888  | 185.7    |

## Target Correlations
|              |   Dry_Green_g |   Dry_Dead_g |   Dry_Clover_g |    GDM_g |   Dry_Total_g |
|:-------------|--------------:|-------------:|---------------:|---------:|--------------:|
| Dry_Green_g  |     1         |    0.0955537 |      -0.276582 | 0.884257 |      0.830315 |
| Dry_Dead_g   |     0.0955537 |    1         |      -0.175548 | 0.012028 |      0.453912 |
| Dry_Clover_g |    -0.276582  |   -0.175548  |       1        | 0.204213 |      0.104185 |
| GDM_g        |     0.884257  |    0.012028  |       0.204213 | 1        |      0.896441 |
| Dry_Total_g  |     0.830315  |    0.453912  |       0.104185 | 0.896441 |      1        |